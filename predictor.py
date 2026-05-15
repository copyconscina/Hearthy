"""
HearthyPredictor: load model TensorFlow sekali, inferensi per request.
Termasuk custom layer FeatureAttentionBlock agar model bisa di-load ulang.
"""
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.recommender import generate_recommendations, generate_risk_comparison


# ── Custom Layer (harus didefinisikan ulang untuk load model) ────────────────

class FeatureAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, units, reduction_ratio=4, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.reduction_ratio = reduction_ratio
        self.dropout_rate = dropout_rate

        self.dense_main = layers.Dense(units, activation="gelu")
        self.layer_norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout_rate)

        se_units = max(units // reduction_ratio, 8)
        self.se_squeeze = layers.Dense(se_units, activation="relu")
        self.se_excite = layers.Dense(units, activation="sigmoid")
        self.proj = layers.Dense(units, use_bias=False)

    def call(self, inputs, training=False):
        x = self.dense_main(inputs)
        x = self.layer_norm(x)
        x = self.dropout(x, training=training)
        attn = self.se_squeeze(x)
        attn = self.se_excite(attn)
        x = x * attn
        residual = self.proj(inputs)
        return tf.keras.activations.gelu(x + residual)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "reduction_ratio": self.reduction_ratio,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ── Urutan fitur sesuai saat training ───────────────────────────────────────
# Sesuaikan dengan df.drop(columns=drop_cols).columns dari notebook
FEATURE_ORDER = [
    "age", "gender", "systolic_bp", "diastolic_bp", "cholesterol_mg_dl",
    "bmi", "bmi_category", "resting_heart_rate", "daily_steps",
    "physical_activity_hours_per_week", "activity_level", "sleep_hours",
    "alcohol_units_per_week", "stress_level", "diet_quality_score",
    "smoking_status", "diabetes", "hypertension",
    "family_history_heart_disease",
    "bp_category", "pulse_pressure", "cardiovascular_age",
    # tambahkan fitur turunan lain sesuai dataset asli
]

SCORE_MAX = 100.0


def _derive_features(req: PredictionRequest) -> dict:
    """Hitung fitur turunan yang tidak dikirim dari frontend."""
    data = req.model_dump()

    # BMI category: 0=underweight, 1=normal, 2=overweight, 3=obese
    bmi = data["bmi"]
    if data.get("bmi_category") is None:
        if bmi < 18.5:
            data["bmi_category"] = 0
        elif bmi < 25:
            data["bmi_category"] = 1
        elif bmi < 30:
            data["bmi_category"] = 2
        else:
            data["bmi_category"] = 3

    # Activity level berdasarkan hours/week
    act = data["physical_activity_hours_per_week"]
    if data.get("activity_level") is None:
        if act * 60 < 75:
            data["activity_level"] = 0
        elif act * 60 < 150:
            data["activity_level"] = 1
        else:
            data["activity_level"] = 2

    # BP category
    sys = data["systolic_bp"]
    dia = data["diastolic_bp"]
    if data.get("bp_category") is None:
        if sys < 120 and dia < 80:
            data["bp_category"] = 0
        elif sys < 130 and dia < 80:
            data["bp_category"] = 1
        elif sys < 140 or dia < 90:
            data["bp_category"] = 2
        else:
            data["bp_category"] = 3

    # Pulse pressure
    if data.get("pulse_pressure") is None:
        data["pulse_pressure"] = sys - dia

    # Cardiovascular age (simplified estimate)
    if data.get("cardiovascular_age") is None:
        cv_age = data["age"]
        if data["smoking_status"] == 1:
            cv_age += 5
        if data["hypertension"] == 1:
            cv_age += 3
        if data["diabetes"] == 1:
            cv_age += 4
        if data.get("family_history_heart_disease"):
            cv_age += 2
        data["cardiovascular_age"] = cv_age

    # Convert bool to int
    data["family_history_heart_disease"] = int(data["family_history_heart_disease"])

    return data


class HearthyPredictor:
    def __init__(self, model_path: str, scaler_path: str, label_encoder_path: str):
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={"FeatureAttentionBlock": FeatureAttentionBlock},
        )
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(label_encoder_path)

    def predict(self, req: PredictionRequest) -> PredictionResponse:
        data = _derive_features(req)

        # Build feature vector sesuai urutan training
        x = np.array([[data.get(f, 0) for f in FEATURE_ORDER]], dtype=np.float32)
        x_scaled = self.scaler.transform(x)

        # Inferensi — model punya 2 output: class + score
        outputs = self.model.predict(x_scaled, verbose=0)

        if isinstance(outputs, list):
            class_probs, score_norm = outputs[0], outputs[1]
        else:
            class_probs = outputs
            score_norm = None

        class_idx = int(np.argmax(class_probs, axis=-1)[0])
        confidence = float(class_probs[0][class_idx])
        risk_category = self.label_encoder.inverse_transform([class_idx])[0]

        if score_norm is not None:
            risk_score = float(score_norm[0][0]) * SCORE_MAX
        else:
            # Fallback: skor berdasarkan probabilitas kelas
            risk_score = float(class_idx / (len(self.label_encoder.classes_) - 1)) * 100

        # Rekomendasi
        user_input_dict = {
            "systolic_bp": data["systolic_bp"],
            "diastolic_bp": data["diastolic_bp"],
            "cholesterol_mg_dl": data["cholesterol_mg_dl"],
            "bmi": data["bmi"],
            "resting_heart_rate": data["resting_heart_rate"],
            "daily_steps": data["daily_steps"],
            "physical_activity_hours_per_week": data["physical_activity_hours_per_week"],
            "sleep_hours": data["sleep_hours"],
            "alcohol_units_per_week": data["alcohol_units_per_week"],
            "stress_level": data["stress_level"],
            "diet_quality_score": data["diet_quality_score"],
            "family_history_heart_disease": bool(data["family_history_heart_disease"]),
        }

        recommendations = generate_recommendations(user_input_dict)
        risk_comparison = [r["narasi"] for r in generate_risk_comparison(user_input_dict)]

        return PredictionResponse(
            risk_category=risk_category,
            risk_score=round(risk_score, 1),
            confidence=round(confidence, 4),
            recommendations=recommendations,
            risk_comparison=risk_comparison,
        )
