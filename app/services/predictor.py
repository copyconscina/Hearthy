"""
HearthyPredictor: load model TensorFlow sekali, inferensi per request.
Setelah prediksi, hasil langsung dikirim ke GeminiRecommender untuk
menghasilkan rekomendasi aktivitas personal.
"""
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.gemini_recommender import GeminiRecommender


# ── Custom Loss ──────────────────────────────────────────────────────────────

class MinorityAwareLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, class_weights=None,
                 name='minority_aware_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        if class_weights is None:
            self.class_weights = tf.constant([1.293, 0.998, 0.817], dtype=tf.float32)
        else:
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_pred    = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        n_classes = tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), n_classes)
        y_true_oh = tf.cast(y_true_oh, tf.float32)
        p_t       = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        ce        = -tf.math.log(p_t)
        focal_w   = tf.pow(1.0 - p_t, self.gamma)
        sample_weights = tf.reduce_sum(y_true_oh * self.class_weights, axis=-1)
        return sample_weights * focal_w * ce

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'class_weights': self.class_weights.numpy().tolist(),
        })
        return config


# ── Custom Layer ─────────────────────────────────────────────────────────────

class FeatureAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, units, reduction_ratio=4, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units           = units
        self.reduction_ratio = reduction_ratio
        self.dropout_rate    = dropout_rate
        self.dense_main = layers.Dense(units, activation='gelu')
        self.layer_norm = layers.LayerNormalization()
        self.dropout    = layers.Dropout(dropout_rate)
        se_units        = max(units // reduction_ratio, 8)
        self.se_squeeze = layers.Dense(se_units, activation='relu')
        self.se_excite  = layers.Dense(units,    activation='sigmoid')
        self.proj       = layers.Dense(units, use_bias=False)

    def build(self, input_shape):
        self.dense_main.build(input_shape)
        self.proj.build(input_shape)
        dense_main_shape = (input_shape[0], self.units)
        self.layer_norm.build(dense_main_shape)
        se_units = max(self.units // self.reduction_ratio, 8)
        self.se_squeeze.build(dense_main_shape)
        se_shape = (input_shape[0], se_units)
        self.se_excite.build(se_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x        = self.dense_main(inputs)
        x        = self.layer_norm(x)
        x        = self.dropout(x, training=training)
        attn     = self.se_squeeze(x)
        attn     = self.se_excite(attn)
        x        = x * attn
        residual = self.proj(inputs)
        return tf.keras.activations.gelu(x + residual)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units'          : self.units,
            'reduction_ratio': self.reduction_ratio,
            'dropout_rate'   : self.dropout_rate,
        })
        return config


# ── Urutan fitur sesuai training (25 fitur) ──────────────────────────────────
FEATURE_ORDER = [
    "age", "bmi", "systolic_bp", "diastolic_bp", "cholesterol_mg_dl",
    "resting_heart_rate", "smoking_status", "daily_steps", "stress_level",
    "physical_activity_hours_per_week", "sleep_hours", "family_history_heart_disease",
    "diet_quality_score", "alcohol_units_per_week", "age_group", "pulse_pressure",
    "blood_pressure_ratio", "hypertension_stage", "bmi_category", "cholesterol_category",
    "activity_level", "sleep_category", "alcohol_category", "lifestyle_risk_score",
    "clinical_risk_score",
]

SCORE_MAX = 100.0
RISK_MAP  = {0: "Low", 1: "Medium", 2: "High"}


def _derive_features(req: PredictionRequest) -> dict:
    """Hitung semua fitur turunan sesuai feature engineering di notebook."""
    data = req.model_dump()

    age   = data["age"]
    sys   = data["systolic_bp"]
    dia   = data["diastolic_bp"]
    chol  = data["cholesterol_mg_dl"]
    sleep = data["sleep_hours"]
    alc   = data["alcohol_units_per_week"]
    act   = data["physical_activity_hours_per_week"]
    bmi   = data["bmi"]

    if age < 30:   data["age_group"] = 6
    elif age < 40: data["age_group"] = 0
    elif age < 50: data["age_group"] = 1
    elif age < 60: data["age_group"] = 2
    elif age < 70: data["age_group"] = 3
    elif age < 80: data["age_group"] = 4
    else:          data["age_group"] = 5

    data["blood_pressure_ratio"] = round(sys / dia, 6) if dia else 0

    if sys < 120 and dia < 80:   data["hypertension_stage"] = 0
    elif sys < 130 and dia < 80: data["hypertension_stage"] = 1
    elif sys < 140 or dia < 90:  data["hypertension_stage"] = 2
    else:                        data["hypertension_stage"] = 3

    if bmi < 18.5:  data["bmi_category"] = 0
    elif bmi < 25:  data["bmi_category"] = 1
    elif bmi < 30:  data["bmi_category"] = 2
    else:           data["bmi_category"] = 3

    if chol < 200:   data["cholesterol_category"] = 0
    elif chol < 240: data["cholesterol_category"] = 1
    else:            data["cholesterol_category"] = 2

    act_min = act * 60
    if act_min < 75:    data["activity_level"] = 0
    elif act_min < 150: data["activity_level"] = 1
    else:               data["activity_level"] = 2

    if sleep < 6:    data["sleep_category"] = 0
    elif sleep <= 9: data["sleep_category"] = 1
    else:            data["sleep_category"] = 2

    if alc <= 0:   data["alcohol_category"] = 0
    elif alc <= 7: data["alcohol_category"] = 1
    else:          data["alcohol_category"] = 2

    data["pulse_pressure"] = sys - dia

    ls = 0
    if data["smoking_status"] == 2:     ls += 1
    if data["activity_level"] == 0:     ls += 1
    if data["sleep_category"] == 0:     ls += 1
    if data["alcohol_category"] == 2:   ls += 1
    if data["bmi_category"] >= 3:       ls += 1
    if data["stress_level"] >= 7:       ls += 1
    if data["diet_quality_score"] <= 3: ls += 1
    data["lifestyle_risk_score"] = ls

    cs = 0
    if data["hypertension_stage"] >= 2:                  cs += 1
    if data["cholesterol_category"] >= 2:                cs += 1
    if data.get("diabetes", 0) == 1:                     cs += 1
    if int(data.get("family_history_heart_disease", 0)): cs += 1
    if age >= 45:                                        cs += 1
    data["clinical_risk_score"] = cs

    data["family_history_heart_disease"] = int(data.get("family_history_heart_disease", 0))

    return data


class HearthyPredictor:
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        label_encoder_path: str,
        recommender: GeminiRecommender,
    ):
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "FeatureAttentionBlock": FeatureAttentionBlock,
                "MinorityAwareLoss"    : MinorityAwareLoss,
            },
        )
        self.scaler        = joblib.load(scaler_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.recommender   = recommender

    def predict(self, req: PredictionRequest) -> PredictionResponse:
        from app.schemas.prediction import F1Scores

        data = _derive_features(req)

        x        = np.array([[data.get(f, 0) for f in FEATURE_ORDER]], dtype=np.float32)
        x_scaled = self.scaler.transform(x)

        outputs = self.model.predict(x_scaled, verbose=0)
        if isinstance(outputs, list):
            class_probs, score_norm = outputs[0], outputs[1]
        else:
            class_probs = outputs
            score_norm  = None

        class_idx     = int(np.argmax(class_probs, axis=-1)[0])
        confidence    = float(class_probs[0][class_idx])
        risk_category = RISK_MAP.get(class_idx, str(class_idx))

        if score_norm is not None:
            risk_score = float(score_norm[0][0]) * SCORE_MAX
        else:
            risk_score = float(class_idx / 2) * 100

        STATIC_F1 = {
            "low"       : 0.94,
            "medium"    : 0.93,
            "high"      : 0.90,
            "macro_avg" : 0.92,
        }
        f1 = F1Scores(**STATIC_F1)

        patient_data = req.model_dump()
        patient_data["confidence"] = confidence

        recommendations = self.recommender.recommend(
            risk_category=risk_category,
            risk_score=round(risk_score, 1),
            patient_data=patient_data,
        )

        return PredictionResponse(
            risk_category=risk_category,
            risk_score=round(risk_score, 1),
            confidence=round(confidence, 4),
            f1_scores=f1,
            recommendations=recommendations,
        )