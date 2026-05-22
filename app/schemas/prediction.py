from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    # Demografis
    age: float = Field(..., ge=1, le=120, description="Usia pasien (tahun)")

    # Vital sign
    systolic_bp: float = Field(..., ge=60, le=250, description="Tekanan darah sistolik (mmHg)")
    diastolic_bp: float = Field(..., ge=40, le=160, description="Tekanan darah diastolik (mmHg)")
    resting_heart_rate: float = Field(..., ge=30, le=220, description="Detak jantung istirahat (bpm)")
    cholesterol_mg_dl: float = Field(..., ge=100, le=600, description="Kadar kolesterol (mg/dL)")

    # Antropometri
    bmi: float = Field(..., ge=10, le=70, description="Body Mass Index")

    # Gaya hidup
    daily_steps: float = Field(..., ge=0, description="Jumlah langkah per hari")
    physical_activity_hours_per_week: float = Field(..., ge=0, le=24, description="Jam olahraga per minggu")
    sleep_hours: float = Field(..., ge=0, le=24, description="Jam tidur per malam")
    alcohol_units_per_week: float = Field(..., ge=0, description="Unit alkohol per minggu")
    stress_level: float = Field(..., ge=1, le=10, description="Tingkat stres (1–10)")
    diet_quality_score: float = Field(..., ge=1, le=10, description="Skor kualitas diet (1–10)")
    smoking_status: int = Field(..., ge=0, le=2, description="0=tidak merokok, 1=merokok, 2=mantan perokok")

    # Riwayat
    family_history_heart_disease: bool = Field(False, description="Ada riwayat penyakit jantung dalam keluarga")

    # Fitur turunan (opsional — dihitung otomatis kalau tidak dikirim)
    age_group: Optional[int] = Field(None, ge=1, le=6, description="1=remaja, 2=dewasa muda, 3=dewasa, 4=paruh baya, 5=lansia, 6=sangat tua")
    bmi_category: Optional[int] = Field(None, ge=1, le=4, description="1=kurus, 2=normal, 3=gemuk, 4=obese")
    hypertension_stage: Optional[int] = Field(None, ge=1, le=3, description="1=normal, 2=elevasi, 3=hipertensi")
    cholesterol_category: Optional[int] = Field(None, ge=0, le=2, description="0=optimal, 1=borderline, 2=tinggi")
    activity_level: Optional[int] = Field(None, ge=0, le=3, description="0=sedentary, 1=ringan, 2=sedang, 3=aktif")
    sleep_category: Optional[int] = Field(None, ge=0, le=2, description="0=kurang, 1=cukup, 2=lebih")
    alcohol_category: Optional[int] = Field(None, ge=0, le=3, description="0=tidak, 1=rendah, 2=sedang, 3=tinggi")
    pulse_pressure: Optional[float] = Field(None, ge=0, description="systolic_bp - diastolic_bp")
    blood_pressure_ratio: Optional[float] = Field(None, ge=0, description="systolic_bp / diastolic_bp")
    lifestyle_risk_score: Optional[int] = Field(None, ge=0, le=6, description="Skor risiko gaya hidup (0–6)")
    clinical_risk_score: Optional[int] = Field(None, ge=0, le=4, description="Skor risiko klinis (0–4)")
    heart_disease_risk_score: Optional[float] = Field(None, ge=0, le=100, description="Skor risiko kontinu (0–100)")


class RecommendationItem(BaseModel):
    parameter: str
    kondisi: str
    rekomendasi: list[str]
    sumber: str


class RiskComparisonItem(BaseModel):
    narasi: str


class PredictionResponse(BaseModel):
    risk_category: str    # "Low" | "Medium" | "High"
    risk_score: float     # 0–100
    confidence: float     # 0–1

    recommendations: dict[str, list] = Field(
        description="{'urgent': [...], 'warning': [...], 'good': [...]}"
    )
    risk_comparison: list[str]
