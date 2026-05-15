from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    # Demografis
    age: float = Field(..., ge=1, le=120, description="Usia pasien (tahun)")
    gender: int = Field(..., ge=0, le=1, description="0=perempuan, 1=laki-laki")

    # Vital sign
    systolic_bp: float = Field(..., ge=60, le=250, description="Tekanan darah sistolik (mmHg)")
    diastolic_bp: float = Field(..., ge=40, le=160, description="Tekanan darah diastolik (mmHg)")
    resting_heart_rate: float = Field(..., ge=30, le=220)
    cholesterol_mg_dl: float = Field(..., ge=100, le=600)

    # Antropometri
    bmi: float = Field(..., ge=10, le=70)

    # Gaya hidup
    daily_steps: float = Field(..., ge=0)
    physical_activity_hours_per_week: float = Field(..., ge=0)
    sleep_hours: float = Field(..., ge=0, le=24)
    alcohol_units_per_week: float = Field(..., ge=0)
    stress_level: float = Field(..., ge=0, le=10)
    diet_quality_score: float = Field(..., ge=0, le=10)
    smoking_status: int = Field(..., ge=0, le=1, description="0=tidak merokok, 1=merokok")

    # Riwayat
    family_history_heart_disease: bool = False
    diabetes: int = Field(0, ge=0, le=1)
    hypertension: int = Field(0, ge=0, le=1)

    # Fitur turunan (opsional — dihitung otomatis kalau tidak dikirim)
    bmi_category: Optional[int] = None
    activity_level: Optional[int] = None
    bp_category: Optional[int] = None
    pulse_pressure: Optional[float] = None
    cardiovascular_age: Optional[float] = None


class RecommendationItem(BaseModel):
    parameter: str
    kondisi: str
    rekomendasi: list[str]
    sumber: str


class RiskComparisonItem(BaseModel):
    narasi: str


class PredictionResponse(BaseModel):
    risk_category: str          # "Low" | "Medium" | "High"
    risk_score: float           # 0–100
    confidence: float           # probabilitas kelas prediksi (0–1)

    recommendations: dict[str, list] = Field(
        description="{'urgent': [...], 'warning': [...], 'good': [...]}"
    )
    risk_comparison: list[str]
