from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    # Demografis — dataset: age 18-90
    age: float = Field(..., ge=18, le=90, description="Usia pasien (tahun)")

    # Vital sign — sesuai range dataset
    systolic_bp: float = Field(..., ge=111, le=183, description="Tekanan darah sistolik (mmHg)")
    diastolic_bp: float = Field(..., ge=69, le=120, description="Tekanan darah diastolik (mmHg)")
    resting_heart_rate: float = Field(..., ge=56, le=92, description="Detak jantung istirahat (bpm)")
    cholesterol_mg_dl: float = Field(..., ge=160, le=320, description="Kolesterol total (mg/dL)")

    # Antropometri — dataset: bmi ~16-40
    bmi: float = Field(..., ge=16, le=40, description="Body Mass Index")

    # Gaya hidup
    daily_steps: float = Field(..., ge=500, le=14000, description="Rata-rata langkah per hari")
    physical_activity_hours_per_week: float = Field(..., ge=0, le=10, description="Jam aktivitas fisik per minggu")
    sleep_hours: float = Field(..., ge=4, le=9, description="Jam tidur per malam")
    alcohol_units_per_week: float = Field(..., ge=0, le=11, description="Unit alkohol per minggu")
    stress_level: float = Field(..., ge=1, le=10, description="Tingkat stres (1-10)")
    diet_quality_score: float = Field(..., ge=1, le=10, description="Skor kualitas diet (1-10)")

    # Nominal/Binary — dataset: Never, Former, Current → 0, 1, 2
    smoking_status: int = Field(..., ge=0, le=2, description="0=Never, 1=Former, 2=Current")
    family_history_heart_disease: bool = Field(False, description="Riwayat keluarga penyakit jantung (Yes/No)")

    # Tidak ada di dataset asli tapi dipakai di derive features
    diabetes: int = Field(0, ge=0, le=1)
    hypertension: int = Field(0, ge=0, le=1)


class PredictionResponse(BaseModel):
    risk_category: str          # "Low" | "Medium" | "High"
    risk_score: float           # 0–100
    confidence: float           # probabilitas kelas prediksi (0–1)

    recommendations: dict[str, list] = Field(
        description="{'urgent': [...], 'warning': [...], 'good': [...]}"
    )
    risk_comparison: list[dict]
