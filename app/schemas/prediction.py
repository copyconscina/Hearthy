from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    age: float = Field(..., ge=18, le=90)
    systolic_bp: float = Field(..., ge=111, le=183)
    diastolic_bp: float = Field(..., ge=69, le=120)
    resting_heart_rate: float = Field(..., ge=56, le=92)
    cholesterol_mg_dl: float = Field(..., ge=160, le=320)
    bmi: float = Field(..., ge=16, le=40)
    daily_steps: float = Field(..., ge=500, le=14000)
    physical_activity_hours_per_week: float = Field(..., ge=0, le=10)
    sleep_hours: float = Field(..., ge=4, le=9)
    alcohol_units_per_week: float = Field(..., ge=0, le=11)
    stress_level: float = Field(..., ge=1, le=10)
    diet_quality_score: float = Field(..., ge=1, le=10)
    smoking_status: int = Field(..., ge=0, le=2, description="0=Never, 1=Former, 2=Current")
    family_history_heart_disease: bool = Field(False)
    diabetes: int = Field(0, ge=0, le=1)
    hypertension: int = Field(0, ge=0, le=1)


class F1Scores(BaseModel):
    low: float
    medium: float
    high: float
    macro_avg: float


class PredictionResponse(BaseModel):
    risk_category: str
    risk_score: float
    confidence: float
    f1_scores: F1Scores
    recommendations: str