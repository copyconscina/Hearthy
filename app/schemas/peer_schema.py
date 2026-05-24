from pydantic import BaseModel, Field
from typing import Optional


class PeersRequest(BaseModel):
    age: float = Field(..., ge=18, le=90)
    risk_score: float = Field(..., ge=0, le=100)
    risk_category: str = Field(..., description="Low | Medium | High")

    # Parameter untuk dibandingkan — semua opsional
    systolic_bp: Optional[float] = Field(None, ge=111, le=183)
    diastolic_bp: Optional[float] = Field(None, ge=69, le=120)
    cholesterol_mg_dl: Optional[float] = Field(None, ge=160, le=320)
    bmi: Optional[float] = Field(None, ge=16, le=40)
    daily_steps: Optional[float] = Field(None, ge=500, le=14000)
    physical_activity_hours_per_week: Optional[float] = Field(None, ge=0, le=10)
    sleep_hours: Optional[float] = Field(None, ge=4, le=9)
    stress_level: Optional[float] = Field(None, ge=1, le=10)
    diet_quality_score: Optional[float] = Field(None, ge=1, le=10)


class PeerStat(BaseModel):
    parameter: str
    nilai_kamu: float
    rata_rata_peers: float
    persentil: float        # kamu lebih baik dari X% peers
    selisih: float          # nilai_kamu - rata_rata (negatif = kamu lebih rendah)
    status: str             # "Lebih Baik dari Peers" | "Rata-rata" | "Perlu Diperbaiki"


class PeersResponse(BaseModel):
    age_group_label: str
    total_peers: int
    risk_score_percentile: float
    risk_category_distribution: dict[str, float]
    parameter_comparison: list[PeerStat]
    summary: str
