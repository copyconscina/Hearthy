from fastapi import APIRouter, Depends, HTTPException
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.predictor import HearthyPredictor
from app.core.dependencies import get_predictor

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(
    req: PredictionRequest,
    predictor: HearthyPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """
    Prediksi risiko kardiovaskular berdasarkan data klinis dan gaya hidup pasien.

    Return:
    - **risk_category**: Low | Medium | High
    - **risk_score**: 0–100
    - **confidence**: probabilitas prediksi (0–1)
    - **recommendations**: rekomendasi klinis terstruktur
    - **risk_comparison**: perbandingan parameter vs standar medis
    """
    try:
        return predictor.predict(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")