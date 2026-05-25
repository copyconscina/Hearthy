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
    Prediksi risiko kardiovaskular + rekomendasi aktivitas dari Gemini.

    Return:
    - **risk_category**: Low | Medium | High
    - **risk_score**: 0–100
    - **confidence**: probabilitas prediksi (0–1)
    - **recommendations**: rekomendasi aktivitas personal dari Gemini AI
    """
    try:
        return predictor.predict(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")