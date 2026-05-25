from functools import lru_cache
from app.services.predictor import HearthyPredictor
from app.services.gemini_recommender import GeminiRecommender
from app.core.config import get_settings


@lru_cache
def get_recommender() -> GeminiRecommender:
    settings = get_settings()
    return GeminiRecommender(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
    )


@lru_cache
def get_predictor() -> HearthyPredictor:
    settings = get_settings()
    return HearthyPredictor(
        model_path=settings.model_path,
        scaler_path=settings.scaler_path,
        label_encoder_path=settings.label_encoder_path,
        recommender=get_recommender(),
    )