"""
Dependency injection: model TF dan HearthyBot di-load sekali saat startup,
lalu di-inject ke endpoint via FastAPI Depends().
"""
from functools import lru_cache
from app.services.predictor import HearthyPredictor
from app.services.chatbot import HearthyBot
from app.core.config import get_settings


@lru_cache
def get_predictor() -> HearthyPredictor:
    settings = get_settings()
    return HearthyPredictor(
        model_path=settings.model_path,
        scaler_path=settings.scaler_path,
        label_encoder_path=settings.label_encoder_path,
    )


@lru_cache
def get_chatbot() -> HearthyBot:
    settings = get_settings()
    return HearthyBot(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
        knowledge_base_path=settings.knowledge_base_path,
    )
