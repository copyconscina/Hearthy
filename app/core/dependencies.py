from functools import lru_cache
from app.services.predictor import HearthyPredictor
from app.services.chatbot import HearthyBot
from app.services.peer_analyzer import PeersAnalyzer
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


@lru_cache
def get_peers_analyzer() -> PeersAnalyzer:
    settings = get_settings()
    return PeersAnalyzer(dataset_path=settings.dataset_path)
