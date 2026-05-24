from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash-lite"

    # Paths
    model_path: str = "app/data/best_attention_model.keras"
    scaler_path: str = "app/data/scaler_attn.pkl"
    label_encoder_path: str = "app/data/label_encoder_attn.pkl"
    knowledge_base_path: str = "app/data/knowledge_base.json"
    dataset_path: str = "app/data/cardiovascular_risk_dataset_feature_engineered.csv"

    # Server
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
