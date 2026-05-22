from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite-preview"

    # Paths
    model_path: str = "app/data/hearthy_model.keras"
    scaler_path: str = "app/data/scaler.pkl"
    label_encoder_path: str = "app/data/label_encoder.pkl"
    knowledge_base_path: str = "app/data/knowledge_base.json"

    # Server
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
