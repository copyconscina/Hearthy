from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.v1.router import router
from app.core.config import get_settings
from app.core.dependencies import get_predictor, get_chatbot


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model dan bot sekali saat startup."""
    settings = get_settings()
    print(f"[Hearthy] env={settings.app_env}")
    get_predictor()   # warm-up: load TF model
    get_chatbot()     # warm-up: init Gemini
    print("[Hearthy] Model dan bot siap.")
    yield


app = FastAPI(
    title="Hearthy API",
    description="Backend prediksi risiko penyakit jantung + HearthyBot",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ganti dengan domain frontend di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/api/v1/health", tags=["System"])
def health():
    return {"status": "ok"}
