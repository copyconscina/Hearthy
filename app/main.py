from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.v1.router import router
from app.core.config import get_settings
from app.core.dependencies import get_predictor, get_recommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    print(f"[Hearthy] env={settings.app_env}")
    get_recommender()  # warm-up: init Gemini client
    get_predictor()    # warm-up: load TF model
    print("[Hearthy] Model dan Gemini siap.")
    yield


app = FastAPI(
    title="Hearthy API",
    description="Prediksi risiko kardiovaskular + rekomendasi aktivitas oleh Gemini AI",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/api/v1/health", tags=["System"])
def health():
    return {"status": "ok"}