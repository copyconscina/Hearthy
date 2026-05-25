from fastapi import APIRouter
from app.api.v1.endpoints import predict

router = APIRouter(prefix="/api/v1")

router.include_router(predict.router, tags=["Prediction"])