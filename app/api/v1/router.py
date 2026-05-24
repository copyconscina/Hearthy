from fastapi import APIRouter
from app.api.v1.endpoints import predict, chat, peer_endpoint

router = APIRouter(prefix="/api/v1")

router.include_router(predict.router, tags=["Prediction"])
router.include_router(chat.router, tags=["Chatbot"])
router.include_router(peer_endpoint.router, tags=["Peers Comparison"])
