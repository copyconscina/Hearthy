from fastapi import APIRouter, Depends, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chatbot import HearthyBot
from app.core.dependencies import get_chatbot

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    bot: HearthyBot = Depends(get_chatbot),
) -> ChatResponse:
    """
    Chat dengan HearthyBot — asisten virtual edukasi kesehatan jantung.

    Kirim history percakapan sebelumnya agar konteks terjaga.
    History dikelola di sisi client (stateless API).
    """
    try:
        return bot.chat(req)
    except Exception as e:
        err = str(e)
        if "429" in err:
            raise HTTPException(status_code=429, detail="Kuota API Gemini habis, coba lagi nanti.")
        elif "403" in err or "401" in err:
            raise HTTPException(status_code=500, detail="API key tidak valid.")
        raise HTTPException(status_code=500, detail=f"Chat error: {err[:200]}")
