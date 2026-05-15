from pydantic import BaseModel
from typing import Optional


class ChatMessage(BaseModel):
    role: str       # "user" | "model"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[list[ChatMessage]] = []


class ChatResponse(BaseModel):
    reply: str
