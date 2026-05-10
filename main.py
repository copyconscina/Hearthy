from fastapi import FastAPI
from backend.routes.predict import router as predict_router
from backend.routes.chat import router as chat_router

app = FastAPI(title="Hearthy API")

app.include_router(predict_router, prefix="/api")
app.include_router(chat_router, prefix="/api")

@app.get("/")
def root():
    return {"status": "Hearthy API aktif"}
