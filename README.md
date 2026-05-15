# Hearthy API

Backend FastAPI untuk aplikasi Hearthy — prediksi risiko penyakit jantung berbasis AI.

## Struktur Proyek

```
hearthy-api/
├── app/
│   ├── main.py                  # Entry point FastAPI
│   ├── core/
│   │   ├── config.py            # Settings (env vars, konstanta)
│   │   └── dependencies.py      # Dependency injection (model, bot)
│   ├── schemas/
│   │   ├── prediction.py        # Request/Response schema prediksi
│   │   └── chat.py              # Request/Response schema chatbot
│   ├── services/
│   │   ├── predictor.py         # Logic inferensi model TF + rekomendasi
│   │   ├── recommender.py       # generate_recommendations & risk_comparison
│   │   └── chatbot.py           # Logic HearthyBot (Gemini)
│   └── api/
│       └── v1/
│           ├── router.py        # Gabung semua endpoint
│           └── endpoints/
│               ├── predict.py   # POST /predict
│               └── chat.py      # POST /chat
├── scripts/
│   └── load_knowledge.py        # Script sekali jalan: PDF → knowledge_base.json
├── .env.example
├── requirements.txt
└── README.md
```

## Cara Pakai

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup .env
```bash
cp .env.example .env
# isi GEMINI_API_KEY dan path model
```

### 3. Build knowledge base (sekali jalan)
```bash
python scripts/load_knowledge.py --folder /path/ke/jurnal --output app/data/knowledge_base.json
```

### 4. Jalankan server
```bash
uvicorn app.main:app --reload
```

## Endpoints

| Method | Path | Deskripsi |
|--------|------|-----------|
| POST | `/api/v1/predict` | Prediksi risiko kardiovaskular |
| POST | `/api/v1/chat` | Chat dengan HearthyBot |
| GET | `/api/v1/health` | Health check |
