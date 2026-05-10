# Hearthy — Struktur Proyek

```
hearthy/
│
├── backend/
│   ├── main.py                  # Entry point FastAPI
│   ├── routes/
│   │   ├── predict.py           # POST /api/predict
│   │   └── chat.py              # POST /api/chat
│   ├── services/
│   │   ├── ml_service.py        # Load & jalankan model TensorFlow
│   │   └── gemini_service.py    # HearthyBot (Gemini API)
│   └── models/
│       └── schemas.py           # Validasi input/output (Pydantic)
│
├── ml/
│   ├── guidelines/              # ← Taruh PDF jurnal di sini
│   │   ├── hypertension.pdf
│   │   ├── cholesterol.pdf
│   │   └── ...
│   ├── cardio_attention_model_final.keras   # ← Hasil training
│   ├── scaler_attn.pkl                      # ← Hasil training
│   └── label_encoder_attn.pkl              # ← Hasil training
│
├── notebooks/
│   ├── HearthyBot_Colab.ipynb   # Notebook chatbot (development)
│   └── percobaan_1.ipynb        # Notebook modelling (development)
│
├── .env.example                 # Template environment variable
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` ke `.env` dan isi API key:
   ```bash
   cp .env.example .env
   ```

3. Salin hasil training dari notebook ke folder `ml/`:
   - `cardio_attention_model_final.keras`
   - `scaler_attn.pkl`
   - `label_encoder_attn.pkl`

4. Taruh PDF guidelines di `ml/guidelines/`

5. Jalankan server:
   ```bash
   uvicorn backend.main:app --reload
   ```

6. Akses API docs di: http://localhost:8000/docs
