"""
HearthyBot: wrapper Gemini API dengan knowledge base dari jurnal medis.
Knowledge base di-load sekali saat inisialisasi.
"""
import json
import google.generativeai as genai

from app.schemas.chat import ChatRequest, ChatResponse, ChatMessage

SYSTEM_PROMPT_TEMPLATE = """
Anda adalah HearthyBot, asisten virtual untuk aplikasi Hearthy,
platform prediksi risiko penyakit jantung berbasis AI.

Berikut adalah referensi jurnal medis yang dapat Anda gunakan:

{knowledge_base}

---

Tugas utama:
1. Memandu pengguna menggunakan fitur Hearthy.
2. Memberikan edukasi tentang deteksi dini kesehatan jantung.
3. Menjelaskan klasifikasi risiko (Rendah, Sedang, Tinggi).
4. Memberikan rekomendasi gaya hidup berbasis jurnal di atas.

Aturan:
- Anda bukan dokter. Jangan berikan diagnosis medis pasti.
- Setiap saran kesehatan wajib sertakan disclaimer skrining awal.
- Gejala darurat (nyeri dada, sesak napas, pingsan) instruksikan hubungi 119.
- Jawab singkat, padat, gunakan poin-poin.
- Pertanyaan di luar topik kesehatan jantung, arahkan kembali dengan sopan.
"""


def _load_knowledge_base(path: str) -> str:
    """Muat knowledge_base.json dan ubah jadi string ringkas untuk system prompt."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            semua_jurnal: dict = json.load(f)
    except FileNotFoundError:
        return "(Tidak ada knowledge base jurnal yang dimuat.)"

    parts = []
    for nama, data in semua_jurnal.items():
        teks_gabung = "\n".join([h["isi"] for h in data.get("konten", [])])
        parts.append(f"[{data.get('judul', nama)}]\n{teks_gabung[:3000]}")

    return "\n\n---\n\n".join(parts)


class HearthyBot:
    def __init__(self, api_key: str, model_name: str, knowledge_base_path: str):
        genai.configure(api_key=api_key)

        knowledge_base = _load_knowledge_base(knowledge_base_path)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(knowledge_base=knowledge_base)

        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
        )

    def chat(self, req: ChatRequest) -> ChatResponse:
        """
        Kirim pesan dengan history percakapan.
        History dikelola di sisi client/frontend dan dikirim tiap request
        (Gemini API stateless — tidak ada session server-side).
        """
        history = [
            {"role": msg.role, "parts": [msg.content]}
            for msg in (req.history or [])
        ]

        session = self._model.start_chat(history=history)
        response = session.send_message(req.message)
        return ChatResponse(reply=response.text)
