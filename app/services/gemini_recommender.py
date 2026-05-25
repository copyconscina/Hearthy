"""
GeminiRecommender: terima hasil prediksi TF + data pasien,
kirim ke Gemini API, kembalikan rekomendasi aktivitas personal.
"""
from google import genai
from google.genai import types


SYSTEM_PROMPT = """
Anda adalah HearthyBot, asisten kesehatan jantung dari aplikasi Hearthy.

Tugas Anda adalah memberikan rekomendasi aktivitas dan gaya hidup yang
PERSONAL dan SPESIFIK berdasarkan hasil prediksi risiko kardiovaskular pasien.

Aturan wajib:
- Tulis dalam Bahasa Indonesia yang ramah dan mudah dipahami.
- Berikan rekomendasi yang KONKRET dan BISA LANGSUNG DILAKUKAN.
- Struktur jawaban: 3 bagian — (1) Ringkasan kondisi, (2) Rekomendasi aktivitas
  fisik, (3) Rekomendasi gaya hidup & pola makan.
- Gunakan poin-poin singkat, maksimal 5 poin per bagian.
- Di akhir, selalu tambahkan disclaimer singkat bahwa ini bukan diagnosis medis.
- Jika risiko Tinggi (High), tekankan untuk segera konsultasi dokter.
- Gejala darurat (nyeri dada, sesak napas mendadak): instruksikan hubungi 119.
"""


def build_prompt(risk_category: str, risk_score: float, data: dict) -> str:
    smoking_map = {0: "Tidak pernah merokok", 1: "Mantan perokok", 2: "Perokok aktif"}
    smoking_label = smoking_map.get(data.get("smoking_status", 0), "-")
    family_history = "Ada" if data.get("family_history_heart_disease") else "Tidak ada"

    return f"""
Berikut adalah data dan hasil prediksi pasien:

**HASIL PREDIKSI MODEL AI:**
- Kategori Risiko : {risk_category}
- Skor Risiko     : {risk_score:.1f} / 100
- Confidence      : {data.get('confidence', 0):.1%}

**DATA KLINIS PASIEN:**
- Usia            : {data.get('age')} tahun
- Tekanan Darah   : {data.get('systolic_bp')}/{data.get('diastolic_bp')} mmHg
- Detak Jantung   : {data.get('resting_heart_rate')} bpm
- Kolesterol      : {data.get('cholesterol_mg_dl')} mg/dL
- BMI             : {data.get('bmi'):.1f}
- Riwayat Keluarga: {family_history}
- Diabetes        : {'Ya' if data.get('diabetes') else 'Tidak'}

**GAYA HIDUP PASIEN:**
- Langkah/hari       : {int(data.get('daily_steps', 0))}
- Aktivitas fisik    : {data.get('physical_activity_hours_per_week')} jam/minggu
- Jam tidur          : {data.get('sleep_hours')} jam/malam
- Alkohol            : {data.get('alcohol_units_per_week')} unit/minggu
- Tingkat stres      : {data.get('stress_level')}/10
- Kualitas diet      : {data.get('diet_quality_score')}/10
- Status merokok     : {smoking_label}

Berikan rekomendasi aktivitas dan gaya hidup yang personal untuk pasien ini.
""".strip()


class GeminiRecommender:
    def __init__(self, api_key: str, model_name: str):
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def recommend(self, risk_category: str, risk_score: float, patient_data: dict) -> str:
        prompt = build_prompt(risk_category, risk_score, patient_data)

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )
        return response.text