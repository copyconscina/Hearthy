"""
Logic generate_recommendations dan generate_risk_comparison.
Dipindahkan dari notebook ke sini agar bisa di-reuse oleh predictor
dan juga langsung di-call dari endpoint kalau perlu.
"""
from typing import Any


def parse_blood_pressure(bp_string: str) -> tuple[float, float]:
    """Parse '120/80' → (120.0, 80.0)"""
    parts = str(bp_string).split("/")
    return float(parts[0]), float(parts[1])


def generate_risk_comparison(user_input: dict[str, Any]) -> list[dict]:
    """
    Bandingkan parameter pasien dengan standar medis internasional.
    Mengembalikan list narasi perbandingan.
    """
    comparisons = []

    standards = {
        "Tekanan Darah": {
            "value_key": ("systolic_bp", "diastolic_bp"),
            "unit": "mmHg",
            "optimal": "< 120/80",
            "normal": "< 130/85",
            "high": ">= 140/90",
            "sumber": "AHA/ACC 2017 Hypertension Guidelines",
        },
        "Kolesterol": {
            "value_key": "cholesterol_mg_dl",
            "unit": "mg/dL",
            "optimal": "< 200",
            "borderline": "200–239",
            "high": ">= 240",
            "sumber": "NCEP ATP III Guidelines",
        },
        "BMI": {
            "value_key": "bmi",
            "unit": "",
            "normal": "18.5–24.9",
            "overweight": "25–29.9",
            "obese": ">= 30",
            "sumber": "WHO BMI Classification",
        },
    }

    # Tekanan darah
    sys_bp = user_input.get("systolic_bp", 0)
    dia_bp = user_input.get("diastolic_bp", 0)
    if sys_bp >= 140 or dia_bp >= 90:
        status = "TINGGI (Hipertensi Stadium 1+)"
    elif sys_bp >= 130 or dia_bp >= 80:
        status = "ELEVASI (Pre-hipertensi)"
    else:
        status = "NORMAL"
    comparisons.append({
        "narasi": (
            f"Tekanan Darah: {sys_bp:.0f}/{dia_bp:.0f} mmHg — {status} "
            f"(standar optimal < 120/80 mmHg · AHA/ACC 2017)"
        )
    })

    # Kolesterol
    chol = user_input.get("cholesterol_mg_dl", 0)
    if chol >= 240:
        chol_status = "TINGGI"
    elif chol >= 200:
        chol_status = "BORDERLINE TINGGI"
    else:
        chol_status = "OPTIMAL"
    comparisons.append({
        "narasi": (
            f"Kolesterol: {chol:.0f} mg/dL — {chol_status} "
            f"(standar optimal < 200 mg/dL · NCEP ATP III)"
        )
    })

    # BMI
    bmi = user_input.get("bmi", 0)
    if bmi >= 30:
        bmi_status = "OBESITAS"
    elif bmi >= 25:
        bmi_status = "OVERWEIGHT"
    elif bmi >= 18.5:
        bmi_status = "NORMAL"
    else:
        bmi_status = "BERAT BADAN KURANG"
    comparisons.append({
        "narasi": (
            f"BMI: {bmi:.1f} — {bmi_status} "
            f"(standar normal 18.5–24.9 · WHO)"
        )
    })

    return comparisons


def generate_recommendations(user_input: dict[str, Any]) -> dict[str, list]:
    """
    Hasilkan rekomendasi berdasarkan kondisi pasien.
    Return: {'urgent': [...], 'warning': [...], 'good': [...]}
    """
    urgent, warning, good = [], [], []

    # ── Tekanan darah ────────────────────────────────────────────
    sys_bp = user_input.get("systolic_bp", 0)
    dia_bp = user_input.get("diastolic_bp", 0)

    if sys_bp >= 140 or dia_bp >= 90:
        urgent.append({
            "parameter": "Tekanan Darah",
            "kondisi": f"{sys_bp:.0f}/{dia_bp:.0f} mmHg — Hipertensi",
            "rekomendasi": [
                "Segera konsultasi dokter untuk evaluasi dan kemungkinan terapi",
                "Kurangi asupan garam (< 2.3 g sodium/hari)",
                "Hindari kafein dan alkohol berlebih",
                "Olahraga aerobik ringan-sedang 30 menit/hari terbukti menurunkan TD",
                "Monitor tekanan darah harian dengan alat di rumah",
            ],
            "sumber": "AHA/ACC 2017 Hypertension Guidelines",
        })
    elif sys_bp >= 130 or dia_bp >= 80:
        warning.append({
            "parameter": "Tekanan Darah",
            "kondisi": f"{sys_bp:.0f}/{dia_bp:.0f} mmHg — Elevasi (Pre-hipertensi)",
            "rekomendasi": [
                "Kurangi asupan garam dan makanan olahan",
                "Tingkatkan aktivitas fisik rutin",
                "Monitor tekanan darah setiap 1–2 minggu",
            ],
            "sumber": "AHA/ACC 2017 Hypertension Guidelines",
        })
    else:
        good.append(f"Tekanan darah normal ({sys_bp:.0f}/{dia_bp:.0f} mmHg)")

    # ── Kolesterol ───────────────────────────────────────────────
    chol = user_input.get("cholesterol_mg_dl", 0)
    if chol >= 240:
        urgent.append({
            "parameter": "Kolesterol",
            "kondisi": f"{chol:.0f} mg/dL — Tinggi",
            "rekomendasi": [
                "Konsultasi dokter untuk evaluasi profil lipid lengkap (HDL, LDL, trigliserida)",
                "Kurangi lemak jenuh dan lemak trans (daging merah, gorengan, margarin)",
                "Perbanyak serat larut: oat, kacang-kacangan, buah apel, alpukat",
                "Olahraga aerobik rutin terbukti meningkatkan HDL dan menurunkan LDL",
            ],
            "sumber": "NCEP ATP III Guidelines",
        })
    elif chol >= 200:
        warning.append({
            "parameter": "Kolesterol",
            "kondisi": f"{chol:.0f} mg/dL — Borderline Tinggi",
            "rekomendasi": [
                "Kurangi konsumsi lemak jenuh dan gorengan",
                "Tambah konsumsi ikan berlemak (salmon, tuna, sarden) 2x/minggu",
                "Periksa profil lipid lengkap dalam 6 bulan ke depan",
            ],
            "sumber": "NCEP ATP III Guidelines",
        })
    else:
        good.append(f"Kolesterol dalam batas optimal ({chol:.0f} mg/dL)")

    # ── BMI ──────────────────────────────────────────────────────
    bmi = user_input.get("bmi", 0)
    if bmi >= 30:
        urgent.append({
            "parameter": "BMI",
            "kondisi": f"{bmi:.1f} — Obesitas",
            "rekomendasi": [
                "Target penurunan berat badan 5–10% dari berat saat ini secara bertahap",
                "Kombinasikan latihan aerobik dan latihan kekuatan minimal 3x/minggu",
                "Konsultasi ahli gizi untuk program diet yang aman",
            ],
            "sumber": "AHA Lifestyle Guidelines & WHO BMI Classification",
        })
    elif bmi >= 25:
        warning.append({
            "parameter": "BMI",
            "kondisi": f"{bmi:.1f} — Overweight",
            "rekomendasi": [
                "Perbanyak konsumsi sayur dan protein tanpa lemak",
                "Kurangi makanan tinggi kalori kosong (minuman manis, snack olahan)",
                "Tambah aktivitas fisik harian minimal 30 menit/hari",
            ],
            "sumber": "AHA Lifestyle Guidelines & WHO BMI Classification",
        })
    elif bmi < 18.5:
        warning.append({
            "parameter": "BMI",
            "kondisi": f"{bmi:.1f} — Berat Badan Kurang",
            "rekomendasi": [
                "Tingkatkan asupan kalori dari sumber nutrisi padat gizi",
                "Konsultasi dokter atau ahli gizi",
            ],
            "sumber": "WHO BMI Classification",
        })
    else:
        good.append(f"BMI dalam batas normal ({bmi:.1f})")

    # ── Aktivitas fisik ──────────────────────────────────────────
    activity = user_input.get("physical_activity_hours_per_week", 0)
    activity_min = activity * 60
    if activity_min < 75:
        urgent.append({
            "parameter": "Aktivitas Fisik",
            "kondisi": f"{activity:.1f} jam/minggu — Sangat Kurang",
            "rekomendasi": [
                "Target minimal 150 menit/minggu aktivitas aerobik intensitas sedang",
                "Mulai bertahap: jalan kaki 10 menit/hari, tingkatkan setiap minggu",
                "Tambah latihan kekuatan (resistance training) 2x/minggu",
            ],
            "sumber": "WHO Physical Activity Guidelines 2020",
        })
    elif activity_min < 150:
        warning.append({
            "parameter": "Aktivitas Fisik",
            "kondisi": f"{activity:.1f} jam/minggu — Kurang dari rekomendasi",
            "rekomendasi": [
                "Tingkatkan durasi olahraga hingga 150 menit/minggu",
                "Coba tambah 1 sesi olahraga per minggu secara bertahap",
            ],
            "sumber": "WHO Physical Activity Guidelines 2020",
        })
    else:
        good.append(f"Aktivitas fisik sudah memenuhi rekomendasi WHO ({activity:.1f} jam/minggu)")

    # ── Langkah per hari ─────────────────────────────────────────
    steps = user_input.get("daily_steps", 0)
    if steps < 5000:
        urgent.append({
            "parameter": "Langkah per Hari",
            "kondisi": f"{int(steps)} langkah — Kurang Aktif",
            "rekomendasi": [
                "Target minimal 7.500–10.000 langkah/hari",
                "Gunakan tangga daripada lift, parkir lebih jauh",
            ],
            "sumber": "JAMA Internal Medicine 2021",
        })
    elif steps < 7500:
        warning.append({
            "parameter": "Langkah per Hari",
            "kondisi": f"{int(steps)} langkah — Cukup Aktif",
            "rekomendasi": ["Tingkatkan ke 7.500–10.000 langkah/hari untuk manfaat optimal"],
            "sumber": "JAMA Internal Medicine 2021",
        })
    else:
        good.append(f"Jumlah langkah harian sudah baik ({int(steps)} langkah/hari)")

    # ── Tidur ────────────────────────────────────────────────────
    sleep = user_input.get("sleep_hours", 0)
    if sleep < 6:
        urgent.append({
            "parameter": "Durasi Tidur",
            "kondisi": f"{sleep} jam/malam — Kurang",
            "rekomendasi": [
                "Target 7–9 jam tidur per malam untuk orang dewasa",
                "Tetapkan jadwal tidur dan bangun yang konsisten setiap hari",
                "Hindari layar gadget minimal 1 jam sebelum tidur",
            ],
            "sumber": "National Sleep Foundation 2015",
        })
    elif sleep > 9:
        warning.append({
            "parameter": "Durasi Tidur",
            "kondisi": f"{sleep} jam/malam — Berlebihan",
            "rekomendasi": ["Konsultasi dokter jika sering merasa lelah meski tidur lama"],
            "sumber": "National Sleep Foundation 2015",
        })
    else:
        good.append(f"Durasi tidur normal ({sleep} jam/malam)")

    # ── Alkohol ──────────────────────────────────────────────────
    alcohol = user_input.get("alcohol_units_per_week", 0)
    if alcohol > 14:
        urgent.append({
            "parameter": "Konsumsi Alkohol",
            "kondisi": f"{alcohol} unit/minggu — Tinggi (Berisiko)",
            "rekomendasi": [
                "Kurangi konsumsi alkohol secara bertahap",
                "Target di bawah 14 unit/minggu, idealnya lebih rendah",
                "Cari dukungan profesional jika sulit mengurangi sendiri",
            ],
            "sumber": "WHO Alcohol Guidelines",
        })
    elif alcohol > 7:
        warning.append({
            "parameter": "Konsumsi Alkohol",
            "kondisi": f"{alcohol} unit/minggu — Sedang",
            "rekomendasi": ["Selipkan hari-hari bebas alkohol dalam seminggu"],
            "sumber": "WHO Alcohol Guidelines",
        })
    else:
        good.append("Konsumsi alkohol dalam batas aman")

    # ── Stres ────────────────────────────────────────────────────
    stress = user_input.get("stress_level", 0)
    if stress >= 7:
        urgent.append({
            "parameter": "Tingkat Stres",
            "kondisi": f"{stress}/10 — Tinggi",
            "rekomendasi": [
                "Latihan pernapasan dalam 5–10 menit/hari",
                "Meditasi atau mindfulness minimal 10 menit/hari",
                "Pertimbangkan konsultasi dengan psikolog atau konselor",
            ],
            "sumber": "AHA Stress & Heart Disease",
        })
    elif stress >= 4:
        warning.append({
            "parameter": "Tingkat Stres",
            "kondisi": f"{stress}/10 — Sedang",
            "rekomendasi": ["Luangkan waktu untuk hobi dan aktivitas relaksasi"],
            "sumber": "AHA Stress & Heart Disease",
        })
    else:
        good.append(f"Tingkat stres terkendali ({stress}/10)")

    # ── Kualitas diet ────────────────────────────────────────────
    diet = user_input.get("diet_quality_score", 0)
    if diet <= 3:
        urgent.append({
            "parameter": "Kualitas Diet",
            "kondisi": f"{diet}/10 — Buruk",
            "rekomendasi": [
                "Perbanyak konsumsi buah dan sayuran minimal 5 porsi/hari",
                "Kurangi makanan ultra-processed (mie instan, fast food, minuman manis)",
                "Konsultasi ahli gizi untuk panduan diet yang terstruktur",
            ],
            "sumber": "AHA Lifestyle Guidelines",
        })
    elif diet <= 6:
        warning.append({
            "parameter": "Kualitas Diet",
            "kondisi": f"{diet}/10 — Cukup",
            "rekomendasi": ["Tingkatkan variasi sayuran dan buah dalam menu harian"],
            "sumber": "AHA Lifestyle Guidelines",
        })
    else:
        good.append(f"Kualitas diet sudah baik ({diet}/10)")

    # ── Riwayat keluarga ─────────────────────────────────────────
    if user_input.get("family_history_heart_disease", False):
        warning.append({
            "parameter": "Riwayat Keluarga",
            "kondisi": "Ada riwayat penyakit jantung dalam keluarga",
            "rekomendasi": [
                "Lakukan skrining jantung rutin minimal 1x per tahun",
                "Informasikan riwayat keluarga ke dokter untuk asesmen risiko genetik",
            ],
            "sumber": "AHA Family History & Heart Disease",
        })

    return {"urgent": urgent, "warning": warning, "good": good}
