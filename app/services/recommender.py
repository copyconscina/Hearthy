"""
Risk Comparison + Sistem Rekomendasi
Sumber: AHA/ACC 2017, AHA/ACC 2018, WHO 2020,
        JAMA 2021, National Sleep Foundation 2015

Kode ini identik dengan notebook percobaan_1.ipynb — tidak ada pengurangan.
"""
from typing import Any


def get_clinical_category(feature: str, value: float) -> dict:
    """
    Klasifikasikan nilai user ke kategori klinis berdasarkan referensi medis.
    Return dict: {category, label, status, ref}
    """
    if feature == 'systolic_bp':
        if value < 120:
            return {'category': 'normal',   'label': 'Normal',             'status': '[NORMAL]',    'ref': '< 120 mmHg (AHA 2017)'}
        elif value < 130:
            return {'category': 'elevated', 'label': 'Elevated',           'status': '[PERHATIAN]', 'ref': '120-129 mmHg (AHA 2017)'}
        elif value < 140:
            return {'category': 'stage1',   'label': 'Hipertensi Stage 1', 'status': '[PERHATIAN]', 'ref': '130-139 mmHg (AHA 2017)'}
        else:
            return {'category': 'stage2',   'label': 'Hipertensi Stage 2', 'status': '[BERISIKO]',  'ref': '>= 140 mmHg (AHA 2017)'}

    elif feature == 'diastolic_bp':
        if value < 80:
            return {'category': 'normal',  'label': 'Normal',             'status': '[NORMAL]',    'ref': '< 80 mmHg (AHA 2017)'}
        elif value < 90:
            return {'category': 'stage1',  'label': 'Hipertensi Stage 1', 'status': '[PERHATIAN]', 'ref': '80-89 mmHg (AHA 2017)'}
        else:
            return {'category': 'stage2',  'label': 'Hipertensi Stage 2', 'status': '[BERISIKO]',  'ref': '>= 90 mmHg (AHA 2017)'}

    elif feature == 'cholesterol_mg_dl':
        if value < 200:
            return {'category': 'optimal',    'label': 'Optimal',           'status': '[NORMAL]',    'ref': '< 200 mg/dL (AHA 2018)'}
        elif value < 240:
            return {'category': 'borderline', 'label': 'Borderline Tinggi', 'status': '[PERHATIAN]', 'ref': '200-239 mg/dL (AHA 2018)'}
        else:
            return {'category': 'high',       'label': 'Tinggi',            'status': '[BERISIKO]',  'ref': '>= 240 mg/dL (AHA 2018)'}

    elif feature == 'bmi':
        if value < 18.5:
            return {'category': 'underweight', 'label': 'Berat Badan Kurang', 'status': '[PERHATIAN]', 'ref': '< 18.5 (WHO)'}
        elif value < 25.0:
            return {'category': 'normal',      'label': 'Normal',             'status': '[NORMAL]',    'ref': '18.5-24.9 (WHO)'}
        elif value < 30.0:
            return {'category': 'overweight',  'label': 'Overweight',         'status': '[PERHATIAN]', 'ref': '25.0-29.9 (WHO)'}
        else:
            return {'category': 'obese',       'label': 'Obesitas',           'status': '[BERISIKO]',  'ref': '>= 30.0 (WHO)'}

    elif feature == 'resting_heart_rate':
        if value < 60:
            return {'category': 'low',    'label': 'Rendah (Bradikardia)', 'status': '[PERHATIAN]', 'ref': '< 60 bpm (AHA)'}
        elif value <= 100:
            return {'category': 'normal', 'label': 'Normal',               'status': '[NORMAL]',    'ref': '60-100 bpm (AHA)'}
        else:
            return {'category': 'high',   'label': 'Tinggi (Takikardia)',  'status': '[BERISIKO]',  'ref': '> 100 bpm (AHA)'}

    elif feature == 'daily_steps':
        if value < 5000:
            return {'category': 'low',        'label': 'Kurang Aktif', 'status': '[BERISIKO]',  'ref': '< 5.000 langkah (JAMA 2021)'}
        elif value < 7500:
            return {'category': 'moderate',   'label': 'Cukup Aktif',  'status': '[PERHATIAN]', 'ref': '5.000-7.499 langkah (JAMA 2021)'}
        elif value < 10000:
            return {'category': 'active',     'label': 'Aktif',        'status': '[NORMAL]',    'ref': '7.500-9.999 langkah (JAMA 2021)'}
        else:
            return {'category': 'very_active','label': 'Sangat Aktif', 'status': '[NORMAL]',    'ref': '>= 10.000 langkah (JAMA 2021)'}

    elif feature == 'physical_activity_hours_per_week':
        if value < 1.25:
            return {'category': 'low',      'label': 'Kurang', 'status': '[BERISIKO]',  'ref': '< 1.25 jam/minggu (WHO 2020)'}
        elif value < 2.5:
            return {'category': 'moderate', 'label': 'Cukup',  'status': '[PERHATIAN]', 'ref': '1.25-2.5 jam/minggu (WHO 2020)'}
        else:
            return {'category': 'good',     'label': 'Baik',   'status': '[NORMAL]',    'ref': '>= 2.5 jam/minggu (WHO 2020)'}

    elif feature == 'sleep_hours':
        if value < 6:
            return {'category': 'insufficient', 'label': 'Kurang',     'status': '[BERISIKO]',  'ref': '< 6 jam (NSF 2015)'}
        elif value <= 9:
            return {'category': 'normal',       'label': 'Normal',     'status': '[NORMAL]',    'ref': '7-9 jam (NSF 2015)'}
        else:
            return {'category': 'excessive',    'label': 'Berlebihan', 'status': '[PERHATIAN]', 'ref': '> 9 jam (NSF 2015)'}

    elif feature == 'alcohol_units_per_week':
        if value == 0:
            return {'category': 'none',     'label': 'Tidak Minum',       'status': '[NORMAL]',    'ref': '0 unit (WHO)'}
        elif value <= 7:
            return {'category': 'low',      'label': 'Rendah',            'status': '[NORMAL]',    'ref': '1-7 unit/minggu (WHO)'}
        elif value <= 14:
            return {'category': 'moderate', 'label': 'Sedang',            'status': '[PERHATIAN]', 'ref': '8-14 unit/minggu (WHO)'}
        else:
            return {'category': 'high',     'label': 'Tinggi (Berisiko)', 'status': '[BERISIKO]',  'ref': '> 14 unit/minggu (WHO)'}

    elif feature == 'stress_level':
        if value <= 3:
            return {'category': 'low',      'label': 'Rendah', 'status': '[NORMAL]',    'ref': '1-3 / 10 (skala subjektif)'}
        elif value <= 6:
            return {'category': 'moderate', 'label': 'Sedang', 'status': '[PERHATIAN]', 'ref': '4-6 / 10 (skala subjektif)'}
        else:
            return {'category': 'high',     'label': 'Tinggi', 'status': '[BERISIKO]',  'ref': '7-10 / 10 (skala subjektif)'}

    elif feature == 'diet_quality_score':
        if value <= 3:
            return {'category': 'poor', 'label': 'Buruk', 'status': '[BERISIKO]',  'ref': '1-3 / 10 (skala kontekstual)'}
        elif value <= 6:
            return {'category': 'fair', 'label': 'Cukup', 'status': '[PERHATIAN]', 'ref': '4-6 / 10 (skala kontekstual)'}
        else:
            return {'category': 'good', 'label': 'Baik',  'status': '[NORMAL]',    'ref': '7-10 / 10 (skala kontekstual)'}

    else:
        return {'category': 'unknown', 'label': 'Tidak diketahui', 'status': '[N/A]', 'ref': '-'}


def generate_risk_comparison(user_input: dict) -> list:
    """
    Input  : user_input dict {nama_fitur: nilai}
    Output : list of dict, tiap dict = satu baris perbandingan
    """
    FEATURE_META = [
        ('systolic_bp',                      'Tekanan Darah Sistolik',  'mmHg'),
        ('diastolic_bp',                     'Tekanan Darah Diastolik', 'mmHg'),
        ('cholesterol_mg_dl',                'Kolesterol Total',        'mg/dL'),
        ('bmi',                              'BMI',                     ''),
        ('resting_heart_rate',               'Detak Jantung Istirahat', 'bpm'),
        ('daily_steps',                      'Langkah per Hari',        'langkah'),
        ('physical_activity_hours_per_week', 'Aktivitas Fisik',         'jam/minggu'),
        ('sleep_hours',                      'Jam Tidur',               'jam/hari'),
        ('alcohol_units_per_week',           'Konsumsi Alkohol',        'unit/minggu'),
        ('stress_level',                     'Tingkat Stres',           '/10'),
        ('diet_quality_score',               'Kualitas Diet',           '/10'),
    ]

    rows = []
    for key, label, unit in FEATURE_META:
        if key not in user_input:
            continue
        value    = user_input[key]
        clinical = get_clinical_category(key, value)
        unit_str = f' {unit}' if unit else ''
        narasi   = (
            f"{clinical['status']} {label}: {value}{unit_str} "
            f"| Kategori: {clinical['label']} "
            f"| Batas normal: {clinical['ref']}"
        )
        rows.append({
            'feature':  key,
            'label':    label,
            'value':    f'{value}{unit_str}',
            'category': clinical['label'],
            'status':   clinical['status'],
            'ref':      clinical['ref'],
            'narasi':   narasi,
        })
    return rows


def generate_recommendations(user_input: dict) -> dict:
    """
    Return dict: {urgent: [...], warning: [...], good: [...]}
    """
    urgent  = []
    warning = []
    good    = []

    sys_bp = user_input.get('systolic_bp', 0)
    if sys_bp >= 140:
        urgent.append({'parameter': 'Tekanan Darah Sistolik', 'kondisi': f'{sys_bp} mmHg — Hipertensi Stage 2',
            'rekomendasi': ['Kurangi konsumsi garam (sodium) di bawah 1.500 mg/hari',
                'Terapkan diet DASH (perbanyak buah, sayur, biji-bijian, rendah lemak jenuh)',
                'Olahraga aerobik minimal 30 menit/hari, 5 hari/minggu',
                'Hindari rokok dan batasi kafein',
                'Segera konsultasi dokter untuk evaluasi obat antihipertensi'],
            'sumber': 'AHA/ACC 2017 Hypertension Guidelines'})
    elif sys_bp >= 130:
        warning.append({'parameter': 'Tekanan Darah Sistolik', 'kondisi': f'{sys_bp} mmHg — Hipertensi Stage 1',
            'rekomendasi': ['Mulai terapkan diet DASH secara bertahap',
                'Kurangi konsumsi garam bertahap ke bawah 2.300 mg/hari',
                'Tambah aktivitas fisik ringan-sedang secara rutin'],
            'sumber': 'AHA/ACC 2017 Hypertension Guidelines'})
    elif sys_bp >= 120:
        warning.append({'parameter': 'Tekanan Darah Sistolik', 'kondisi': f'{sys_bp} mmHg — Elevated',
            'rekomendasi': ['Jaga pola makan rendah garam', 'Pertahankan berat badan ideal'],
            'sumber': 'AHA/ACC 2017 Hypertension Guidelines'})
    else:
        good.append('Tekanan Darah Sistolik dalam batas normal')

    dia_bp = user_input.get('diastolic_bp', 0)
    if dia_bp >= 90:
        urgent.append({'parameter': 'Tekanan Darah Diastolik', 'kondisi': f'{dia_bp} mmHg — Hipertensi Stage 2',
            'rekomendasi': ['Segera konsultasi dokter — diastolik >=90 mmHg memerlukan evaluasi medis',
                'Hindari stres berlebih dan istirahat cukup', 'Batasi konsumsi alkohol'],
            'sumber': 'AHA/ACC 2017 Hypertension Guidelines'})
    elif dia_bp >= 80:
        warning.append({'parameter': 'Tekanan Darah Diastolik', 'kondisi': f'{dia_bp} mmHg — Hipertensi Stage 1',
            'rekomendasi': ['Kelola stres dengan meditasi atau teknik relaksasi', 'Kurangi konsumsi alkohol'],
            'sumber': 'AHA/ACC 2017 Hypertension Guidelines'})
    else:
        good.append('Tekanan Darah Diastolik dalam batas normal')

    chol = user_input.get('cholesterol_mg_dl', 0)
    if chol >= 240:
        urgent.append({'parameter': 'Kolesterol Total', 'kondisi': f'{chol} mg/dL — Tinggi',
            'rekomendasi': ['Kurangi makanan tinggi lemak jenuh (daging merah, produk susu tinggi lemak)',
                'Perbanyak serat larut (oatmeal, kacang-kacangan, buah apel, pir)',
                'Konsumsi ikan berlemak (salmon, sarden) 2x seminggu untuk omega-3',
                'Hindari makanan trans fat (gorengan, makanan olahan)',
                'Konsultasi dokter untuk pertimbangan terapi statin'],
            'sumber': 'AHA/ACC 2018 Cholesterol Guidelines'})
    elif chol >= 200:
        warning.append({'parameter': 'Kolesterol Total', 'kondisi': f'{chol} mg/dL — Borderline Tinggi',
            'rekomendasi': ['Mulai kurangi lemak jenuh dalam makanan sehari-hari',
                'Tambah konsumsi serat dan sayuran hijau',
                'Rutin periksa kolesterol setiap 6 bulan'],
            'sumber': 'AHA/ACC 2018 Cholesterol Guidelines'})
    else:
        good.append('Kolesterol Total dalam batas optimal')

    bmi = user_input.get('bmi', 0)
    if bmi >= 30:
        urgent.append({'parameter': 'BMI', 'kondisi': f'{bmi:.1f} — Obesitas',
            'rekomendasi': ['Target penurunan berat badan 5-10% dari berat saat ini secara bertahap',
                'Defisit kalori moderat (300-500 kkal/hari), hindari diet ekstrem',
                'Kombinasikan latihan aerobik dan latihan kekuatan minimal 3x/minggu',
                'Konsultasi ahli gizi untuk program diet yang aman'],
            'sumber': 'AHA Lifestyle Guidelines & WHO BMI Classification'})
    elif bmi >= 25:
        warning.append({'parameter': 'BMI', 'kondisi': f'{bmi:.1f} — Overweight',
            'rekomendasi': ['Perbanyak konsumsi sayur dan protein tanpa lemak',
                'Kurangi makanan tinggi kalori kosong (minuman manis, snack olahan)',
                'Tambah aktivitas fisik harian minimal 30 menit/hari'],
            'sumber': 'AHA Lifestyle Guidelines & WHO BMI Classification'})
    elif bmi < 18.5:
        warning.append({'parameter': 'BMI', 'kondisi': f'{bmi:.1f} — Berat Badan Kurang',
            'rekomendasi': ['Tingkatkan asupan kalori dari sumber nutrisi padat gizi',
                'Konsultasi dokter atau ahli gizi'],
            'sumber': 'WHO BMI Classification'})
    else:
        good.append(f'BMI dalam batas normal ({bmi:.1f})')

    activity = user_input.get('physical_activity_hours_per_week', 0)
    activity_min = activity * 60
    if activity_min < 75:
        urgent.append({'parameter': 'Aktivitas Fisik', 'kondisi': f'{activity:.1f} jam/minggu — Sangat Kurang',
            'rekomendasi': ['Target minimal 150 menit/minggu aktivitas aerobik intensitas sedang',
                'Mulai bertahap: jalan kaki 10 menit/hari, tingkatkan setiap minggu',
                'Pilih aktivitas yang menyenangkan: bersepeda, renang, senam',
                'Tambah latihan kekuatan (resistance training) 2x/minggu'],
            'sumber': 'WHO Physical Activity Guidelines 2020'})
    elif activity_min < 150:
        warning.append({'parameter': 'Aktivitas Fisik', 'kondisi': f'{activity:.1f} jam/minggu — Kurang dari rekomendasi',
            'rekomendasi': ['Tingkatkan durasi olahraga hingga 150 menit/minggu',
                'Coba tambah 1 sesi olahraga per minggu secara bertahap'],
            'sumber': 'WHO Physical Activity Guidelines 2020'})
    else:
        good.append(f'Aktivitas fisik sudah memenuhi rekomendasi WHO ({activity:.1f} jam/minggu)')

    steps = user_input.get('daily_steps', 0)
    if steps < 5000:
        urgent.append({'parameter': 'Langkah per Hari', 'kondisi': f'{int(steps)} langkah — Kurang Aktif',
            'rekomendasi': ['Target minimal 7.500-10.000 langkah/hari',
                'Gunakan tangga daripada lift, parkir lebih jauh',
                'Jalan kaki saat istirahat makan siang 10-15 menit'],
            'sumber': 'JAMA Internal Medicine 2021'})
    elif steps < 7500:
        warning.append({'parameter': 'Langkah per Hari', 'kondisi': f'{int(steps)} langkah — Cukup Aktif',
            'rekomendasi': ['Tingkatkan ke 7.500-10.000 langkah/hari untuk manfaat optimal'],
            'sumber': 'JAMA Internal Medicine 2021'})
    else:
        good.append(f'Jumlah langkah harian sudah baik ({int(steps)} langkah/hari)')

    sleep = user_input.get('sleep_hours', 0)
    if sleep < 6:
        urgent.append({'parameter': 'Durasi Tidur', 'kondisi': f'{sleep} jam/malam — Kurang',
            'rekomendasi': ['Target 7-9 jam tidur per malam untuk orang dewasa',
                'Tetapkan jadwal tidur dan bangun yang konsisten setiap hari',
                'Hindari layar gadget minimal 1 jam sebelum tidur',
                'Ciptakan lingkungan tidur yang gelap, sejuk, dan tenang'],
            'sumber': 'National Sleep Foundation 2015'})
    elif sleep > 9:
        warning.append({'parameter': 'Durasi Tidur', 'kondisi': f'{sleep} jam/malam — Berlebihan',
            'rekomendasi': ['Tidur >9 jam dapat mengindikasikan masalah kesehatan tertentu',
                'Konsultasi dokter jika sering merasa lelah meski tidur lama'],
            'sumber': 'National Sleep Foundation 2015'})
    else:
        good.append(f'Durasi tidur normal ({sleep} jam/malam)')

    alcohol = user_input.get('alcohol_units_per_week', 0)
    if alcohol > 14:
        urgent.append({'parameter': 'Konsumsi Alkohol', 'kondisi': f'{alcohol} unit/minggu — Tinggi (Berisiko)',
            'rekomendasi': ['Kurangi konsumsi alkohol secara bertahap',
                'Target di bawah 14 unit/minggu, idealnya lebih rendah',
                'Cari dukungan profesional jika sulit mengurangi sendiri',
                'Alkohol berlebih meningkatkan risiko hipertensi dan kardiomiopati'],
            'sumber': 'WHO Alcohol Guidelines'})
    elif alcohol > 7:
        warning.append({'parameter': 'Konsumsi Alkohol', 'kondisi': f'{alcohol} unit/minggu — Sedang',
            'rekomendasi': ['Pertimbangkan untuk mengurangi ke bawah 7 unit/minggu',
                'Selipkan hari-hari bebas alkohol dalam seminggu'],
            'sumber': 'WHO Alcohol Guidelines'})
    else:
        good.append('Konsumsi alkohol dalam batas aman')

    stress = user_input.get('stress_level', 0)
    if stress >= 7:
        urgent.append({'parameter': 'Tingkat Stres', 'kondisi': f'{stress}/10 — Tinggi',
            'rekomendasi': ['Latihan pernapasan dalam (deep breathing) 5-10 menit/hari',
                'Meditasi atau mindfulness minimal 10 menit/hari',
                'Olahraga rutin terbukti signifikan menurunkan hormon stres',
                'Batasi paparan berita negatif dan media sosial',
                'Pertimbangkan konsultasi dengan psikolog atau konselor'],
            'sumber': 'AHA Stress & Heart Disease'})
    elif stress >= 4:
        warning.append({'parameter': 'Tingkat Stres', 'kondisi': f'{stress}/10 — Sedang',
            'rekomendasi': ['Luangkan waktu untuk hobi dan aktivitas relaksasi',
                'Jaga keseimbangan kerja dan istirahat'],
            'sumber': 'AHA Stress & Heart Disease'})
    else:
        good.append(f'Tingkat stres terkendali ({stress}/10)')

    diet = user_input.get('diet_quality_score', 0)
    if diet <= 3:
        urgent.append({'parameter': 'Kualitas Diet', 'kondisi': f'{diet}/10 — Buruk',
            'rekomendasi': ['Perbanyak konsumsi buah dan sayuran minimal 5 porsi/hari',
                'Kurangi makanan ultra-processed (mie instan, fast food, minuman manis)',
                'Ganti karbohidrat sederhana dengan karbohidrat kompleks (nasi merah, oat)',
                'Konsultasi ahli gizi untuk panduan diet yang terstruktur'],
            'sumber': 'AHA Lifestyle Guidelines'})
    elif diet <= 6:
        warning.append({'parameter': 'Kualitas Diet', 'kondisi': f'{diet}/10 — Cukup',
            'rekomendasi': ['Tingkatkan variasi sayuran dan buah dalam menu harian',
                'Kurangi konsumsi gula tambahan dan garam berlebih'],
            'sumber': 'AHA Lifestyle Guidelines'})
    else:
        good.append(f'Kualitas diet sudah baik ({diet}/10)')

    family_history = user_input.get('family_history_heart_disease', False)
    if family_history:
        warning.append({'parameter': 'Riwayat Keluarga', 'kondisi': 'Ada riwayat penyakit jantung dalam keluarga',
            'rekomendasi': ['Lakukan skrining jantung rutin minimal 1x per tahun',
                'Informasikan riwayat keluarga ke dokter untuk asesmen risiko genetik',
                'Jaga semua parameter gaya hidup lebih ketat dari rata-rata orang'],
            'sumber': 'AHA Family History & Heart Disease'})

    return {'urgent': urgent, 'warning': warning, 'good': good}
