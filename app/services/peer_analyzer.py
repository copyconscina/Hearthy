"""
PeersAnalyzer: bandingkan data pasien vs orang seumuran dari dataset.
Dataset di-load sekali saat startup, query per request.
"""
import pandas as pd
import numpy as np
from app.schemas.peer_schema import PeersRequest, PeersResponse, PeerStat

# Mapping age_group dari dataset (hasil reverse engineer dari data)
# age < 30 → 6, 30-39 → 0, 40-49 → 1, 50-59 → 2, 60-69 → 3, 70-79 → 4, 80+ → 5
AGE_GROUP_LABELS = {
    6: "< 30 tahun",
    0: "30-39 tahun",
    1: "40-49 tahun",
    2: "50-59 tahun",
    3: "60-69 tahun",
    4: "70-79 tahun",
    5: ">= 80 tahun",
}

RISK_MAP = {0: "Low", 1: "Medium", 2: "High"}

COMPARE_PARAMS = [
    ("systolic_bp",                      "Tekanan Darah Sistolik",  True),   # True = lebih rendah lebih baik
    ("diastolic_bp",                     "Tekanan Darah Diastolik", True),
    ("cholesterol_mg_dl",                "Kolesterol",              True),
    ("bmi",                              "BMI",                     True),
    ("daily_steps",                      "Langkah per Hari",        False),  # False = lebih tinggi lebih baik
    ("physical_activity_hours_per_week", "Aktivitas Fisik",         False),
    ("sleep_hours",                      "Jam Tidur",               False),
    ("stress_level",                     "Tingkat Stres",           True),
    ("diet_quality_score",               "Kualitas Diet",           False),
]


def _get_age_group(age: float) -> int:
    if age < 30:
        return 6
    elif age < 40:
        return 0
    elif age < 50:
        return 1
    elif age < 60:
        return 2
    elif age < 70:
        return 3
    elif age < 80:
        return 4
    else:
        return 5


class PeersAnalyzer:
    def __init__(self, dataset_path: str):
        self._df = pd.read_csv(dataset_path)
        # Pastikan risk_category sudah ada sebagai string
        if "risk_category" not in self._df.columns:
            raise ValueError("Dataset tidak punya kolom risk_category")

    def analyze(self, req: PeersRequest) -> PeersResponse:
        age_group = _get_age_group(req.age)
        peers = self._df[self._df["age_group"] == age_group].copy()
        total_peers = len(peers)

        if total_peers == 0:
            # Fallback: pakai seluruh dataset kalau tidak ada peers
            peers = self._df.copy()
            total_peers = len(peers)

        age_group_label = AGE_GROUP_LABELS.get(age_group, f"Usia {int(req.age)} tahun")

        # ── Risk score percentile ─────────────────────────────────
        # Risk score lebih rendah = lebih baik
        peer_scores = peers["heart_disease_risk_score"].dropna().values
        risk_score_percentile = float(
            np.mean(peer_scores > req.risk_score) * 100
        )

        # ── Distribusi risk category peers ───────────────────────
        cat_counts = peers["risk_category"].value_counts(normalize=True) * 100
        # risk_category di dataset bisa berupa int (0,1,2) atau string
        risk_category_distribution = {}
        for k, v in cat_counts.items():
            label = RISK_MAP.get(int(k), str(k)) if str(k).isdigit() else str(k)
            risk_category_distribution[label] = round(float(v), 1)

        # ── Parameter comparison ──────────────────────────────────
        parameter_comparison = []
        for col, label, lower_is_better in COMPARE_PARAMS:
            user_val = getattr(req, col, None)
            if user_val is None or col not in peers.columns:
                continue

            peer_vals = peers[col].dropna().values
            if len(peer_vals) == 0:
                continue

            mean_val = float(np.mean(peer_vals))
            selisih  = float(user_val) - mean_val

            # Persentil: seberapa besar % peers yang kondisinya lebih buruk dari kita
            if lower_is_better:
                # Untuk parameter yang lebih rendah lebih baik (BP, kolesterol, dll)
                # Kita lebih baik jika nilai kita LEBIH RENDAH dari peers
                pct = float(np.mean(peer_vals > user_val) * 100)
            else:
                # Untuk parameter yang lebih tinggi lebih baik (steps, aktivitas, dll)
                pct = float(np.mean(peer_vals < user_val) * 100)

            if pct >= 60:
                status = "Lebih Baik dari Peers"
            elif pct >= 40:
                status = "Rata-rata"
            else:
                status = "Perlu Diperbaiki"

            parameter_comparison.append(PeerStat(
                parameter=label,
                nilai_kamu=round(float(user_val), 1),
                rata_rata_peers=round(mean_val, 1),
                persentil=round(pct, 1),
                selisih=round(selisih, 1),
                status=status,
            ))

        # ── Summary narasi ────────────────────────────────────────
        better_count = sum(1 for p in parameter_comparison if p.status == "Lebih Baik dari Peers")
        total_params = len(parameter_comparison)

        if risk_score_percentile >= 70:
            summary = (
                f"Skor risiko Anda ({req.risk_score:.1f}) lebih rendah dari "
                f"{risk_score_percentile:.0f}% orang di kelompok usia {age_group_label}. "
                f"Anda lebih baik dari rata-rata peers di {better_count} dari {total_params} parameter yang dibandingkan."
            )
        elif risk_score_percentile >= 40:
            summary = (
                f"Skor risiko Anda ({req.risk_score:.1f}) berada di kisaran rata-rata "
                f"kelompok usia {age_group_label} (lebih baik dari {risk_score_percentile:.0f}% peers). "
                f"Ada {total_params - better_count} parameter yang masih bisa ditingkatkan."
            )
        else:
            summary = (
                f"Skor risiko Anda ({req.risk_score:.1f}) lebih tinggi dari rata-rata "
                f"kelompok usia {age_group_label} — hanya {risk_score_percentile:.0f}% peers yang lebih buruk. "
                f"Fokus perbaikan pada parameter dengan status 'Perlu Diperbaiki'."
            )

        return PeersResponse(
            age_group_label=age_group_label,
            total_peers=total_peers,
            risk_score_percentile=round(risk_score_percentile, 1),
            risk_category_distribution=risk_category_distribution,
            parameter_comparison=parameter_comparison,
            summary=summary,
        )
