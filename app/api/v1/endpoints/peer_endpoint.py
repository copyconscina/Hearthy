from fastapi import APIRouter, Depends, HTTPException
from app.schemas.peer_schema import PeersRequest, PeersResponse
from app.services.peer_analyzer import PeersAnalyzer
from app.core.dependencies import get_peers_analyzer

router = APIRouter()


@router.post("/peers", response_model=PeersResponse)
def peers(
    req: PeersRequest,
    analyzer: PeersAnalyzer = Depends(get_peers_analyzer),
) -> PeersResponse:
    """
    Bandingkan profil risiko pasien vs orang-orang seumuran dari dataset.

    Input: hasil prediksi + parameter klinis/gaya hidup pasien.

    Return:
    - **age_group_label**: kelompok usia pembanding
    - **total_peers**: jumlah data pembanding
    - **risk_score_percentile**: kamu lebih baik dari X% peers
    - **risk_category_distribution**: distribusi Low/Medium/High di kelompok usia
    - **parameter_comparison**: perbandingan per parameter vs rata-rata peers
    - **summary**: narasi ringkas
    """
    try:
        return analyzer.analyze(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Peers analysis error: {str(e)}")
