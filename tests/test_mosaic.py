import numpy as np

from chromarecover import recover
from chromarecover.synthetic import make_polygon_mosaic


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 1.0


def test_matched_histogram_polygon_mosaic_recovers_structure_envelope() -> None:
    case = make_polygon_mosaic(size=256, seed=42, structured=True)
    assert case.mask is not None

    result = recover(case.image, mode="digital", top_k=5)

    assert max(
        _iou(candidate.structure_mask, case.mask)
        for candidate in result.candidates
        if candidate.structure_mask is not None
    ) >= 0.30
    assert all(candidate.evidence_map is not None for candidate in result.candidates)


def test_matched_histogram_randomized_mosaic_abstains() -> None:
    case = make_polygon_mosaic(size=256, seed=42, structured=False)

    result = recover(case.image, mode="digital", top_k=3)

    assert result.status == "uncertain"
    assert result.best is not None
    assert result.best.decision_confidence < 0.74
