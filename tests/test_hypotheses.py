import numpy as np

from chromarecover.config import RecoverConfig
from chromarecover.hypotheses import MaskHypothesis, _deduplicate


def test_matrix_deduplication_preserves_deterministic_first_winner() -> None:
    first = np.zeros((64, 64), dtype=bool)
    first[8:32, 8:32] = True
    duplicate = first.copy()
    different = np.zeros_like(first)
    different[35:58, 35:58] = True
    hypotheses = [
        MaskHypothesis(first, "first", "a"),
        MaskHypothesis(duplicate, "duplicate", "b"),
        MaskHypothesis(different, "different", "c"),
    ]

    kept = _deduplicate(hypotheses, RecoverConfig())

    assert [hypothesis.source for hypothesis in kept] == ["first", "different"]
