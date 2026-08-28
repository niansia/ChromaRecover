import cv2
import numpy as np

from chromarecover import recover
from chromarecover.semantics import (
    DigitMatch,
    _has_unresolved_six_eight,
    recognize_digit_sequence,
    semantic_consensus,
)
from chromarecover.synthetic import make_polygon_mosaic


def test_multifont_shape_match_recognizes_rendered_digits() -> None:
    mask = np.zeros((120, 300), dtype=np.uint8)
    cv2.putText(mask, "820", (12, 98), cv2.FONT_HERSHEY_DUPLEX, 3.2, 1, 8, cv2.LINE_AA)
    evidence = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3.0)
    evidence /= evidence.max()

    result = recognize_digit_sequence(evidence)

    assert result.hypothesis == "820"
    assert result.accepted


def test_sparse_generated_mosaic_abstains_from_unverified_digit_label() -> None:
    case = make_polygon_mosaic(size=256, text="820", seed=42, structured=True)

    result = recover(
        case.image,
        mode="camera-screen",
        top_k=3,
        semantics="auto",
        auto_rectify=False,
    )

    # Sparse primitive samples are insufficient to promote a digit guess to truth even
    # though the generator provides an exact structure label for evaluation.
    assert result.semantics["accepted"] is False
    assert result.semantics["hypothesis"] is None
    assert all(
        candidate.semantics["accepted"] or candidate.semantics["hypothesis"] is None
        for candidate in result.candidates
        if candidate.semantics
    )


def test_recovery_integrates_digit_consensus_after_mask_selection() -> None:
    image = np.full((180, 420, 3), 150, dtype=np.uint8)
    cv2.putText(
        image,
        "820",
        (25, 140),
        cv2.FONT_HERSHEY_DUPLEX,
        3.5,
        (166, 130, 153),
        8,
        cv2.LINE_AA,
    )

    result = recover(image, semantics="auto", top_k=3)

    assert result.semantics["hypothesis"] == "820"
    assert result.semantics["accepted"] is True
    assert result.status == "ok"


def test_digit_semantics_does_not_change_mask_first_default() -> None:
    image = np.full((128, 128, 3), 150, dtype=np.uint8)

    result = recover(image, semantics="auto", top_k=2)

    assert result.semantics["accepted"] is False
    assert result.status == "uncertain"


def test_held_out_script_style_never_accepts_a_wrong_digit() -> None:
    accepted = 0
    for digit in "0123456789":
        mask = np.zeros((100, 80), dtype=np.uint8)
        cv2.putText(
            mask,
            digit,
            (7, 82),
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
            2.6,
            1,
            5,
            cv2.LINE_AA,
        )
        evidence = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)
        evidence /= max(float(evidence.max()), 1.0)

        result = recognize_digit_sequence(evidence)
        if result.accepted:
            accepted += 1
            assert result.hypothesis == digit

    assert accepted >= 8


def test_randomized_negative_patterns_never_promote_a_digit() -> None:
    from chromarecover.synthetic import make_polygon_mosaic

    for seed in range(3):
        case = make_polygon_mosaic(size=160, seed=seed, structured=False)
        result = recover(case.image, semantics="auto", top_k=3)
        assert result.semantics["accepted"] is False


def test_close_six_eight_shape_match_abstains() -> None:
    ambiguous = DigitMatch(
        label="8",
        confidence=0.84,
        margin=0.055,
        bbox=(0, 0, 10, 20),
        alternatives=(("8", 0.82), ("6", 0.765), ("9", 0.71)),
    )

    assert _has_unresolved_six_eight([ambiguous])


def test_semantic_shape_cannot_override_weak_visual_evidence() -> None:
    consensus = semantic_consensus(
        [
            {
                "hypothesis": "8",
                "confidence": 0.99,
                "accepted": True,
                "candidate_rank": 1,
                "candidate_decision_confidence": 0.40,
            }
        ]
    )

    assert consensus["accepted"] is False
    assert consensus["hypothesis"] is None
    assert consensus["reason"] == "weak_visual_evidence"


def test_cross_family_shape_can_be_reported_without_promoting_visual_status() -> None:
    consensus = semantic_consensus(
        [
            {
                "hypothesis": "82",
                "confidence": 0.90,
                "accepted": True,
                "candidate_decision_confidence": 0.32,
                "candidate_capture_confidence": 0.53,
                "candidate_source_family": "global_lab",
                "characters": [{"margin": 0.09}],
                "candidate_boundary_evidence": 0.90,
                "candidate_color_separation": 0.80,
                "candidate_weak_structure_evidence": False,
                "candidate_quality_blocked": False,
            },
            {
                "hypothesis": "82",
                "confidence": 0.88,
                "accepted": True,
                "candidate_decision_confidence": 0.31,
                "candidate_capture_confidence": 0.52,
                "candidate_source_family": "local_lab",
                "characters": [{"margin": 0.08}],
                "candidate_boundary_evidence": 0.85,
                "candidate_color_separation": 0.82,
                "candidate_weak_structure_evidence": False,
                "candidate_quality_blocked": False,
            },
            {
                "hypothesis": "82",
                "confidence": 0.86,
                "accepted": True,
                "candidate_decision_confidence": 0.30,
                "candidate_capture_confidence": 0.51,
                "candidate_source_family": "global_lab",
                "characters": [{"margin": 0.07}],
                "candidate_boundary_evidence": 0.52,
                "candidate_color_separation": 0.75,
                "candidate_weak_structure_evidence": False,
                "candidate_quality_blocked": False,
            },
        ]
    )

    assert consensus["accepted"] is True
    assert consensus["hypothesis"] == "82"
    assert consensus["reason"] == "cross_family_shape_validation"
