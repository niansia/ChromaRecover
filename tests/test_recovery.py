from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from chromarecover import RecoverConfig, recover
from chromarecover.color import build_representations
from chromarecover.exceptions import InvalidInputError
from chromarecover.quality import assess_quality
from chromarecover.synthetic import make_chromatic_pattern, make_polygon_mosaic


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union)


def test_recovers_structure_in_top_three() -> None:
    case = make_chromatic_pattern(seed=7, structured=True)
    result = recover(case.image, mode="digital", top_k=3)

    assert result.status == "ok"
    assert case.mask is not None
    assert max(_iou(candidate.mask, case.mask) for candidate in result.candidates) >= 0.70
    assert result.best is not None
    assert result.best.overall_confidence >= 0.70


def test_rotation_keeps_structure_recoverable() -> None:
    case = make_chromatic_pattern(seed=11, structured=True)
    rotated_image = np.rot90(case.image)
    rotated_mask = np.rot90(case.mask)
    result = recover(rotated_image, mode="digital", top_k=5)

    assert max(_iou(candidate.mask, rotated_mask) for candidate in result.candidates) >= 0.62


def test_generated_820_mosaic_support_is_retained_in_top_three() -> None:
    case = make_polygon_mosaic(size=256, text="820", seed=42, structured=True)
    assert case.support_mask is not None

    result = recover(case.image, mode="digital", top_k=3)

    assert max(
        _iou(candidate.mask, case.support_mask) for candidate in result.candidates
    ) >= 0.75


def test_ok_status_uses_the_public_decision_confidence_contract() -> None:
    case = make_chromatic_pattern(size=192, seed=7, structured=True)
    result = recover(case.image, mode="digital", top_k=3)

    assert result.status == "ok"
    assert result.best is not None
    assert result.best.decision_confidence >= result.config["confidence_threshold"]


def test_uncalibrated_camera_mosaic_does_not_promote_score_into_confidence() -> None:
    case = make_polygon_mosaic(size=256, text="820", seed=42, structured=True)

    result = recover(case.image, mode="camera-screen", top_k=3, auto_rectify=False)

    assert result.status == "uncertain"
    assert result.best is not None
    assert result.best.metrics["mosaic_distribution_score"] >= 0.77
    assert result.best.capture_confidence < result.config["confidence_threshold"]
    assert result.best.decision_confidence < result.best.capture_confidence


def test_blur_is_measured_before_analysis_downsampling() -> None:
    small = make_polygon_mosaic(size=256, text="820", seed=42).image
    large = cv2.resize(small, (2052, 1491), interpolation=cv2.INTER_CUBIC)
    analysis = cv2.resize(large, (384, 279), interpolation=cv2.INTER_AREA)
    colors = build_representations(analysis, RecoverConfig().local_sigma_ratio)
    blurred = cv2.GaussianBlur(large.astype(np.float32), (0, 0), 8.0).astype(np.uint8)

    sharp_report = assess_quality(analysis, colors, blur_rgb=large)
    blurred_report = assess_quality(analysis, colors, blur_rgb=blurred)

    assert blurred_report.blur > sharp_report.blur + 0.45
    assert "severe_blur" in blurred_report.warnings


def test_random_dot_field_abstains() -> None:
    case = make_chromatic_pattern(seed=7, structured=False)
    result = recover(case.image, mode="digital", top_k=3)

    assert result.status == "uncertain"
    assert result.best is not None
    assert result.best.overall_confidence < 0.70


def test_top_k_does_not_hide_ambiguity() -> None:
    case = make_chromatic_pattern(size=192, seed=0, structured=True)
    one = recover(case.image, mode="digital", top_k=1)
    three = recover(case.image, mode="digital", top_k=3)

    assert one.status == three.status
    assert one.best is not None and three.best is not None
    assert one.best.id == three.best.id


def test_smooth_color_cast_with_checker_texture_abstains() -> None:
    size = 192
    yy, xx = np.mgrid[0:size, 0:size]
    horizontal = xx / (size - 1)
    image = np.empty((size, size, 3), dtype=np.float32)
    image[..., 0] = 110 + 70 * horizontal
    image[..., 1] = 165 - 45 * horizontal
    image[..., 2] = 110 + 10 * horizontal
    checker = ((xx + yy) % 2) * 10 - 5
    image = np.clip(image + checker[..., None], 0, 255).astype(np.uint8)

    result = recover(image, mode="digital", top_k=3)

    assert result.status == "uncertain"
    assert result.best is not None
    assert result.best.overall_confidence < 0.74


def test_smooth_rgb_field_with_achromatic_stripes_abstains() -> None:
    size = 160
    yy, xx = np.mgrid[:size, :size]
    x_normalized = xx / (size - 1)
    y_normalized = yy / (size - 1)
    base = np.array([167.395605, 133.887844, 175.859792])
    horizontal = np.array([27.631524, -56.815171, 66.587129])
    vertical = np.array([26.113970, 28.606431, -37.188637])
    texture = 7.855017 * (2 * (((xx + yy + 1) % 5) < 2.5) - 1)
    image = np.clip(
        base
        + horizontal * x_normalized[..., None]
        + vertical * y_normalized[..., None]
        + texture[..., None],
        0,
        255,
    ).astype(np.uint8)

    result = recover(image, mode="digital", top_k=3)

    assert result.status == "uncertain"
    assert result.best is not None
    assert result.best.overall_confidence < 0.74


def test_tiny_hard_negative_is_not_confident() -> None:
    case = make_chromatic_pattern(size=96, seed=3, structured=False)
    result = recover(case.image, mode="digital", top_k=1)

    assert result.status == "uncertain"
    assert "low_effective_resolution" in result.quality.warnings
    assert result.best is not None
    assert result.best.overall_confidence < 0.60


def test_128px_hard_negative_is_not_confident() -> None:
    case = make_chromatic_pattern(size=128, seed=5, structured=False)
    result = recover(case.image, mode="digital", top_k=3)

    assert result.status == "uncertain"
    assert result.best is not None
    assert result.best.overall_confidence < 0.60


def test_sparse_equal_chroma_hue_structure_is_not_suppressed() -> None:
    size = 192
    target = np.zeros((size, size), dtype=np.uint8)
    cv2.putText(target, "8", (55, 145), cv2.FONT_HERSHEY_DUPLEX, 3.4, 1, 12, cv2.LINE_AA)
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[:] = [205, 210, 70]
    image[target.astype(bool)] = [210, 70, 167]

    result = recover(image, mode="digital", top_k=3)

    assert "insufficient_chromatic_variation" not in result.quality.warnings
    assert max(_iou(candidate.mask, target.astype(bool)) for candidate in result.candidates) > 0.95
    assert result.best is not None
    assert result.best.overall_confidence > 0.70


def test_low_luminance_contrast_color_text_is_not_mislabeled_as_blur() -> None:
    size = 220
    target = np.zeros((size, size), dtype=np.uint8)
    cv2.putText(target, "A7", (25, 150), cv2.FONT_HERSHEY_DUPLEX, 3.2, 1, 10, cv2.LINE_AA)
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[:] = [150, 145, 120]
    image[target.astype(bool)] = [154, 140, 124]

    result = recover(image, mode="digital", top_k=3)

    assert result.status == "ok"
    assert result.quality.blur < 0.20
    assert result.best is not None
    assert _iou(result.best.mask, target.astype(bool)) > 0.98


def test_retry_status_reduces_decision_confidence() -> None:
    case = make_chromatic_pattern(size=192, seed=7, structured=True)
    blurred = cv2.GaussianBlur(case.image.astype(np.float32), (0, 0), 8.0).astype(np.uint8)

    result = recover(blurred, mode="camera", top_k=3, auto_rectify=False)

    assert result.status == "retry_recommended"
    assert result.best is not None
    assert result.best.structure_confidence >= result.best.capture_confidence
    assert result.best.decision_confidence < result.best.capture_confidence * 0.40


def test_save_writes_versioned_artifacts(tmp_path: Path) -> None:
    case = make_chromatic_pattern(size=160, seed=3, structured=True)
    result = recover(case.image, mode="digital", top_k=2)
    result_path = result.save(tmp_path)

    assert result_path.is_file()
    assert (tmp_path / "mask_01.png").is_file()
    assert (tmp_path / "overlay_01.png").is_file()
    assert Image.open(tmp_path / "mask_01.png").size == (160, 160)
    payload = result.to_dict()
    assert payload["schema_version"] == "0.5"
    assert (tmp_path / "structure_01.png").is_file()
    assert (tmp_path / "evidence_01.png").is_file()
    assert payload["candidates"][0]["structure_confidence"] >= payload["candidates"][0][
        "decision_confidence"
    ]
    assert payload["algorithm_version"].startswith("cre-")
    assert payload["config"]["analysis_max_side"] == 384
    assert {"numpy", "opencv", "pillow"} <= payload["runtime"].keys()


def test_empty_image_returns_normal_result_without_best_candidate() -> None:
    image = np.full((192, 256, 3), 255, dtype=np.uint8)

    result = recover(image, mode="auto")

    assert result.status == "uncertain"
    assert result.candidates == []
    assert result.best is None


def test_presentation_artifacts_are_bounded_but_masks_remain_full_resolution(
    tmp_path: Path,
) -> None:
    case = make_chromatic_pattern(size=320, seed=7, structured=True)
    config = RecoverConfig(presentation_max_side=128)

    result = recover(case.image, config=config, top_k=1)
    assert result.best is not None
    result.save(tmp_path)

    candidate = result.best
    assert candidate.mask.shape == (320, 320)
    assert candidate.structure_mask is not None
    assert candidate.structure_mask.shape == (320, 320)
    assert candidate.evidence_map is not None
    assert candidate.evidence_map.shape == (128, 128)
    assert candidate.overlay is not None
    assert candidate.overlay.shape[:2] == (128, 128)
    assert Image.open(tmp_path / "mask_01.png").size == (320, 320)
    assert Image.open(tmp_path / "evidence_01.png").size == (128, 128)
    assert Image.open(tmp_path / "overlay_01.png").size == (128, 128)
    assert candidate.transform["artifact_spaces"]["overlay"]["coordinate_space"] == (
        "presentation"
    )


def test_nearby_image_scales_do_not_flip_status_at_a_consensus_threshold() -> None:
    case = make_polygon_mosaic(size=256, text="820", seed=42, structured=True)
    first = cv2.resize(case.image, (400, 300), interpolation=cv2.INTER_AREA)
    second = cv2.resize(case.image, (412, 300), interpolation=cv2.INTER_AREA)

    first_result = recover(first, mode="digital", top_k=3)
    second_result = recover(second, mode="digital", top_k=3)

    assert first_result.status == second_result.status
    assert first_result.best is not None and second_result.best is not None
    assert abs(
        first_result.best.capture_confidence - second_result.best.capture_confidence
    ) < 0.08


def test_debug_bundle_is_materialized_only_when_requested(tmp_path: Path) -> None:
    case = make_chromatic_pattern(size=128, seed=5, structured=True)
    result = recover(case.image, return_debug=True, top_k=1)

    result.save(tmp_path)

    assert result.debug_artifacts
    assert (tmp_path / "debug" / "analysis_input.png").is_file()
    assert (tmp_path / "debug" / "primitive_labels.png").is_file()


def test_pixel_limit_is_checked_before_processing() -> None:
    oversized_for_test = np.zeros((11, 10, 3), dtype=np.uint8)
    config = RecoverConfig(max_pixels=100)

    try:
        recover(oversized_for_test, config=config)
    except InvalidInputError as error:
        assert "allowed maximum" in str(error)
    else:
        raise AssertionError("Expected InvalidInputError")


def test_default_pixel_limit_matches_practical_memory_envelope() -> None:
    assert RecoverConfig().max_pixels == 50_000_000
    assert RecoverConfig().max_pixels >= 8_064 * 6_048


def test_invalid_config_fails_with_public_value_error() -> None:
    try:
        RecoverConfig(cluster_counts=(0,))
    except ValueError as error:
        assert "cluster_counts" in str(error)
    else:
        raise AssertionError("Expected ValueError")
