import cv2
import numpy as np
import pytest

from chromarecover import recover
from chromarecover.exceptions import InvalidInputError
from chromarecover.geometry import (
    detect_content_quad,
    detect_document_quad,
    inverse_map_evidence,
    inverse_map_mask,
    normalize_geometry,
)
from chromarecover.photometry import normalize_camera_photometry
from chromarecover.synthetic import make_chromatic_pattern, make_polygon_mosaic


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 1.0


def _synthetic_screen_signature(height: int = 220, width: int = 280) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    wave = np.sin(2 * np.pi * (xx / 5.0 + yy / 8.0))
    image = np.full((height, width, 3), 160.0, dtype=np.float32)
    image[..., 0] += 20 * wave
    image[..., 1] -= 14 * wave
    image[..., 2] += 10 * wave
    return np.clip(image, 0, 255).astype(np.uint8)


def test_explicit_perspective_inverse_mapping_round_trips_mask() -> None:
    rectified = np.zeros((160, 240), dtype=np.uint8)
    cv2.putText(rectified, "820", (18, 112), cv2.FONT_HERSHEY_DUPLEX, 2.3, 1, 8, cv2.LINE_AA)
    corners = np.float32([[42, 28], [310, 12], [332, 222], [24, 238]])
    destination = np.float32([[0, 0], [239, 0], [239, 159], [0, 159]])
    rectified_to_camera = cv2.getPerspectiveTransform(destination, corners)
    camera_mask = cv2.warpPerspective(rectified, rectified_to_camera, (360, 260)) > 0
    camera_rgb = np.full((260, 360, 3), 180, dtype=np.uint8)

    geometry = normalize_geometry(
        camera_rgb, mode="camera", max_side=384, document_corners=corners
    )
    normalized_mask = cv2.warpPerspective(
        camera_mask.astype(np.uint8),
        geometry.forward_matrix,
        (geometry.rgb.shape[1], geometry.rgb.shape[0]),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    mapped = inverse_map_mask(normalized_mask, geometry, camera_rgb.shape[:2])

    assert geometry.metadata["source"] == "user_corners"
    assert geometry.perspective_score > 0.05
    assert _iou(mapped, camera_mask) > 0.90


@pytest.mark.parametrize(
    ("corners", "message"),
    [
        ([[float("nan"), 10], [90, 10], [90, 90], [10, 90]], "finite"),
        ([[float("inf"), 10], [90, 10], [90, 90], [10, 90]], "finite"),
        ([[10, 10], [10, 10], [90, 90], [10, 90]], "distinct"),
        ([[10, 10], [50, 10], [90, 10], [10, 90]], "convex"),
        (
            [[-1_000_000_000, 10], [90, 10], [90, 90], [10, 90]],
            "coordinate margin",
        ),
    ],
)
def test_user_document_corners_reject_nonfinite_or_degenerate_input(
    corners: list[list[float]], message: str
) -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(InvalidInputError, match=message):
        normalize_geometry(
            image,
            mode="camera",
            max_side=384,
            document_corners=corners,
        )


def test_symmetric_diamond_corners_remain_four_unique_points() -> None:
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    corners = [[60, 5], [114, 60], [60, 114], [5, 60]]

    geometry = normalize_geometry(
        image,
        mode="camera",
        max_side=384,
        document_corners=corners,
    )

    ordered = np.asarray(geometry.metadata["corners"])
    assert len(np.unique(ordered, axis=0)) == 4
    assert np.all(np.isfinite(geometry.forward_matrix))


def test_automatic_document_quad_detection_rectifies_large_page() -> None:
    image = np.full((300, 420, 3), 35, dtype=np.uint8)
    expected = np.array([[55, 35], [370, 22], [392, 270], [37, 282]], dtype=np.int32)
    cv2.fillConvexPoly(image, expected, (225, 220, 205))
    cv2.polylines(image, [expected], True, (250, 250, 250), 4)
    cv2.putText(image, "820", (105, 175), cv2.FONT_HERSHEY_DUPLEX, 2, (180, 155, 180), 5)

    detected = detect_document_quad(image)
    geometry = normalize_geometry(image, mode="camera", max_side=384)

    assert detected is not None
    assert np.max(np.linalg.norm(detected - expected, axis=1)) < 8
    assert geometry.metadata["source"] == "auto_quad"


def test_low_contrast_borderless_document_uses_content_quad_fallback() -> None:
    image = np.full((360, 520, 3), [63, 76, 119], dtype=np.uint8)
    expected = np.array([[72, 54], [458, 45], [472, 309], [58, 320]], dtype=np.int32)
    cv2.fillConvexPoly(image, expected, (206, 194, 189))
    cv2.putText(image, "REPORT", (145, 184), cv2.FONT_HERSHEY_DUPLEX, 1.2, (169, 157, 153), 3)
    image = cv2.GaussianBlur(image, (0, 0), 1.4)

    detected = detect_content_quad(image)

    assert detected is not None
    assert np.max(np.linalg.norm(detected - expected, axis=1)) < 24


def test_evidence_inverse_mapping_can_target_scaled_original_space() -> None:
    image = np.full((400, 600, 3), 150, dtype=np.uint8)
    corners = np.float32([[80, 55], [525, 35], [548, 348], [62, 365]])
    geometry = normalize_geometry(
        image,
        mode="camera",
        max_side=384,
        document_corners=corners,
    )
    evidence = np.ones(geometry.rgb.shape[:2], dtype=np.float32)

    mapped = inverse_map_evidence(
        evidence,
        geometry,
        (100, 150),
        coordinate_shape=(400, 600),
    )

    assert mapped.shape == (100, 150)
    assert mapped.dtype == np.float32
    assert 0.35 < float(np.mean(mapped > 0.5)) < 0.80


def test_camera_recovery_maps_perspective_structure_to_original_photo() -> None:
    case = make_chromatic_pattern(size=128, seed=7, structured=True)
    assert case.mask is not None
    corners = np.float32([[22, 28], [190, 16], [198, 188], [12, 200]])
    source_corners = np.float32([[0, 0], [127, 0], [127, 127], [0, 127]])
    transform = cv2.getPerspectiveTransform(source_corners, corners)
    camera = np.full((220, 220, 3), 210, dtype=np.uint8)
    warped = cv2.warpPerspective(case.image, transform, (220, 220), borderValue=(210, 210, 210))
    page = cv2.warpPerspective(np.ones((128, 128), np.uint8), transform, (220, 220)) > 0
    camera[page] = warped[page]
    expected = cv2.warpPerspective(
        case.mask.astype(np.uint8), transform, (220, 220), flags=cv2.INTER_NEAREST
    ).astype(bool)

    result = recover(
        camera,
        mode="camera",
        top_k=3,
        document_corners=corners.tolist(),
    )

    assert max(_iou(candidate.mask, expected) for candidate in result.candidates) > 0.75
    assert all(candidate.mask.shape == camera.shape[:2] for candidate in result.candidates)


def test_camera_photometry_marks_glare_and_reduces_shadow_field() -> None:
    height, width = 180, 260
    _yy, xx = np.mgrid[:height, :width]
    shade = 0.45 + 0.55 * xx / (width - 1)
    image = np.empty((height, width, 3), dtype=np.float32)
    image[:] = np.array([186, 151, 126], dtype=np.float32)
    image *= shade[..., None]
    image[55:105, 95:155] = 255
    image = np.clip(image, 0, 255).astype(np.uint8)

    result = normalize_camera_photometry(image)
    before_l = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)[..., 0]
    after_l = cv2.cvtColor(
        result.rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB
    )[..., 0]
    before_span = np.percentile(before_l, 90) - np.percentile(before_l, 10)
    after_span = np.percentile(after_l, 90) - np.percentile(after_l, 10)

    assert result.invalid_mask[80, 125]
    assert result.glare_score > 0.03
    assert "possible_glare" in result.warnings
    assert after_span < before_span * 0.55


def test_periodic_color_moire_is_detected_and_suppressed() -> None:
    height, width = 220, 280
    yy, xx = np.mgrid[:height, :width]
    wave = np.sin(2 * np.pi * (xx / 5.0 + yy / 8.0))
    image = np.full((height, width, 3), 160.0, dtype=np.float32)
    image[..., 0] += 30 * wave
    image[..., 1] -= 20 * wave
    image[..., 2] += 15 * wave
    image = np.clip(image, 0, 255).astype(np.uint8)

    result = normalize_camera_photometry(image)
    raw_lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    corrected_lab = cv2.cvtColor(
        result.rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB
    )
    raw_chroma_energy = np.std(raw_lab[..., 1:3])
    corrected_chroma_energy = np.std(corrected_lab[..., 1:3])

    assert result.moire_score > 0.42
    assert "possible_moire" in result.warnings
    assert "camera_moire_chroma_suppression" in result.steps
    assert corrected_chroma_energy < raw_chroma_energy * 0.9


def test_generated_mosaic_camera_branch_keeps_exact_support_in_top_three() -> None:
    case = make_polygon_mosaic(size=256, text="820", seed=42, structured=True)
    assert case.support_mask is not None

    result = recover(case.image, mode="camera", top_k=3, auto_rectify=False)

    assert max(
        _iou(candidate.mask, case.support_mask) for candidate in result.candidates
    ) >= 0.75
    assert result.input_info["photometry"]["applied"] is True
    assert "inverse_mask_mapping" in result.preprocess
    assert result.best is not None
    assert result.best.transform["interpolation"] == "nearest"


def test_screen_and_paper_profiles_are_reported_separately() -> None:
    image = _synthetic_screen_signature()

    screen = recover(image, mode="camera-screen", top_k=1, auto_rectify=False)
    paper = recover(image, mode="camera-paper", top_k=1, auto_rectify=False)

    assert screen.input_info["photometry"]["capture_profile"] == "camera-screen"
    assert paper.input_info["photometry"]["capture_profile"] == "camera-paper"


def test_auto_routes_strong_screen_signature_but_not_digital_pattern() -> None:
    screen_capture = _synthetic_screen_signature()

    screen = recover(screen_capture, mode="auto", top_k=1, auto_rectify=False)
    digital_case = make_chromatic_pattern(size=192, seed=7, structured=True)
    digital = recover(digital_case.image, mode="auto", top_k=1)

    assert screen.input_info["mode_resolved"] == "camera-screen"
    assert screen.input_info["auto_route"]["rule"] == "strong_screen_signature_v1"
    assert digital.input_info["mode_resolved"] == "digital"


def test_curved_page_uses_reversible_dense_ribbon_mapping() -> None:
    flat = np.full((150, 250, 3), [220, 212, 188], dtype=np.uint8)
    flat_mask = np.zeros((150, 250), dtype=np.uint8)
    cv2.putText(flat_mask, "A7", (35, 108), cv2.FONT_HERSHEY_DUPLEX, 2.5, 1, 7)
    flat[flat_mask.astype(bool)] = [130, 72, 105]
    camera = np.full((230, 310, 3), 25, dtype=np.uint8)
    expected = np.zeros(camera.shape[:2], dtype=bool)
    for x in range(250):
        camera_x = x + 30
        top = round(28 + 13 * np.sin(2 * np.pi * x / 249))
        bottom = round(197 - 10 * np.sin(2 * np.pi * x / 249))
        column = cv2.resize(flat[:, x : x + 1], (1, bottom - top + 1))
        mask_column = cv2.resize(
            flat_mask[:, x : x + 1],
            (1, bottom - top + 1),
            interpolation=cv2.INTER_NEAREST,
        )
        camera[top : bottom + 1, camera_x] = column[:, 0]
        expected[top : bottom + 1, camera_x] = mask_column[:, 0] > 0

    geometry = normalize_geometry(camera, mode="camera", max_side=384, auto_dewarp=True)
    normalized_mask = cv2.remap(
        expected.astype(np.uint8),
        geometry.sampling_map_x,
        geometry.sampling_map_y,
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    mapped = inverse_map_mask(normalized_mask, geometry, camera.shape[:2])

    assert geometry.metadata["source"] == "auto_dense_ribbon"
    assert _iou(mapped, expected) > 0.68
