import cv2
import numpy as np
import pytest

from chromarecover import RecoverConfig, recover_burst
from chromarecover.burst import fuse_burst
from chromarecover.exceptions import InvalidInputError
from chromarecover.synthetic import make_chromatic_pattern


def test_burst_fusion_recovers_pixels_hidden_by_moving_glare() -> None:
    base = make_chromatic_pattern(size=160, seed=7, structured=True).image
    frames: list[np.ndarray] = []
    for x in (10, 60, 110):
        frame = base.copy()
        frame[40:105, x : x + 38] = 255
        frames.append(frame)

    fusion = fuse_burst(frames)

    assert np.mean(np.abs(fusion.rgb.astype(float) - base.astype(float))) < 1.0
    assert np.mean(fusion.valid_count > 0) > 0.97
    assert not fusion.rejected_frames


def test_recover_burst_exposes_coverage_and_uncertainty() -> None:
    case = make_chromatic_pattern(size=128, seed=9, structured=True)
    frames = [case.image.copy() for _index in range(3)]
    frames[0][20:70, 10:48] = 255
    frames[1][20:70, 45:83] = 255
    frames[2][20:70, 80:118] = 255

    result = recover_burst(
        frames,
        mode="camera-screen",
        top_k=3,
        return_debug=True,
        auto_rectify=False,
    )

    assert result.input_info["burst"]["aligned_count"] == 3
    assert result.input_info["burst"]["input_total_pixels"] == 3 * 128 * 128
    assert result.input_info["burst"]["total_pixel_limit"] == 80_000_000
    assert result.quality.recoverable_fraction > 0.95
    assert "burst_uncertainty" in result.debug_artifacts


def test_burst_registers_small_handheld_translation() -> None:
    base = make_chromatic_pattern(size=192, seed=13, structured=True).image
    shifted = cv2.warpAffine(
        base,
        np.float32([[1, 0, 5], [0, 1, -4]]),
        (192, 192),
        borderMode=cv2.BORDER_REFLECT,
    )
    third = cv2.warpAffine(
        base,
        np.float32([[1, 0, -3], [0, 1, 4]]),
        (192, 192),
        borderMode=cv2.BORDER_REFLECT,
    )

    fusion = fuse_burst([base, shifted, third])

    assert not fusion.rejected_frames
    assert any(not np.allclose(transform, np.eye(3)) for transform in fusion.transforms[1:])
    assert np.mean(np.abs(fusion.rgb.astype(float) - base.astype(float))) < 3.0


def test_recover_burst_rejects_frame_count_before_decode() -> None:
    missing_sources = [f"missing-{index}.png" for index in range(13)]

    with pytest.raises(InvalidInputError, match="between 2 and 12"):
        recover_burst(missing_sources)


def test_recover_burst_rejects_aggregate_pixels_before_fusion() -> None:
    frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _index in range(3)]
    config = RecoverConfig(max_burst_total_pixels=1_000)

    with pytest.raises(InvalidInputError, match="total pixels"):
        recover_burst(frames, config=config)
