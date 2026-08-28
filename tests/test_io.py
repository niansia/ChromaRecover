from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from chromarecover.exceptions import InvalidInputError, UnsupportedFormatError
from chromarecover.io import decode_image


def _decode(path: Path):
    return decode_image(
        path,
        max_pixels=1_000_000,
        max_file_bytes=10_000_000,
        max_icc_profile_bytes=4 * 1024 * 1024,
        analysis_max_side=384,
    )


def test_decoded_format_not_suffix_controls_acceptance(tmp_path: Path) -> None:
    pixels = np.full((20, 30, 3), [20, 90, 180], dtype=np.uint8)
    renamed_png = tmp_path / "valid-image.txt"
    Image.fromarray(pixels).save(renamed_png, format="PNG")

    decoded = _decode(renamed_png)

    assert decoded.input_info["format"] == "PNG"


def test_mismatched_jpeg_suffix_reports_decoded_format(tmp_path: Path) -> None:
    pixels = np.full((20, 30, 3), [20, 90, 180], dtype=np.uint8)
    jpeg_named_png = tmp_path / "photo.png"
    Image.fromarray(pixels).save(jpeg_named_png, format="JPEG")

    assert _decode(jpeg_named_png).input_info["format"] == "JPEG"


def test_unsupported_content_is_rejected_even_with_png_suffix(tmp_path: Path) -> None:
    pixels = np.full((20, 30, 3), [20, 90, 180], dtype=np.uint8)
    bmp_named_png = tmp_path / "unsupported.png"
    Image.fromarray(pixels).save(bmp_named_png, format="BMP")

    with pytest.raises(UnsupportedFormatError):
        _decode(bmp_named_png)


def test_embedded_icc_profile_is_converted_to_srgb(tmp_path: Path) -> None:
    pixels = np.full((20, 30, 3), [20, 90, 180], dtype=np.uint8)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    source = tmp_path / "profiled.png"
    Image.fromarray(pixels).save(source, icc_profile=profile)

    decoded = _decode(source)

    assert decoded.input_info["icc_profile"] == "converted_to_srgb"
    assert "icc_to_srgb" in decoded.preprocess


def test_file_size_limit_is_checked_before_decode(tmp_path: Path) -> None:
    source = tmp_path / "oversized.png"
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(source)

    with pytest.raises(InvalidInputError, match="allowed maximum"):
        decode_image(
            source,
            max_pixels=1_000_000,
            max_file_bytes=source.stat().st_size - 1,
            max_icc_profile_bytes=4 * 1024 * 1024,
            analysis_max_side=384,
        )


def test_oversized_icc_profile_is_ignored_before_littlecms() -> None:
    image = Image.fromarray(np.full((20, 30, 3), [20, 90, 180], dtype=np.uint8))
    image.info["icc_profile"] = b"not-a-real-profile" * 8

    decoded = decode_image(
        image,
        max_pixels=1_000_000,
        max_file_bytes=10_000_000,
        max_icc_profile_bytes=32,
        analysis_max_side=384,
    )

    assert decoded.input_info["icc_profile"] == "oversized_ignored"
    assert "icc_oversized_ignored" in decoded.preprocess
