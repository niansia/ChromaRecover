"""Safe image decoding and deterministic analysis resizing."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from .exceptions import InvalidInputError, UnsupportedFormatError


@dataclass
class DecodedImage:
    rgb: np.ndarray
    analysis_rgb: np.ndarray
    input_info: dict[str, Any]
    preprocess: list[str]


def _to_rgb(image: Image.Image, max_icc_profile_bytes: int) -> tuple[Image.Image, str]:
    image = ImageOps.exif_transpose(image)
    profile_data = image.info.get("icc_profile")
    if profile_data:
        if not isinstance(profile_data, (bytes, bytearray, memoryview)):
            icc_status = "invalid_ignored"
        elif len(profile_data) > max_icc_profile_bytes:
            icc_status = "oversized_ignored"
        else:
            try:
                source_profile = ImageCms.ImageCmsProfile(BytesIO(profile_data))
                destination_profile = ImageCms.createProfile("sRGB")
                image = ImageCms.profileToProfile(
                    image, source_profile, destination_profile, outputMode="RGB"
                )
                return image, "converted_to_srgb"
            except (OSError, ValueError, TypeError):
                icc_status = "invalid_ignored"
    else:
        icc_status = "none"

    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    return image, icc_status


def _validate_dimensions(width: int, height: int, max_pixels: int) -> None:
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise InvalidInputError(
            f"Image has {width * height:,} pixels; allowed maximum is {max_pixels:,}"
        )


def _validate_file_source(path: Path, max_file_bytes: int) -> int:
    try:
        if not path.is_file():
            raise InvalidInputError(f"Image does not exist: {path}")
        file_bytes = path.stat().st_size
    except OSError as exc:
        raise InvalidInputError(f"Could not inspect image: {exc}") from exc
    if file_bytes > max_file_bytes:
        raise InvalidInputError(
            f"Image file has {file_bytes:,} bytes; allowed maximum is {max_file_bytes:,}"
        )
    return file_bytes


def _validate_decoded_format(decoded_format: str) -> None:
    if decoded_format not in {"PNG", "JPEG", "WEBP"}:
        raise UnsupportedFormatError(f"Unsupported decoded image format: {decoded_format}")


def probe_image_size(
    source: str | Path | np.ndarray | Image.Image,
    *,
    max_pixels: int,
    max_file_bytes: int,
) -> tuple[int, int]:
    """Read only enough metadata to enforce burst budgets before full decoding."""

    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            _validate_file_source(path, max_file_bytes)
            with Image.open(path) as opened:
                _validate_decoded_format((opened.format or "unknown").upper())
                width, height = opened.size
        elif isinstance(source, Image.Image):
            width, height = source.size
        elif isinstance(source, np.ndarray):
            array = np.asarray(source)
            if array.ndim != 3 or array.shape[2] not in {3, 4}:
                raise InvalidInputError("NumPy input must have shape HxWx3 or HxWx4")
            height, width = array.shape[:2]
        else:
            raise InvalidInputError(f"Unsupported input type: {type(source).__name__}")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidInputError(f"Could not inspect image: {exc}") from exc

    _validate_dimensions(width, height, max_pixels)
    return width, height


def decode_image(
    source: str | Path | np.ndarray | Image.Image,
    *,
    max_pixels: int,
    max_file_bytes: int,
    max_icc_profile_bytes: int,
    analysis_max_side: int,
) -> DecodedImage:
    preprocess: list[str] = ["exif_orientation", "convert_rgb"]
    source_name = "array"
    decoded_format = "array"
    icc_status = "none"
    file_bytes: int | None = None

    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            source_name = path.name
            file_bytes = _validate_file_source(path, max_file_bytes)
            with Image.open(path) as opened:
                decoded_format = (opened.format or "unknown").upper()
                _validate_decoded_format(decoded_format)
                _validate_dimensions(*opened.size, max_pixels)
                image, icc_status = _to_rgb(opened, max_icc_profile_bytes)
        elif isinstance(source, Image.Image):
            source_name = "PIL.Image"
            decoded_format = (source.format or "PIL").upper()
            _validate_dimensions(*source.size, max_pixels)
            image, icc_status = _to_rgb(source, max_icc_profile_bytes)
        elif isinstance(source, np.ndarray):
            array = np.asarray(source)
            if array.ndim != 3 or array.shape[2] not in {3, 4}:
                raise InvalidInputError("NumPy input must have shape HxWx3 or HxWx4")
            _validate_dimensions(array.shape[1], array.shape[0], max_pixels)
            if array.dtype != np.uint8:
                if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
                    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
                else:
                    array = np.clip(array, 0, 255).astype(np.uint8)
            image, icc_status = _to_rgb(Image.fromarray(array), max_icc_profile_bytes)
        else:
            raise InvalidInputError(f"Unsupported input type: {type(source).__name__}")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidInputError(f"Could not decode image: {exc}") from exc

    width, height = image.size
    _validate_dimensions(width, height, max_pixels)

    rgb = np.asarray(image, dtype=np.uint8)
    if icc_status == "converted_to_srgb":
        preprocess.append("icc_to_srgb")
    elif icc_status == "invalid_ignored":
        preprocess.append("icc_invalid_ignored")
    elif icc_status == "oversized_ignored":
        preprocess.append("icc_oversized_ignored")
    scale = min(1.0, analysis_max_side / max(width, height))
    if scale < 1.0:
        analysis = cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        preprocess.append(f"analysis_resize:{analysis.shape[1]}x{analysis.shape[0]}")
    else:
        analysis = rgb.copy()

    return DecodedImage(
        rgb=rgb,
        analysis_rgb=analysis,
        input_info={
            "source": source_name,
            "format": decoded_format,
            "icc_profile": icc_status,
            "width": width,
            "height": height,
            "analysis_width": int(analysis.shape[1]),
            "analysis_height": int(analysis.shape[0]),
            "file_bytes": file_bytes,
        },
        preprocess=preprocess,
    )
