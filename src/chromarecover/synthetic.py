"""Deterministic development fixtures with exact structure masks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SyntheticCase:
    image: np.ndarray
    mask: np.ndarray | None
    source_group_id: str
    support_mask: np.ndarray | None = None


def _hsv_color(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    hsv = np.array([[[hue % 180, saturation, value]]], dtype=np.uint8)
    return tuple(int(channel) for channel in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])


def _text_mask(size: int, text: str) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = size / 58.0
    thickness = max(5, round(size / 18))
    (width, height), _baseline = cv2.getTextSize(text, font, scale, thickness)
    if width > size * 0.86:
        scale *= size * 0.86 / width
        thickness = max(3, round(thickness * size * 0.86 / width))
        (width, height), _baseline = cv2.getTextSize(text, font, scale, thickness)
    origin = ((size - width) // 2, (size + height) // 2)
    cv2.putText(mask, text, origin, font, scale, 1, thickness, cv2.LINE_AA)
    return mask.astype(bool)


def make_chromatic_pattern(
    *,
    size: int = 256,
    text: str = "8",
    seed: int = 7,
    structured: bool = True,
    illumination: float = 0.22,
) -> SyntheticCase:
    """Create a dense dot mosaic where only relative chroma carries the target."""

    generator = np.random.default_rng(seed)
    target = _text_mask(size, text)
    background = np.array(_hsv_color(29, 168, 210), dtype=np.uint8)
    image = np.broadcast_to(background, (size, size, 3)).copy()
    truth = np.zeros((size, size), dtype=np.uint8)
    spacing = max(8, size // 25)

    for y in range(-spacing, size + spacing, spacing):
        for x in range(-spacing, size + spacing, spacing):
            center_x = int(x + generator.integers(-spacing // 3, spacing // 3 + 1))
            center_y = int(y + generator.integers(-spacing // 3, spacing // 3 + 1))
            radius = int(generator.integers(max(3, spacing // 2), max(4, spacing)))
            if structured:
                belongs = bool(
                    target[np.clip(center_y, 0, size - 1), np.clip(center_x, 0, size - 1)]
                )
            else:
                belongs = bool(generator.integers(0, 2))
            mean_hue = 13 if belongs else 31
            hue = mean_hue + generator.normal(0, 5.0)
            saturation = np.clip(generator.normal(175, 17), 105, 230)
            value = np.clip(generator.normal(207, 17), 145, 245)
            color = _hsv_color(hue, saturation, value)
            cv2.circle(image, (center_x, center_y), radius, color, -1, cv2.LINE_AA)
            if structured:
                # Keep the label synchronized with painter's-order occlusion.
                cv2.circle(
                    truth,
                    (center_x, center_y),
                    radius,
                    1 if belongs else 0,
                    -1,
                    cv2.LINE_AA,
                )

    if illumination:
        yy, xx = np.mgrid[0:size, 0:size]
        field = 1.0 + illumination * (
            0.55 * (xx / max(size - 1, 1) - 0.5) + 0.45 * (yy / max(size - 1, 1) - 0.5)
        )
        image = np.clip(image.astype(np.float32) * field[..., None], 0, 255).astype(np.uint8)

    return SyntheticCase(
        image=image,
        mask=truth.astype(bool) if structured else None,
        source_group_id=f"dots-{text}-{seed}",
    )


def degrade_camera(
    image: np.ndarray,
    *,
    seed: int = 19,
    blur_sigma: float = 0.65,
    perspective: float = 0.035,
    jpeg_quality: int = 78,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    height, width = image.shape[:2]
    dx = width * perspective
    dy = height * perspective
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    destination = np.float32(
        [[dx, dy], [width - 1 - dx, 0], [width - 1, height - 1 - dy], [0, height - 1]]
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(image, transform, (width, height), borderMode=cv2.BORDER_REFLECT)
    if blur_sigma > 0:
        warped = cv2.GaussianBlur(warped, (0, 0), blur_sigma)
    noise = generator.normal(0, 2.0, warped.shape)
    warped = np.clip(warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    success, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(warped, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )
    if not success:
        return warped
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def make_polygon_mosaic(
    *,
    size: int = 320,
    text: str = "820",
    seed: int = 41,
    structured: bool = True,
) -> SyntheticCase:
    """Create a matched-histogram mosaic encoded by local primitive populations."""

    generator = np.random.default_rng(seed)
    target = _text_mask(size, text)
    spacing = max(13, size // 18)
    points: list[tuple[float, float]] = []
    for y in range(spacing // 2, size, spacing):
        for x in range(spacing // 2, size, spacing):
            jitter = spacing * 0.34
            points.append(
                (
                    float(np.clip(x + generator.uniform(-jitter, jitter), 2, size - 3)),
                    float(np.clip(y + generator.uniform(-jitter, jitter), 2, size - 3)),
                )
            )

    subdivision = cv2.Subdiv2D((0, 0, size, size))
    for point in points:
        subdivision.insert(point)
    facets, centers = subdivision.getVoronoiFacetList([])
    centers = np.asarray(centers, dtype=np.float32)
    inside_target = np.array(
        [
            target[
                int(np.clip(round(center[1]), 0, size - 1)),
                int(np.clip(round(center[0]), 0, size - 1)),
            ]
            for center in centers
        ],
        dtype=bool,
    )
    special_count = max(1, round(len(facets) * 0.24))
    if structured:
        target_indices = np.flatnonzero(inside_target)
        if len(target_indices) > special_count:
            chosen_target = generator.choice(
                target_indices, size=special_count, replace=False
            )
            special_indices = {int(index) for index in chosen_target}
        else:
            outside_indices = np.flatnonzero(~inside_target)
            filler = generator.choice(
                outside_indices,
                size=special_count - len(target_indices),
                replace=False,
            )
            special_indices = {
                int(index) for index in np.concatenate([target_indices, filler])
            }
    else:
        weights = np.ones(len(facets), dtype=np.float64)
        weights /= weights.sum()
        special_indices = {
            int(index)
            for index in generator.choice(
                len(facets), size=special_count, replace=False, p=weights
            )
        }
    palette = np.array(
        [
            [113, 50, 46],
            [56, 61, 48],
            [187, 177, 74],
            [211, 196, 112],
            [151, 118, 74],
        ],
        dtype=np.int16,
    )
    image = np.full((size, size, 3), 232, dtype=np.uint8)
    support = np.zeros((size, size), dtype=np.uint8)
    for index, facet in enumerate(facets):
        polygon = np.round(np.asarray(facet)).astype(np.int32)
        if len(polygon) < 3:
            continue
        palette_index = 0 if index in special_indices else int(generator.integers(1, len(palette)))
        noise = generator.normal(0, 6, 3)
        color = tuple(int(value) for value in np.clip(palette[palette_index] + noise, 0, 255))
        cv2.fillConvexPoly(image, polygon, color, cv2.LINE_AA)
        cv2.polylines(image, [polygon], True, (232, 232, 224), 3, cv2.LINE_AA)
        if index in special_indices:
            cv2.fillConvexPoly(support, polygon, 1, cv2.LINE_AA)
            cv2.polylines(support, [polygon], True, 0, 3, cv2.LINE_AA)

    return SyntheticCase(
        image=image,
        mask=target if structured else None,
        source_group_id=f"polygon-matched-{text}-{seed}",
        support_mask=support.astype(bool) if structured else None,
    )
