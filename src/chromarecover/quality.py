"""Conservative image quality signals; these never invent a structure."""

from __future__ import annotations

import cv2
import numpy as np

from .color import ColorRepresentations
from .types import QualityReport


def _contrast_normalized_blur(rgb: np.ndarray) -> tuple[float, float, bool]:
    """Estimate blur without confusing low luminance contrast with defocus."""

    scale = min(1.0, 1024.0 / max(rgb.shape[:2]))
    source = (
        cv2.resize(
            rgb,
            (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else rgb
    )
    lab = cv2.cvtColor(source.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    sharpness_values: list[float] = []
    edge_support: list[float] = []
    for channel_index, minimum_span in enumerate((1.5, 0.8, 0.8)):
        channel = lab[..., channel_index]
        # NumPy 2 follows NEP 50 scalar promotion: subtracting an np.float64 percentile
        # from this float32 channel promotes the full image to float64. OpenCV then rejects
        # a CV_32F Laplacian destination for that CV_64F source.
        low, high = (float(value) for value in np.percentile(channel, [1.0, 99.0]))
        span = float(high - low)
        if span < minimum_span:
            continue
        normalized = (np.clip((channel - low) / span, 0.0, 1.0) * 255.0).astype(
            np.float32, copy=False
        )
        normalized = cv2.GaussianBlur(normalized, (0, 0), 1.0)
        laplacian_variance = float(cv2.Laplacian(normalized, cv2.CV_32F).var())
        horizontal = cv2.Sobel(normalized, cv2.CV_32F, 1, 0)
        vertical = cv2.Sobel(normalized, cv2.CV_32F, 0, 1)
        sharpness_values.append(laplacian_variance)
        edge_support.append(float(np.mean(np.hypot(horizontal, vertical) > 8.0)))

    if not sharpness_values:
        return 0.35, 0.0, False
    sharpness = max(sharpness_values)
    # Logistic calibration: normalized Laplacian variance around six is the transition
    # between a spread edge and a sharp edge after the fixed one-pixel prefilter.
    blur = 1.0 / (1.0 + np.exp(2.0 * (np.log(max(sharpness, 1e-6)) - np.log(6.0))))
    return float(np.clip(blur, 0.0, 1.0)), max(edge_support), True


def assess_quality(
    rgb: np.ndarray,
    colors: ColorRepresentations,
    *,
    blur_rgb: np.ndarray | None = None,
) -> QualityReport:
    luminance = np.clip(colors.lab[..., 0] / 100.0, 0.0, 1.0)
    blur_source = blur_rgb if blur_rgb is not None else rgb
    blur, texture_support, blur_is_observable = _contrast_normalized_blur(blur_source)
    clipping = float(np.mean((luminance <= 0.015) | (luminance >= 0.985)))
    chroma = colors.lch[..., 1]
    glare = float(np.mean((luminance > 0.92) & (chroma < 8.0)))
    # Joint a/b displacement detects hue-only differences and sparse foregrounds. A MAD of
    # chroma magnitude alone incorrectly reports zero when foreground and background have
    # equal chroma or when the foreground occupies less than half the image.
    ab = colors.lab[..., 1:3]
    ab_median = np.median(ab.reshape(-1, 2), axis=0)
    ab_displacement = np.linalg.norm(ab - ab_median, axis=-1)
    color_spread = float(np.clip(np.percentile(ab_displacement, 98) / 24.0, 0, 1))

    warnings: list[str] = []
    if not blur_is_observable:
        warnings.append("low_texture_blur_indeterminate")
    elif blur > 0.68:
        warnings.append("severe_blur")
    elif blur > 0.50:
        warnings.append("mild_blur")
    if clipping > 0.18:
        warnings.append("exposure_clipping")
    if glare > 0.08:
        warnings.append("possible_glare")
    if color_spread < 0.025:
        warnings.append("insufficient_chromatic_variation")

    return QualityReport(
        blur=blur,
        clipping=clipping,
        glare=glare,
        color_spread=color_spread,
        texture_support=texture_support,
        warnings=warnings,
    )
