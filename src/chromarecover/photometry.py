"""Camera photometric normalization and invalid-region estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .color import linear_to_srgb, srgb_to_linear


@dataclass
class PhotometricResult:
    rgb: np.ndarray
    invalid_mask: np.ndarray
    glare_score: float
    shadow_score: float
    fold_score: float
    moire_score: float
    banding_score: float = 0.0
    capture_profile: str = "camera-paper"
    steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


def estimate_capture_route(rgb: np.ndarray) -> dict[str, float | str]:
    """Conservatively route only strong display-capture signatures.

    Periodic banding alone also occurs in synthetic/digital mosaics, so automatic screen
    routing requires strong chromatic moire or joint moire+banding support. Paper routing
    remains explicit until it has source-group-held-out camera fixtures.
    """

    source = np.asarray(rgb, dtype=np.uint8)
    lab = cv2.cvtColor(source.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    moire = _moire_score(source)
    banding = _banding_score(lab[..., 0])
    screen = moire >= 0.55 or (moire >= 0.35 and banding >= 0.28)
    return {
        "resolved_mode": "camera-screen" if screen else "digital",
        "moire": float(moire),
        "banding": float(banding),
        "rule": "strong_screen_signature_v1" if screen else "conservative_digital_fallback",
    }


def _moire_score(rgb: np.ndarray) -> float:
    height, width = rgb.shape[:2]
    scale = min(1.0, 256.0 / max(height, width))
    preview = (
        cv2.resize(
            rgb,
            (max(32, round(width * scale)), max(32, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else rgb
    )
    preview_float = preview.astype(np.float32) / 255.0
    lab = cv2.cvtColor(preview_float, cv2.COLOR_RGB2LAB)
    channels = [lab[..., 0] / 100.0, lab[..., 1] / 128.0, lab[..., 2] / 128.0]
    yy, xx = np.mgrid[: lab.shape[0], : lab.shape[1]]
    scores: list[float] = []
    for channel in channels:
        residual = channel - cv2.GaussianBlur(channel, (0, 0), 2.5)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual)))
        center_y = (spectrum.shape[0] - 1) / 2.0
        center_x = (spectrum.shape[1] - 1) / 2.0
        radius = np.hypot(yy - center_y, xx - center_x)
        spectrum[radius < 0.07 * min(spectrum.shape)] = 0.0
        positive = spectrum[spectrum > 0]
        if positive.size < 16:
            continue
        peak = float(np.percentile(positive, 99.95))
        baseline = float(np.percentile(positive, 90.0))
        ratio = peak / max(baseline, 1e-6)
        spectral_score = float(
            np.clip((np.log1p(ratio) - np.log1p(5.0)) / np.log(5.0), 0.0, 1.0)
        )
        # A single sharp crease also creates a strong Fourier peak but is spatially sparse.
        # Display moire is repeated across a meaningful fraction of the frame.
        coverage = float(np.mean(np.abs(residual) > 0.01))
        scores.append(spectral_score * float(np.clip(coverage / 0.20, 0.0, 1.0)))
    return max(scores, default=0.0)


def _fold_score(luminance: np.ndarray) -> float:
    """Estimate elongated crease-like curvature without claiming a physical model."""

    sigma = max(2.0, min(luminance.shape) * 0.012)
    mid_scale = cv2.GaussianBlur(luminance, (0, 0), sigma)
    curvature = np.abs(cv2.Laplacian(mid_scale, cv2.CV_32F))
    threshold = max(0.10, float(np.percentile(curvature, 97.5)))
    ridges = (curvature >= threshold).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(ridges, connectivity=8)
    best = 0.0
    for label in range(1, count):
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        elongation = max(width, height) / max(min(width, height), 1)
        span = max(width / luminance.shape[1], height / luminance.shape[0])
        best = max(best, np.clip((elongation - 3.0) / 12.0, 0.0, 1.0) * span)
    strength = float(np.clip(np.percentile(curvature, 99.5) / 0.55, 0.0, 1.0))
    return float(np.clip(best * strength, 0.0, 1.0))


def _banding_score(luminance: np.ndarray) -> float:
    row_signal = np.median(luminance, axis=1)
    column_signal = np.median(luminance, axis=0)

    def periodic_energy(signal: np.ndarray) -> float:
        signal = signal - cv2.GaussianBlur(signal.reshape(-1, 1), (0, 0), 9.0).ravel()
        spectrum = np.abs(np.fft.rfft(signal))
        if len(spectrum) < 8:
            return 0.0
        spectrum[:2] = 0
        peak = float(np.percentile(spectrum, 98))
        baseline = float(np.percentile(spectrum, 65))
        return float(np.clip((peak / max(baseline, 1e-6) - 2.5) / 8.0, 0.0, 1.0))

    return max(periodic_energy(row_signal), periodic_energy(column_signal))


def normalize_camera_photometry(
    rgb: np.ndarray, *, profile: str = "auto"
) -> PhotometricResult:
    if profile not in {"auto", "paper", "screen"}:
        raise ValueError("profile must be one of: auto, paper, screen")
    source = np.asarray(rgb, dtype=np.uint8)
    rgb_float = source.astype(np.float32) / 255.0
    lab_source = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)
    chroma = np.hypot(lab_source[..., 1], lab_source[..., 2])
    maximum = rgb_float.max(axis=2)
    glare_mask = (lab_source[..., 0] > 93.0) & (chroma < 12.0) & (maximum > 0.96)
    glare_mask = cv2.dilate(glare_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    glare_score = float(glare_mask.mean())

    valid = ~glare_mask
    linear = srgb_to_linear(rgb_float)
    channel_means = np.array(
        [np.mean(linear[..., channel][valid]) if np.any(valid) else 1.0 for channel in range(3)],
        dtype=np.float32,
    )
    target = float(np.exp(np.mean(np.log(np.maximum(channel_means, 1e-5)))))
    gains = np.clip(target / np.maximum(channel_means, 1e-5), 0.72, 1.38)
    balanced = linear_to_srgb(np.clip(linear * gains, 0.0, 1.0))

    lab = cv2.cvtColor(balanced.astype(np.float32), cv2.COLOR_RGB2LAB)
    sigma = max(6.0, min(source.shape[:2]) * 0.09)
    illumination = cv2.GaussianBlur(lab[..., 0], (0, 0), sigma)
    illumination_center = float(np.median(illumination[valid])) if np.any(valid) else 50.0
    span = float(np.percentile(illumination, 95) - np.percentile(illumination, 5))
    shadow_score = float(np.clip(span / 35.0, 0.0, 1.0))
    fold_score = _fold_score(lab[..., 0])
    lab[..., 0] = np.clip(lab[..., 0] - illumination + illumination_center, 0.0, 100.0)

    moire_score = _moire_score(source)
    banding_score = _banding_score(lab[..., 0])
    resolved_profile = "screen" if profile == "auto" and max(moire_score, banding_score) > 0.38 else profile
    if resolved_profile == "auto":
        resolved_profile = "paper"
    steps = ["camera_gray_world", "camera_local_illumination"]
    moire_threshold = 0.38 if resolved_profile == "screen" else 0.68
    if moire_score > moire_threshold:
        chroma_sigma = 1.05 if resolved_profile == "screen" else 0.72
        lab[..., 1] = cv2.GaussianBlur(lab[..., 1], (0, 0), chroma_sigma)
        lab[..., 2] = cv2.GaussianBlur(lab[..., 2], (0, 0), chroma_sigma)
        lab[..., 0] = cv2.bilateralFilter(lab[..., 0].astype(np.float32), 5, 5.0, 2.0)
        steps.append("camera_moire_chroma_suppression")
    if resolved_profile == "screen" and banding_score > 0.35:
        row_bias = np.median(lab[..., 0], axis=1)
        row_bias -= cv2.GaussianBlur(row_bias.reshape(-1, 1), (0, 0), 11.0).ravel()
        lab[..., 0] = np.clip(lab[..., 0] - row_bias[:, None], 0.0, 100.0)
        steps.append("camera_refresh_banding_suppression")

    corrected = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    warnings: list[str] = []
    if glare_score > 0.03:
        warnings.append("possible_glare")
    if shadow_score > 0.55:
        warnings.append("strong_shadow")
    if resolved_profile == "paper" and fold_score > 0.55:
        warnings.append("possible_fold_or_wrinkle")
    if moire_score > moire_threshold:
        warnings.append("possible_moire")
    if resolved_profile == "screen" and banding_score > 0.35:
        warnings.append("possible_refresh_banding")

    return PhotometricResult(
        rgb=corrected,
        invalid_mask=glare_mask,
        glare_score=glare_score,
        shadow_score=shadow_score,
        fold_score=fold_score,
        moire_score=moire_score,
        banding_score=banding_score,
        capture_profile=f"camera-{resolved_profile}",
        steps=steps,
        warnings=warnings,
        metadata={
            "white_balance_gains": gains.tolist(),
            "illumination_span": span,
            "invalid_fraction": glare_score,
            "capture_profile": f"camera-{resolved_profile}",
            "banding_score": banding_score,
        },
    )
