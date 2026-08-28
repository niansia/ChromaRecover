"""Color representations used by the relative chromatic residual ensemble."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    if values.max(initial=0.0) > 1.0:
        values = values / 255.0
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def linear_rgb_to_lab(linear: np.ndarray) -> np.ndarray:
    """Convert linear sRGB to CIELAB D65 without hidden byte-range conventions."""

    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = np.asarray(linear, dtype=np.float32) @ matrix.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6.0 / 29.0
    threshold = delta**3
    transformed = np.where(
        xyz > threshold,
        np.cbrt(xyz),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    lab = np.empty_like(transformed)
    lab[..., 0] = 116.0 * transformed[..., 1] - 16.0
    lab[..., 1] = 500.0 * (transformed[..., 0] - transformed[..., 1])
    lab[..., 2] = 200.0 * (transformed[..., 1] - transformed[..., 2])
    return lab.astype(np.float32)


def lab_to_lch(lab: np.ndarray) -> np.ndarray:
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.arctan2(lab[..., 2], lab[..., 1])
    return np.stack([lab[..., 0], chroma, hue], axis=-1).astype(np.float32)


def robust_standardize(features: np.ndarray) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1])
    median = np.median(flat, axis=0)
    mad = np.median(np.abs(flat - median), axis=0)
    scale = np.maximum(mad * 1.4826, 1e-4)
    return np.clip((features - median) / scale, -8.0, 8.0).astype(np.float32)


@dataclass
class ColorRepresentations:
    lab: np.ndarray
    lch: np.ndarray
    feature_sets: dict[str, np.ndarray]
    local_illumination: np.ndarray


def build_representations(
    rgb: np.ndarray,
    local_sigma_ratio: float,
    *,
    raw_rgb: np.ndarray | None = None,
) -> ColorRepresentations:
    linear = srgb_to_linear(rgb)
    lab = linear_rgb_to_lab(linear)
    lch = lab_to_lch(lab)
    height, width = lab.shape[:2]
    sigma = max(2.0, min(height, width) * local_sigma_ratio)

    low_frequency = np.stack(
        [cv2.GaussianBlur(lab[..., channel], (0, 0), sigma) for channel in range(3)],
        axis=-1,
    )
    residual = lab - low_frequency
    opponent = np.stack(
        [
            linear[..., 0] - linear[..., 1],
            0.5 * (linear[..., 0] + linear[..., 1]) - linear[..., 2],
        ],
        axis=-1,
    )

    global_chroma = robust_standardize(lab[..., 1:3])
    local_chroma = robust_standardize(residual[..., 1:3])
    opponent_z = robust_standardize(opponent)
    l_residual = robust_standardize(residual[..., :1])
    hybrid = np.concatenate(
        [global_chroma, local_chroma, opponent_z, 0.25 * l_residual], axis=-1
    ).astype(np.float32)

    feature_sets = {
        "global_lab": global_chroma,
        "local_lab": local_chroma,
        "opponent": opponent_z,
        "hybrid": hybrid,
    }
    # Camera normalization can remove nuisance illumination, but it may also attenuate a
    # legitimate weak signal. Keep an uncorrected chroma branch as an independent fallback.
    if raw_rgb is not None:
        raw_linear = srgb_to_linear(raw_rgb)
        raw_lab = linear_rgb_to_lab(raw_linear)
        raw_low_frequency = np.stack(
            [cv2.GaussianBlur(raw_lab[..., channel], (0, 0), sigma) for channel in range(3)],
            axis=-1,
        )
        raw_residual = raw_lab - raw_low_frequency
        feature_sets["raw_global_lab"] = robust_standardize(raw_lab[..., 1:3])
        raw_opponent = np.stack(
            [
                raw_linear[..., 0] - raw_linear[..., 1],
                0.5 * (raw_linear[..., 0] + raw_linear[..., 1]) - raw_linear[..., 2],
            ],
            axis=-1,
        )
        feature_sets["raw_opponent"] = robust_standardize(raw_opponent)
        raw_local = robust_standardize(raw_residual[..., 1:3])
        feature_sets["raw_local_lab"] = raw_local
        feature_sets["raw_hybrid"] = np.concatenate(
            [
                feature_sets["raw_global_lab"],
                raw_local,
                feature_sets["raw_opponent"],
                0.25 * robust_standardize(raw_residual[..., :1]),
            ],
            axis=-1,
        ).astype(np.float32)

    return ColorRepresentations(
        lab=lab,
        lch=lch,
        feature_sets=feature_sets,
        local_illumination=low_frequency[..., 0],
    )
