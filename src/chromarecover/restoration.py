"""Conservative single-frame restoration hypotheses.

Restored pixels are never emitted as ground truth. They only create additional color
representations; the original observation and quality warning remain authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RestorationHypothesis:
    rgb: np.ndarray
    psf_sigma: float
    regularization: float


def _gaussian_psf(shape: tuple[int, int], sigma: float) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    yy = np.minimum(yy, height - yy)
    xx = np.minimum(xx, width - xx)
    psf = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return psf / max(float(psf.sum()), 1e-8)


def _wiener_channel(channel: np.ndarray, sigma: float, regularization: float) -> np.ndarray:
    psf = _gaussian_psf(channel.shape, sigma)
    transfer = np.fft.fft2(psf)
    observed = np.fft.fft2(channel)
    restored = np.fft.ifft2(
        np.conj(transfer) * observed / (np.square(np.abs(transfer)) + regularization)
    ).real
    return restored.astype(np.float32)


def wiener_restoration_hypotheses(rgb: np.ndarray) -> list[RestorationHypothesis]:
    source = rgb.astype(np.float32) / 255.0
    hypotheses: list[RestorationHypothesis] = []
    for sigma, regularization in ((0.8, 0.020), (1.15, 0.035), (1.65, 0.060)):
        restored = np.stack(
            [
                _wiener_channel(source[..., channel], sigma, regularization)
                for channel in range(3)
            ],
            axis=-1,
        )
        out_of_range = float(np.mean((restored < -0.08) | (restored > 1.08)))
        if out_of_range > 0.025:
            continue
        restored = np.clip(restored, 0.0, 1.0)
        # A weak bilateral pass limits ringing without erasing recovered chroma edges.
        restored_u8 = np.round(restored * 255).astype(np.uint8)
        restored_u8 = cv2.bilateralFilter(restored_u8, 3, 4.0, 1.5)
        hypotheses.append(
            RestorationHypothesis(
                rgb=restored_u8,
                psf_sigma=sigma,
                regularization=regularization,
            )
        )
    return hypotheses
