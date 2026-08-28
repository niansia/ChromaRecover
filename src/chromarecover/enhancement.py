"""Accessible mask presentations that do not alter the recovery decision."""

from __future__ import annotations

import cv2
import numpy as np


def make_structure_evidence(
    mask: np.ndarray, evidence_hint: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Create a continuous evidence field and presentation envelope.

    The support mask remains the auditable observation. The envelope is explicitly a
    derived presentation layer and is never fed back into candidate ranking.
    """

    height, width = mask.shape
    if evidence_hint is None:
        values = mask.astype(np.float32)
        minimum_side = min(height, width)
        fine = cv2.GaussianBlur(values, (0, 0), max(2.0, minimum_side * 0.022))
        broad = cv2.GaussianBlur(values, (0, 0), max(3.0, minimum_side * 0.042))
        evidence = 0.58 * fine + 0.42 * broad
    else:
        evidence = cv2.resize(
            np.asarray(evidence_hint, dtype=np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    low, high = (float(value) for value in np.percentile(evidence, [5.0, 99.0]))
    normalized = np.clip((evidence - low) / max(high - low, 1e-6), 0.0, 1.0).astype(
        np.float32, copy=False
    )
    evidence_u8 = np.round(normalized * 255.0).astype(np.uint8)
    otsu_threshold, _unused = cv2.threshold(
        evidence_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Slightly below Otsu joins sparse visual primitives into a readable envelope while
    # preserving gaps such as the counters in 0/8/A.
    threshold = max(26, round(float(otsu_threshold) * 0.72))
    envelope = evidence_u8 >= threshold
    kernel_size = max(3, round(min(height, width) * 0.012) | 1)
    envelope = cv2.morphologyEx(
        envelope.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
    ).astype(bool)
    return normalized.astype(np.float32), envelope


def make_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    structure_mask: np.ndarray | None = None,
) -> np.ndarray:
    base = np.asarray(rgb, dtype=np.uint8)
    selected = mask.astype(bool)
    color = np.array([0, 188, 212], dtype=np.uint16)  # cyan, plus a white outline for redundancy
    output = base.copy()
    if np.any(selected):
        selected_values = base[selected].astype(np.uint16)
        output[selected] = ((48 * selected_values + 52 * color + 50) // 100).astype(
            np.uint8
        )
    contours, _hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    if structure_mask is not None:
        structure_contours, _hierarchy = cv2.findContours(
            structure_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, structure_contours, -1, (255, 214, 10), 2, lineType=cv2.LINE_AA)
    return output
