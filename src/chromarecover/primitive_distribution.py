"""Image-native visual primitives and local color-population hypotheses.

Unlike the connected-component refinement in :mod:`primitives`, this branch begins from
the observed image. It is designed for mosaics where foreground and background reuse the
same palette but differ in the local population of colored tiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color import robust_standardize
from .config import RecoverConfig
from .hypotheses import MaskHypothesis


@dataclass(frozen=True)
class ImagePrimitive:
    label: int
    area: int
    center_x: float
    center_y: float
    mean_lab: tuple[float, float, float]
    fill: float
    elongation: float


@dataclass
class PrimitiveDistributionResult:
    hypotheses: list[MaskHypothesis]
    labels: np.ndarray
    primitive_count: int
    debug_rgb: np.ndarray


def _segment_tiles(lab: np.ndarray) -> np.ndarray:
    luminance = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    l_center = float(np.median(luminance))
    l_scale = max(float(np.median(np.abs(luminance - l_center))) * 1.4826, 1.0)
    c_center = float(np.median(chroma))
    c_scale = max(float(np.median(np.abs(chroma - c_center))) * 1.4826, 1.0)
    seam_score = (luminance - l_center) / l_scale - 0.38 * (chroma - c_center) / c_scale

    # Bright, low-chroma separators rank high. Eroding the complement prevents adjacent
    # tiles from touching through anti-aliased seam pixels.
    seam_threshold = float(np.percentile(seam_score, 69.0))
    tile_core = seam_score < seam_threshold
    tile_core = cv2.morphologyEx(
        tile_core.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    )
    return tile_core.astype(bool)


def _extract_image_primitives(
    lab: np.ndarray,
) -> tuple[np.ndarray, list[ImagePrimitive], np.ndarray]:
    tile_core = _segment_tiles(lab)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        tile_core.astype(np.uint8), connectivity=8
    )
    image_area = lab.shape[0] * lab.shape[1]
    minimum_area = max(5, round(image_area * 0.00010))
    maximum_area = image_area * 0.035
    primitives: list[ImagePrimitive] = []
    kept = np.zeros(count, dtype=bool)
    for label in range(1, count):
        _x, _y, width, height, area = (int(value) for value in stats[label])
        if not minimum_area <= area <= maximum_area:
            continue
        if min(width, height) < 2:
            continue
        values = lab[labels == label]
        mean_lab = tuple(float(value) for value in values.mean(axis=0))
        primitives.append(
            ImagePrimitive(
                label=label,
                area=area,
                center_x=float(centroids[label, 0]),
                center_y=float(centroids[label, 1]),
                mean_lab=mean_lab,
                fill=float(area / max(width * height, 1)),
                elongation=float(max(width, height) / max(min(width, height), 1)),
            )
        )
        kept[label] = True
    filtered_labels = labels.copy()
    filtered_labels[~kept[labels]] = 0
    return filtered_labels, primitives, tile_core


def _primitive_features(primitives: list[ImagePrimitive], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    raw = np.array(
        [
            [
                primitive.mean_lab[1],
                primitive.mean_lab[2],
                primitive.mean_lab[0],
                np.log1p(primitive.area),
                primitive.fill,
                min(primitive.elongation, 8.0),
                primitive.center_x / max(width, 1),
                primitive.center_y / max(height, 1),
            ]
            for primitive in primitives
        ],
        dtype=np.float32,
    )
    standardized = robust_standardize(raw.reshape(len(raw), 1, raw.shape[1])).reshape(raw.shape)
    weights = np.array([1.0, 1.0, 0.22, 0.24, 0.16, 0.12, 0.0, 0.0], dtype=np.float32)
    return standardized * weights


def _distribution_map(
    primitives: list[ImagePrimitive], membership: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, float]:
    height, width = shape
    all_impulses = np.zeros(shape, dtype=np.float32)
    member_impulses = np.zeros(shape, dtype=np.float32)
    for primitive, selected in zip(primitives, membership):
        x = int(np.clip(round(primitive.center_x), 0, width - 1))
        y = int(np.clip(round(primitive.center_y), 0, height - 1))
        weight = float(np.sqrt(primitive.area))
        all_impulses[y, x] += weight
        if selected:
            member_impulses[y, x] += weight

    scales = (max(3.0, min(shape) * 0.035), max(5.0, min(shape) * 0.065))
    proportions: list[np.ndarray] = []
    for sigma in scales:
        numerator = cv2.GaussianBlur(member_impulses, (0, 0), sigma)
        denominator = cv2.GaussianBlur(all_impulses, (0, 0), sigma)
        global_rate = float(membership.mean())
        proportions.append((numerator + 0.08 * global_rate) / (denominator + 0.08))
    proportion = 0.62 * proportions[0] + 0.38 * proportions[1]
    valid = cv2.GaussianBlur(all_impulses, (0, 0), scales[1]) > 1e-4
    baseline = float(np.median(proportion[valid])) if np.any(valid) else 0.0
    upper = float(np.percentile(proportion[valid], 95)) if np.any(valid) else baseline
    contrast = max(upper - baseline, 0.0)
    evidence = np.clip((proportion - baseline) / max(contrast, 1e-6), 0.0, 1.0)
    evidence[~valid] = 0.0
    return evidence.astype(np.float32), float(np.clip(contrast / 0.42, 0.0, 1.0))


def _label_preview(labels: np.ndarray, primitives: list[ImagePrimitive]) -> np.ndarray:
    preview = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for primitive in primitives:
        hue = (primitive.label * 47) % 180
        color = cv2.cvtColor(
            np.uint8([[[hue, 150, 230]]]), cv2.COLOR_HSV2RGB
        )[0, 0]
        preview[labels == primitive.label] = color
    return preview


def generate_primitive_distribution_hypotheses(
    lab: np.ndarray,
    config: RecoverConfig,
) -> PrimitiveDistributionResult:
    labels, primitives, _tile_core = _extract_image_primitives(lab)
    if not 18 <= len(primitives) <= 1400:
        return PrimitiveDistributionResult(
            hypotheses=[],
            labels=labels,
            primitive_count=len(primitives),
            debug_rgb=_label_preview(labels, primitives),
        )

    features = np.ascontiguousarray(_primitive_features(primitives, labels.shape), np.float32)
    hypotheses: list[MaskHypothesis] = []
    for cluster_count in (3, 4, 5, 6):
        if len(primitives) < cluster_count * 3:
            continue
        cv2.setRNGSeed(config.random_seed + 2000 + cluster_count)
        _compactness, assignments, _centers = cv2.kmeans(
            features,
            cluster_count,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 45, 1e-3),
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        assignments = assignments.ravel()
        for cluster in range(cluster_count):
            membership = assignments == cluster
            rate = float(membership.mean())
            if not 0.025 <= rate <= 0.65:
                continue
            selected_labels = [
                primitive.label
                for primitive, selected in zip(primitives, membership)
                if selected
            ]
            support = np.isin(labels, selected_labels)
            evidence, distribution_evidence = _distribution_map(
                primitives, membership, labels.shape
            )
            if distribution_evidence < 0.08:
                continue
            hypotheses.append(
                MaskHypothesis(
                    mask=support,
                    source=f"primitive_distribution:k{cluster_count}:c{cluster}",
                    family="primitive_distribution",
                    evidence_map=evidence,
                    metadata={
                        "distribution_evidence": distribution_evidence,
                        "primitive_count": float(membership.sum()),
                        "primitive_total": float(len(primitives)),
                    },
                )
            )

    return PrimitiveDistributionResult(
        hypotheses=hypotheses,
        labels=labels,
        primitive_count=len(primitives),
        debug_rgb=_label_preview(labels, primitives),
    )
