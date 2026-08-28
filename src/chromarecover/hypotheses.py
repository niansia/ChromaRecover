"""Multi-family foreground hypothesis generation without a fixed foreground color."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import RecoverConfig


@dataclass
class MaskHypothesis:
    mask: np.ndarray
    source: str
    family: str
    evidence_map: np.ndarray | None = None
    metadata: dict[str, float] = field(default_factory=dict)


def _sample_rows(values: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(values) <= limit:
        return values
    generator = np.random.default_rng(seed)
    return values[generator.choice(len(values), size=limit, replace=False)]


def _kmeans_masks(
    features: np.ndarray,
    family: str,
    cluster_counts: tuple[int, ...],
    sample_limit: int,
    attempts: int,
    seed: int,
) -> list[MaskHypothesis]:
    height, width, channels = features.shape
    flat = np.ascontiguousarray(features.reshape(-1, channels), dtype=np.float32)
    sample = np.ascontiguousarray(_sample_rows(flat, sample_limit, seed), dtype=np.float32)
    generated: list[MaskHypothesis] = []

    for cluster_count in cluster_counts:
        if len(sample) < cluster_count:
            continue
        cv2.setRNGSeed(seed + cluster_count)
        _compactness, _labels, centers = cv2.kmeans(
            sample,
            cluster_count,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 1e-3),
            attempts,
            cv2.KMEANS_PP_CENTERS,
        )
        distances = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1).reshape(height, width)
        for cluster in range(cluster_count):
            mask = labels == cluster
            generated.append(
                MaskHypothesis(mask=mask, source=f"{family}:k{cluster_count}:c{cluster}", family=family)
            )
            if cluster_count > 2:
                generated.append(
                    MaskHypothesis(
                        mask=~mask,
                        source=f"{family}:k{cluster_count}:not-c{cluster}",
                        family=family,
                    )
                )
    return generated


def _projection_masks(
    features: np.ndarray,
    family: str,
    quantiles: tuple[float, ...],
    sample_limit: int,
    seed: int,
) -> list[MaskHypothesis]:
    height, width, channels = features.shape
    flat = features.reshape(-1, channels).astype(np.float32)
    sample = _sample_rows(flat, sample_limit, seed)
    centered = sample - np.median(sample, axis=0)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, np.argsort(eigenvalues)[::-1][: min(2, channels)]]
    full_centered = flat - np.median(sample, axis=0)
    generated: list[MaskHypothesis] = []

    for axis_index in range(axes.shape[1]):
        projected = (full_centered @ axes[:, axis_index]).reshape(height, width)
        for quantile in quantiles:
            threshold = float(np.quantile(projected, quantile))
            if quantile < 0.5:
                mask = projected <= threshold
                side = "low"
            else:
                mask = projected >= threshold
                side = "high"
            generated.append(
                MaskHypothesis(
                    mask=mask,
                    source=f"{family}:pca{axis_index}:{side}:q{quantile:.2f}",
                    family=f"{family}_projection",
                )
            )
    return generated


def _deduplicate(
    hypotheses: list[MaskHypothesis], config: RecoverConfig
) -> list[MaskHypothesis]:
    eligible: list[MaskHypothesis] = []
    previews: list[np.ndarray] = []
    for hypothesis in hypotheses:
        occupancy = float(hypothesis.mask.mean())
        if not config.min_occupancy <= occupancy <= config.max_occupancy:
            continue
        preview = cv2.resize(
            hypothesis.mask.astype(np.uint8), (48, 48), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        eligible.append(hypothesis)
        previews.append(preview)
    if not eligible:
        return []

    # Compute all compact-preview IoUs once. The short loop below preserves deterministic
    # first-wins behavior without hundreds of thousands of Python-level IoU calls.
    matrix = np.stack(previews).reshape(len(previews), -1).astype(np.float32)
    intersections = matrix @ matrix.T
    areas = matrix.sum(axis=1)
    unions = areas[:, None] + areas[None, :] - intersections
    pairwise_iou = np.divide(
        intersections,
        unions,
        out=np.ones_like(intersections),
        where=unions > 0,
    )
    keep = np.zeros(len(eligible), dtype=bool)
    for index in range(len(eligible)):
        if not np.any(pairwise_iou[index, :index][keep[:index]] >= config.duplicate_iou):
            keep[index] = True
    return [hypothesis for hypothesis, retain in zip(eligible, keep) if retain]


def generate_hypotheses(
    feature_sets: dict[str, np.ndarray], config: RecoverConfig
) -> list[MaskHypothesis]:
    generated: list[MaskHypothesis] = []
    for index, (family, features) in enumerate(feature_sets.items()):
        generated.extend(
            _kmeans_masks(
                features,
                family,
                config.cluster_counts,
                config.max_kmeans_samples,
                config.kmeans_attempts,
                config.random_seed + index * 101,
            )
        )
        if family != "hybrid":
            generated.extend(
                _projection_masks(
                    features,
                    family,
                    config.projection_quantiles,
                    config.max_kmeans_samples,
                    config.random_seed + index * 101,
                )
            )
    deduplicated = _deduplicate(generated, config)
    # Imported lazily to keep the primitive module independently testable without a
    # module-level cycle (its public types are MaskHypothesis objects).
    from .primitives import augment_with_primitive_graph

    return _deduplicate(augment_with_primitive_graph(deduplicated), config)
