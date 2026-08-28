"""Mask scoring for the Chromatic Residual Ensemble (CRE)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color import ColorRepresentations
from .hypotheses import MaskHypothesis
from .types import QualityReport


@dataclass
class ScoredMask:
    mask: np.ndarray
    source: str
    family: str
    score: float
    confidence: float
    metrics: dict[str, float]
    evidence_map: np.ndarray | None = None


def _separation(mask: np.ndarray, features: np.ndarray) -> float:
    flat = features.reshape(-1, features.shape[-1])
    selected = mask.reshape(-1)
    inside = flat[selected]
    outside = flat[~selected]
    if len(inside) < 8 or len(outside) < 8:
        return 0.0
    difference = np.linalg.norm(np.mean(inside, axis=0) - np.mean(outside, axis=0))
    pooled = np.sqrt(0.5 * (np.var(inside, axis=0).mean() + np.var(outside, axis=0).mean()))
    effect = difference / max(float(pooled), 1e-4)
    return float(np.tanh(effect / 3.0))


def _spatial_structure(mask: np.ndarray) -> tuple[float, float]:
    values = mask.astype(np.float32)
    occupancy = float(values.mean())
    noise_scale = np.sqrt(max(occupancy * (1.0 - occupancy), 1e-6))
    minimum_side = min(mask.shape)
    signals: list[float] = []
    component_scores: list[float] = []

    # Fine-to-coarse scales retain narrow multi-character strokes while still requiring
    # density modulation beyond individual primitives.
    for ratio in (0.02, 0.04, 0.08):
        sigma = max(2.0, minimum_side * ratio)
        density = cv2.GaussianBlur(values, (0, 0), sigma)
        normalized_energy = float(density.std() / noise_scale)
        signals.append(normalized_energy)

        density_peak = float(np.percentile(density, 99.5))
        relative_threshold = occupancy + 0.25 * max(density_peak - occupancy, 0.0)
        coarse = (density >= relative_threshold).astype(np.uint8)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            coarse, connectivity=8
        )
        areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0)
        coarse_area = int(areas.sum())
        if coarse_area:
            # A word or number row may legitimately contain several disconnected glyphs.
            # Reward the mass explained by up to six dominant components instead of only
            # the single largest component, which structurally favored large masks.
            dominant = np.sort(areas)[-6:]
            component_scores.append(float(dominant.sum() / coarse_area))
        else:
            component_scores.append(0.0)

    spatial = float(np.clip((np.mean(signals) - 0.12) / 0.34, 0, 1))
    return spatial, float(np.mean(component_scores))


def _frame_focus(mask: np.ndarray) -> float:
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return float(np.clip(1.0 - border.mean(), 0.0, 1.0))


def _planar_explanation(mask: np.ndarray) -> float:
    """Return R² for a low-order spatial field explaining the candidate mask.

    A half-frame threshold caused by illumination or a smooth color cast is highly
    predictable from position alone. This is weak negative evidence, not a general ban on
    large or simple foregrounds.
    """

    preview = cv2.resize(mask.astype(np.float32), (48, 48), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[-1.0:1.0:48j, -1.0:1.0:48j]
    design = np.stack(
        [
            np.ones(preview.size),
            xx.ravel(),
            yy.ravel(),
            np.square(xx).ravel(),
            np.square(yy).ravel(),
            (xx * yy).ravel(),
        ],
        axis=1,
    )
    observed = preview.ravel()
    coefficients, *_rest = np.linalg.lstsq(design, observed, rcond=None)
    predicted = design @ coefficients
    residual = float(np.square(observed - predicted).sum())
    total = float(np.square(observed - observed.mean()).sum())
    return float(np.clip(1.0 - residual / max(total, 1e-8), 0.0, 1.0))


def _boundary_evidence(mask: np.ndarray, chroma_gradient: np.ndarray, reference: float) -> float:
    boundary = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)
    if not np.any(boundary):
        return 0.0
    ratio = float(np.median(chroma_gradient[boundary]) / max(reference, 1e-6))
    return float(np.clip((ratio - 0.45) / 0.95, 0.0, 1.0))


def _quality_factor(quality: QualityReport) -> float:
    factor = (
        1.0
        - 0.24 * quality.blur
        - 0.55 * quality.clipping
        - 0.45 * quality.glare
        - 0.18 * quality.shadow
        - 0.10 * quality.fold
        - 0.12 * quality.moire
        - 0.08 * quality.banding
    )
    if quality.color_spread < 0.025:
        factor *= 0.45
    if "low_effective_resolution" in quality.warnings:
        factor *= 0.55
    return float(np.clip(factor, 0.15, 1.0))


def score_hypotheses(
    hypotheses: list[MaskHypothesis],
    colors: ColorRepresentations,
    quality: QualityReport,
    invalid_mask: np.ndarray | None = None,
) -> list[ScoredMask]:
    if not hypotheses:
        return []
    preliminary: list[dict[str, object]] = []
    source_height, source_width = hypotheses[0].mask.shape if hypotheses else colors.lab.shape[:2]
    # Ranking operates on a fixed physical scale; 144 px retains thin glyph strokes while
    # keeping hundreds of multi-family proposals practical on CPU.
    evaluation_scale = min(1.0, 144.0 / max(source_height, source_width))
    evaluation_size = (
        max(1, round(source_width * evaluation_scale)),
        max(1, round(source_height * evaluation_scale)),
    )
    evaluation_masks = [
        cv2.resize(item.mask.astype(np.uint8), evaluation_size, interpolation=cv2.INTER_NEAREST).astype(
            bool
        )
        for item in hypotheses
    ]
    global_features = cv2.resize(
        colors.feature_sets["global_lab"], evaluation_size, interpolation=cv2.INTER_AREA
    )
    evaluation_lab = cv2.resize(colors.lab, evaluation_size, interpolation=cv2.INTER_AREA)
    chroma_gradients: list[np.ndarray] = []
    for channel in (1, 2):
        horizontal = cv2.Sobel(evaluation_lab[..., channel], cv2.CV_32F, 1, 0, ksize=3)
        vertical = cv2.Sobel(evaluation_lab[..., channel], cv2.CV_32F, 0, 1, ksize=3)
        chroma_gradients.append(cv2.magnitude(horizontal, vertical))
    chroma_gradient = np.hypot(chroma_gradients[0], chroma_gradients[1])
    gradient_reference = float(np.percentile(chroma_gradient, 90))
    previews = [
        cv2.resize(item.astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST).astype(
            bool
        )
        for item in evaluation_masks
    ]
    # Compute every preview IoU in one matrix operation. The earlier pairwise Python loop
    # became the dominant runtime once k=5/6 and sparse projection tails were added.
    preview_matrix = np.stack(previews).reshape(len(previews), -1).astype(np.float32)
    intersections = preview_matrix @ preview_matrix.T
    preview_areas = preview_matrix.sum(axis=1)
    unions = preview_areas[:, None] + preview_areas[None, :] - intersections
    overlap_matrix = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions > 0,
    )
    evaluation_invalid = (
        cv2.resize(
            invalid_mask.astype(np.uint8), evaluation_size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        if invalid_mask is not None
        else np.zeros((evaluation_size[1], evaluation_size[0]), dtype=bool)
    )

    for hypothesis, evaluation_mask in zip(hypotheses, evaluation_masks):
        occupancy = float(evaluation_mask.mean())
        separation = _separation(evaluation_mask, global_features)
        spatial, component = _spatial_structure(evaluation_mask)
        frame_focus = _frame_focus(evaluation_mask)
        selected_count = int(evaluation_mask.sum())
        invalid_overlap = (
            float(np.count_nonzero(evaluation_mask & evaluation_invalid) / selected_count)
            if selected_count
            else 0.0
        )
        preliminary.append(
            {
                "hypothesis": hypothesis,
                "occupancy": occupancy,
                "separation": separation,
                "spatial": spatial,
                "component": component,
                "frame_focus": frame_focus,
                "invalid_overlap": invalid_overlap,
                "planar_explanation": _planar_explanation(evaluation_mask),
                "boundary_evidence": _boundary_evidence(
                    evaluation_mask, chroma_gradient, gradient_reference
                ),
            }
        )

    scored: list[ScoredMask] = []
    factor = _quality_factor(quality)
    family_names = np.array(
        [
            item["hypothesis"].family
            if isinstance(item["hypothesis"], MaskHypothesis)
            else ""
            for item in preliminary
        ]
    )
    local_family_flags = np.array(
        [name.startswith(("local_lab", "raw_local_lab")) for name in family_names],
        dtype=bool,
    )
    for index, item in enumerate(preliminary):
        hypothesis = item["hypothesis"]
        assert isinstance(hypothesis, MaskHypothesis)
        other_families = family_names != hypothesis.family
        other_families[index] = False
        local_families = local_family_flags & other_families
        agreement = (
            float(overlap_matrix[index, other_families].max())
            if np.any(other_families)
            else 0.0
        )
        local_residual_support = (
            float(overlap_matrix[index, local_families].max())
            if np.any(local_families)
            else 0.0
        )

        metrics = {
            "occupancy": float(item["occupancy"]),
            "color_separation": float(item["separation"]),
            "spatial_structure": float(item["spatial"]),
            "coarse_component": float(item["component"]),
            "frame_focus": float(item["frame_focus"]),
            "planar_explanation": float(item["planar_explanation"]),
            "boundary_evidence": float(item["boundary_evidence"]),
            "cross_family_agreement": agreement,
            "local_residual_support": local_residual_support,
            "quality_factor": factor,
            "invalid_overlap": float(item["invalid_overlap"]),
            "primitive_distribution_evidence": float(
                hypothesis.metadata.get("distribution_evidence", 0.0)
            ),
        }
        structure_score = (
            0.40 * metrics["color_separation"]
            + 0.40 * metrics["spatial_structure"]
            + 0.06 * metrics["coarse_component"]
            + 0.10 * metrics["frame_focus"]
            # Color spaces are correlated evidence, not independent votes.
            + 0.04 * min(metrics["cross_family_agreement"], 0.85)
        )
        field_likeness = np.clip((metrics["planar_explanation"] - 0.28) / 0.45, 0.0, 1.0)
        weak_boundary = np.clip((0.85 - metrics["boundary_evidence"]) / 0.65, 0.0, 1.0)
        planar_penalty = 1.0 - 0.82 * field_likeness * weak_boundary
        structure_score *= float(planar_penalty)
        # Color partitions often arrive as both a sparse structure and its background
        # complement. Above 62% occupancy, apply a soft canonical-foreground prior; this
        # resolves that symmetry without forbidding legitimately broad masks.
        foreground_prior = 1.0 - 0.32 * float(
            np.clip((metrics["occupancy"] - 0.62) / 0.38, 0.0, 1.0)
        )
        metrics["foreground_prior"] = foreground_prior
        structure_score *= foreground_prior
        if hypothesis.source.endswith(":primitive_graph"):
            # Graph masks are proposals derived from another hypothesis, not independent
            # evidence. A small complexity prior prevents fragment pruning from winning
            # solely because it makes a mask artificially compact.
            structure_score *= 0.88
        # Saturated glare has no recoverable chromatic evidence. Penalize overlap but keep
        # it soft because a true foreground may legitimately cross a small reflection.
        structure_score *= 1.0 - 0.42 * metrics["invalid_overlap"]
        if metrics["primitive_distribution_evidence"] > 0:
            structure_score = 0.88 * structure_score + 0.12 * metrics[
                "primitive_distribution_evidence"
            ]
        # Confidence is deliberately conservative until calibrated on independent real data.
        confidence = structure_score * factor
        confidence *= 1.0 - 0.48 * metrics["invalid_overlap"]
        if metrics["spatial_structure"] < 0.08:
            confidence *= 0.55
        scored.append(
            ScoredMask(
                mask=hypothesis.mask,
                source=hypothesis.source,
                family=hypothesis.family,
                score=float(np.clip(structure_score, 0, 1)),
                confidence=float(np.clip(confidence, 0, 0.92)),
                metrics=metrics,
                evidence_map=hypothesis.evidence_map,
            )
        )

    scored.sort(key=lambda candidate: (-candidate.score, candidate.source))
    return scored
