"""End-to-end public recovery API."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import __version__ as pillow_version

from ._version import __version__
from .color import build_representations
from .config import RecoverConfig
from .enhancement import make_overlay, make_structure_evidence
from .geometry import inverse_map_evidence, inverse_map_mask, normalize_geometry
from .hypotheses import generate_hypotheses
from .io import decode_image
from .photometry import estimate_capture_route, normalize_camera_photometry
from .primitive_distribution import generate_primitive_distribution_hypotheses
from .quality import assess_quality
from .recovery import ScoredMask, score_hypotheses
from .restoration import wiener_restoration_hypotheses
from .semantics import recognize_digit_sequence, semantic_consensus
from .types import Candidate, RecoverResult


def _select_diverse(scored: list[ScoredMask], count: int) -> list[ScoredMask]:
    selected: list[ScoredMask] = []
    for candidate in scored:
        too_similar = False
        for existing in selected:
            union = np.count_nonzero(candidate.mask | existing.mask)
            iou = np.count_nonzero(candidate.mask & existing.mask) / union if union else 1.0
            if iou > 0.92:
                too_similar = True
                break
        if not too_similar:
            selected.append(candidate)
        if len(selected) >= count:
            break
    return selected


def _mask_id(mask: np.ndarray) -> str:
    return hashlib.sha1(np.packbits(mask).tobytes()).hexdigest()[:12]


def _bounded_presentation_shape(
    original_shape: tuple[int, int], max_side: int
) -> tuple[int, int]:
    height, width = original_shape
    scale = min(1.0, max_side / max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def recover(
    source: str | Path | np.ndarray,
    *,
    mode: str = "digital",
    semantics: str = "none",
    top_k: int = 3,
    return_debug: bool = False,
    config: RecoverConfig | None = None,
    roi: tuple[int, int, int, int] | None = None,
    document_corners: list[list[float]] | None = None,
    auto_rectify: bool = True,
    auto_dewarp: bool = True,
) -> RecoverResult:
    """Recover top-k color-carried spatial structures from an image.

    Optional digit-sequence recognition runs only after visual candidates are selected.
    It cannot create or rerank masks, override abstention, or infer missing pixels.
    """

    valid_modes = {"auto", "digital", "camera", "camera-paper", "camera-screen"}
    if mode not in valid_modes:
        raise ValueError(
            "mode must be one of: auto, digital, camera, camera-paper, camera-screen"
        )
    if semantics not in {"none", "auto"}:
        raise ValueError("The alpha release supports semantics='none' or 'auto' only")
    if top_k < 1 or top_k > 10:
        raise ValueError("top_k must be between 1 and 10")
    settings = config or RecoverConfig()
    started = perf_counter()

    decoded = decode_image(
        source,
        max_pixels=settings.max_pixels,
        max_file_bytes=settings.max_file_bytes,
        max_icc_profile_bytes=settings.max_icc_profile_bytes,
        analysis_max_side=settings.analysis_max_side,
    )
    auto_route: dict[str, object] | None = None
    if mode == "auto":
        auto_route = estimate_capture_route(decoded.analysis_rgb)
        resolved_mode = str(auto_route["resolved_mode"])
    else:
        resolved_mode = mode
    geometry_mode = "camera" if resolved_mode.startswith("camera") else resolved_mode
    geometry = normalize_geometry(
        decoded.rgb,
        mode=geometry_mode,
        max_side=settings.analysis_max_side,
        roi=roi,
        document_corners=document_corners,
        auto_rectify=auto_rectify,
        auto_dewarp=auto_dewarp,
    )
    analysis_rgb = geometry.rgb
    invalid_mask = np.zeros(analysis_rgb.shape[:2], dtype=bool)
    photometry_metadata: dict[str, object] = {"applied": False}
    photometry_steps: list[str] = []
    photometry_warnings: list[str] = []
    if resolved_mode.startswith("camera"):
        capture_profile = resolved_mode.removeprefix("camera-")
        if capture_profile == "camera":
            capture_profile = "auto"
        photometry = normalize_camera_photometry(analysis_rgb, profile=capture_profile)
        corrected_rgb = photometry.rgb
        invalid_mask = photometry.invalid_mask
        photometry_metadata = {
            "applied": True,
            "glare": photometry.glare_score,
            "shadow": photometry.shadow_score,
            "fold": photometry.fold_score,
            "moire": photometry.moire_score,
            "banding": photometry.banding_score,
            "capture_profile": photometry.capture_profile,
            **photometry.metadata,
        }
        photometry_steps = photometry.steps
        photometry_warnings = photometry.warnings
    else:
        corrected_rgb = analysis_rgb

    decoded.input_info["mode_requested"] = mode
    decoded.input_info["mode_resolved"] = resolved_mode
    if auto_route is not None:
        decoded.input_info["auto_route"] = auto_route
    decoded.input_info["analysis_width"] = int(corrected_rgb.shape[1])
    decoded.input_info["analysis_height"] = int(corrected_rgb.shape[0])
    decoded.input_info["geometry"] = geometry.metadata
    decoded.input_info["photometry"] = photometry_metadata
    if mode == "auto":
        decoded.preprocess.append(f"auto_mode:{resolved_mode}")
    decoded_elapsed = perf_counter()

    colors = build_representations(
        corrected_rgb,
        settings.local_sigma_ratio,
        raw_rgb=analysis_rgb if resolved_mode.startswith("camera") else None,
    )
    blur_source = decoded.rgb
    geometry_corners = geometry.metadata.get("corners")
    if geometry_corners is not None:
        corner_array = np.asarray(geometry_corners, dtype=np.float32)
        x0 = max(0, int(np.floor(corner_array[:, 0].min())))
        y0 = max(0, int(np.floor(corner_array[:, 1].min())))
        x1 = min(decoded.rgb.shape[1], int(np.ceil(corner_array[:, 0].max())) + 1)
        y1 = min(decoded.rgb.shape[0], int(np.ceil(corner_array[:, 1].max())) + 1)
        if x1 > x0 and y1 > y0:
            blur_source = decoded.rgb[y0:y1, x0:x1]
            decoded.preprocess.append("quality_blur_source:geometry_bbox")
    quality = assess_quality(corrected_rgb, colors, blur_rgb=blur_source)
    quality.perspective = geometry.perspective_score
    if resolved_mode.startswith("camera"):
        quality.glare = max(quality.glare, photometry.glare_score)
        quality.shadow = photometry.shadow_score
        quality.fold = photometry.fold_score
        quality.moire = photometry.moire_score
        quality.banding = photometry.banding_score
        quality.invalid_fraction = float(invalid_mask.mean())
        quality.warnings = list(
            dict.fromkeys(quality.warnings + geometry.warnings + photometry_warnings)
        )
    if min(corrected_rgb.shape[:2]) <= 128:
        quality.warnings.append("low_effective_resolution")
    restoration_debug: dict[str, np.ndarray] = {}
    if quality.blur > 0.68 and quality.texture_support > 0.015:
        for index, restored in enumerate(
            wiener_restoration_hypotheses(corrected_rgb)[:2], 1
        ):
            restored_colors = build_representations(
                restored.rgb, settings.local_sigma_ratio
            )
            colors.feature_sets[f"restored_{index}_global_lab"] = (
                restored_colors.feature_sets["global_lab"]
            )
            colors.feature_sets[f"restored_{index}_hybrid"] = restored_colors.feature_sets[
                "hybrid"
            ]
            restoration_debug[f"restoration_{index}"] = restored.rgb
        if restoration_debug:
            decoded.preprocess.append("single_frame_wiener_hypotheses")
    color_elapsed = perf_counter()

    hypotheses = generate_hypotheses(colors.feature_sets, settings)
    primitive_result = generate_primitive_distribution_hypotheses(colors.lab, settings)
    hypotheses.extend(primitive_result.hypotheses)
    scored = score_hypotheses(hypotheses, colors, quality, invalid_mask=invalid_mask)
    # Always retain a runner-up for ambiguity checks, even when the caller requests top_k=1.
    status_candidates = _select_diverse(scored, max(top_k, 2))
    selected = status_candidates[:top_k]
    recovery_elapsed = perf_counter()

    original_height, original_width = decoded.rgb.shape[:2]
    presentation_height, presentation_width = _bounded_presentation_shape(
        (original_height, original_width), settings.presentation_max_side
    )
    presentation_shape = (presentation_height, presentation_width)
    if presentation_shape == (original_height, original_width):
        presentation_rgb = decoded.rgb
    else:
        presentation_rgb = cv2.resize(
            decoded.rgb,
            (presentation_width, presentation_height),
            interpolation=cv2.INTER_AREA,
        )
        decoded.preprocess.append(
            f"presentation_resize:{presentation_width}x{presentation_height}"
        )
    scale_x = (presentation_width - 1) / max(original_width - 1, 1)
    scale_y = (presentation_height - 1) / max(original_height - 1, 1)
    decoded.input_info["presentation"] = {
        "width": presentation_width,
        "height": presentation_height,
        "max_side": settings.presentation_max_side,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }
    candidates: list[Candidate] = []
    for rank, scored_mask in enumerate(selected, 1):
        analysis_evidence, analysis_structure = make_structure_evidence(
            scored_mask.mask.astype(bool), scored_mask.evidence_map
        )
        mask = inverse_map_mask(
            scored_mask.mask.astype(bool), geometry, (original_height, original_width)
        )
        structure_mask = inverse_map_mask(
            analysis_structure, geometry, (original_height, original_width)
        )
        evidence_map = inverse_map_evidence(
            analysis_evidence,
            geometry,
            presentation_shape,
            coordinate_shape=(original_height, original_width),
        )
        presentation_mask = cv2.resize(
            mask.astype(np.uint8),
            (presentation_width, presentation_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        presentation_structure = cv2.resize(
            structure_mask.astype(np.uint8),
            (presentation_width, presentation_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        candidates.append(
            Candidate(
                id=_mask_id(mask),
                rank=rank,
                mask=mask,
                structure_score=scored_mask.score,
                structure_confidence=scored_mask.score,
                capture_confidence=scored_mask.confidence,
                decision_confidence=scored_mask.confidence,
                source_hypothesis=scored_mask.source,
                metrics=scored_mask.metrics,
                overlay=make_overlay(
                    presentation_rgb, presentation_mask, presentation_structure
                ),
                structure_mask=structure_mask,
                evidence_map=evidence_map,
                transform={
                    "model": geometry.metadata.get("source", "unknown"),
                    "analysis_to_original": geometry.inverse_matrix.tolist(),
                    "original_to_analysis": geometry.forward_matrix.tolist(),
                    "interpolation": "nearest",
                    "artifact_spaces": {
                        "mask": {
                            "coordinate_space": "original",
                            "width": original_width,
                            "height": original_height,
                        },
                        "structure_mask": {
                            "coordinate_space": "original",
                            "width": original_width,
                            "height": original_height,
                        },
                        "evidence_map": {
                            "coordinate_space": "presentation",
                            "width": presentation_width,
                            "height": presentation_height,
                            "original_to_artifact_scale": [scale_x, scale_y],
                        },
                        "overlay": {
                            "coordinate_space": "presentation",
                            "width": presentation_width,
                            "height": presentation_height,
                            "original_to_artifact_scale": [scale_x, scale_y],
                        },
                    },
                },
            )
        )

    decision_factor = 1.0
    if not candidates or "low_effective_resolution" in quality.warnings:
        status = "uncertain"
    elif (
        "severe_blur" in quality.warnings
        or quality.clipping > 0.35
        or quality.glare > 0.18
        or quality.shadow > 0.78
        or quality.fold > 0.78
        or quality.moire > 0.78
    ):
        status = "retry_recommended" if mode == "auto" or mode.startswith("camera") else "uncertain"
    else:
        best = status_candidates[0].confidence
        runner_up = status_candidates[1] if len(status_candidates) > 1 else None
        margin = best - runner_up.confidence if runner_up is not None else best
        consensus_iou = 0.0
        if runner_up is not None:
            union = np.count_nonzero(status_candidates[0].mask | runner_up.mask)
            consensus_iou = (
                np.count_nonzero(status_candidates[0].mask & runner_up.mask) / union if union else 1.0
            )
        margin_support = float(
            np.clip(margin / max(settings.ambiguity_margin, 1e-6), 0.0, 1.0)
        )
        consensus_position = float(np.clip((consensus_iou - 0.45) / 0.35, 0.0, 1.0))
        # Smoothstep turns runner-up overlap into continuous ambiguity evidence. A 3% resize
        # can no longer remove the entire ambiguity penalty by crossing one IoU threshold.
        consensus_support = consensus_position * consensus_position * (
            3.0 - 2.0 * consensus_position
        )
        ambiguity_support = max(margin_support, consensus_support)
        decision_factor = 0.45 + 0.55 * ambiguity_support
        best_metrics = status_candidates[0].metrics
        best_metrics["runner_up_margin"] = margin
        best_metrics["runner_up_consensus_iou"] = consensus_iou
        best_metrics["margin_support"] = margin_support
        best_metrics["consensus_support"] = consensus_support
        best_metrics["ambiguity_support"] = ambiguity_support
        residual_evidence = (
            best_metrics["boundary_evidence"] >= 0.55
            and best_metrics["local_residual_support"] >= 0.30
            and best_metrics["color_separation"] >= 0.60
        )
        discrete_region_evidence = (
            best_metrics["boundary_evidence"] >= 0.80
            and best_metrics["color_separation"] >= 0.70
            and best_metrics["planar_explanation"] < 0.45
        )
        primitive_distribution_evidence = (
            best_metrics.get("primitive_distribution_evidence", 0.0) >= 0.42
            and best_metrics["spatial_structure"] >= 0.40
        )
        spatial_distribution = (
            0.55 * best_metrics["spatial_structure"]
            + 0.45 * best_metrics["coarse_component"]
        )
        chromatic_support = (
            0.40 * best_metrics["color_separation"]
            + 0.32 * best_metrics["cross_family_agreement"]
            + 0.28 * best_metrics["local_residual_support"]
        )
        nuisance_resistance = best_metrics["frame_focus"] ** 3 * (
            1.0 - best_metrics["planar_explanation"]
        ) ** 2
        sparse_support = float(np.clip(best_metrics["occupancy"] / 0.06, 0.0, 1.0))
        mosaic_distribution_score = float(
            0.40 * spatial_distribution
            + 0.35 * chromatic_support
            + 0.15 * nuisance_resistance
            + 0.10 * sparse_support
        )
        best_metrics["mosaic_distribution_score"] = mosaic_distribution_score
        mosaic_distribution_evidence = mosaic_distribution_score >= 0.77
        evidence_gate = (
            residual_evidence
            or discrete_region_evidence
            or primitive_distribution_evidence
            or mosaic_distribution_evidence
        )
        status = (
            "ok"
            if best * decision_factor >= settings.confidence_threshold
            and evidence_gate
            else "uncertain"
        )
        if margin < settings.ambiguity_margin:
            if consensus_support >= margin_support and consensus_iou > 0.45:
                candidates[0].warnings.append("cross_hypothesis_consensus")
            if decision_factor < 0.98:
                candidates[0].warnings.append("multiple_plausible_structures")
        if not evidence_gate:
            candidates[0].warnings.append("weak_structure_evidence")
            decision_factor = min(decision_factor, 0.65)

    status_factor = {"ok": 1.0, "uncertain": 0.62, "retry_recommended": 0.32}[status]
    for candidate in candidates:
        candidate.decision_confidence *= decision_factor * status_factor
        candidate.metrics["decision_factor"] = decision_factor
        candidate.metrics["status_factor"] = status_factor

    semantics_result: dict[str, object] = {"mode": "none"}
    if semantics == "auto":
        for candidate in candidates:
            if candidate.evidence_map is None:
                continue
            candidate.semantics = recognize_digit_sequence(candidate.evidence_map).to_dict()
            candidate.semantics["candidate_rank"] = candidate.rank
            candidate.semantics["candidate_decision_confidence"] = round(
                float(candidate.decision_confidence), 6
            )
            candidate.semantics["candidate_capture_confidence"] = round(
                float(candidate.capture_confidence), 6
            )
            candidate.semantics["candidate_source_family"] = (
                candidate.source_hypothesis.split(":", maxsplit=1)[0]
            )
            candidate.semantics["candidate_boundary_evidence"] = round(
                float(candidate.metrics.get("boundary_evidence", 0.0)), 6
            )
            candidate.semantics["candidate_color_separation"] = round(
                float(candidate.metrics.get("color_separation", 0.0)), 6
            )
            candidate.semantics["candidate_weak_structure_evidence"] = (
                "weak_structure_evidence" in candidate.warnings
            )
            candidate.semantics["candidate_quality_blocked"] = bool(
                {"severe_blur", "low_effective_resolution"} & set(quality.warnings)
            )
        semantics_result = semantic_consensus(
            [candidate.semantics for candidate in candidates if candidate.semantics]
        )
        decoded.preprocess.append("post_recovery_digit_shape_consensus")

    finished = perf_counter()
    debug_artifacts: dict[str, np.ndarray] = {}
    if return_debug:
        debug_artifacts = {
            "analysis_input": analysis_rgb,
            "analysis_corrected": corrected_rgb,
            "invalid_regions": invalid_mask,
            "primitive_labels": primitive_result.debug_rgb,
            **restoration_debug,
        }
        decoded.preprocess.append("debug_bundle:enabled")

    return RecoverResult(
        status=status,
        candidates=candidates,
        quality=quality,
        input_info=decoded.input_info,
        preprocess=decoded.preprocess
        + geometry.steps
        + photometry_steps
        + [
            "relative_chromatic_residuals",
            "multi_family_hypotheses",
            "contour_primitives",
            "image_native_primitives",
            "local_primitive_distribution",
            "spatial_graph_grouping",
            "structure_ranking",
            "inverse_mask_mapping",
        ],
        timing_ms={
            "decode": (decoded_elapsed - started) * 1000,
            "color_and_quality": (color_elapsed - decoded_elapsed) * 1000,
            "hypotheses_and_ranking": (recovery_elapsed - color_elapsed) * 1000,
            "presentation": (finished - recovery_elapsed) * 1000,
            "total": (finished - started) * 1000,
        },
        algorithm_version=f"cre-{__version__}",
        config_fingerprint=settings.fingerprint(),
        config=asdict(settings),
        runtime={
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pillow": pillow_version,
        },
        original=decoded.rgb,
        semantics=semantics_result,
        debug_artifacts=debug_artifacts,
    )
