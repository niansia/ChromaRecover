"""Evidence-preserving burst alignment and robust multi-frame fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import RecoverConfig
from .exceptions import InvalidInputError
from .io import decode_image, probe_image_size
from .types import RecoverResult


@dataclass
class BurstFusionResult:
    rgb: np.ndarray
    valid_count: np.ndarray
    uncertainty: np.ndarray
    transforms: list[np.ndarray]
    rejected_frames: list[int] = field(default_factory=list)


def _glare_and_clipping_mask(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(values, cv2.COLOR_RGB2LAB)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    maximum = values.max(axis=2)
    minimum = values.min(axis=2)
    return ((maximum > 0.965) & (chroma < 14.0)) | (minimum < 0.008) | (maximum > 0.995)


def _align_to_reference(reference: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = reference.shape[:2]
    scale = min(1.0, 720.0 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    reference_preview = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
    frame_preview = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    reference_gray = cv2.cvtColor(reference_preview, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.cvtColor(frame_preview, cv2.COLOR_RGB2GRAY)
    detector = cv2.ORB_create(nfeatures=1800, fastThreshold=7)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(reference_gray, None)
    frame_keypoints, frame_descriptors = detector.detectAndCompute(frame_gray, None)
    if reference_descriptors is None or frame_descriptors is None:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(frame_descriptors, reference_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < 0.76 * second.distance]
    if len(good) < 10:
        return None
    frame_points = np.float32([frame_keypoints[item.queryIdx].pt for item in good])
    reference_points = np.float32([reference_keypoints[item.trainIdx].pt for item in good])
    preview_transform, inliers = cv2.findHomography(
        frame_points, reference_points, cv2.RANSAC, 3.0
    )
    if preview_transform is None or inliers is None or int(inliers.sum()) < 8:
        return None
    scaling = np.diag([scale, scale, 1.0])
    transform = np.linalg.inv(scaling) @ preview_transform @ scaling
    aligned = cv2.warpPerspective(
        frame,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned, transform


def fuse_burst(frames: list[np.ndarray]) -> BurstFusionResult:
    if not 2 <= len(frames) <= 12:
        raise ValueError("burst must contain between 2 and 12 frames")
    reference = np.asarray(frames[0], dtype=np.uint8)
    if any(np.asarray(frame).shape != reference.shape for frame in frames):
        raise InvalidInputError("All burst frames must have identical dimensions")

    aligned_frames = [reference]
    transforms = [np.eye(3, dtype=np.float64)]
    rejected: list[int] = []
    for index, frame in enumerate(frames[1:], 1):
        candidate = np.asarray(frame, dtype=np.uint8)
        # Only bypass registration for genuinely identical observations. A generous
        # pixel-difference shortcut silently accepted small hand-held translations and
        # mixed edges during fusion.
        difference = float(np.mean(np.abs(candidate.astype(np.float32) - reference)))
        if difference < 0.5:
            aligned_frames.append(candidate)
            transforms.append(np.eye(3, dtype=np.float64))
            continue
        aligned = _align_to_reference(reference, candidate)
        if aligned is None:
            rejected.append(index)
            continue
        aligned_frames.append(aligned[0])
        transforms.append(aligned[1])

    if len(aligned_frames) < 2:
        raise InvalidInputError("Burst frames could not be aligned to a common reference")

    stack = np.stack(aligned_frames).astype(np.float32)
    invalid = np.stack([_glare_and_clipping_mask(frame) for frame in aligned_frames])
    median = np.median(stack, axis=0)
    color_deviation = np.linalg.norm(stack - median[None, ...], axis=3)
    if len(aligned_frames) >= 3:
        deviation_limit = np.percentile(color_deviation, 65, axis=0) + 18.0
        invalid |= color_deviation > deviation_limit[None, ...]

    valid = ~invalid
    gray_stack = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in aligned_frames]
    ).astype(np.float32)
    sharpness = np.abs(
        np.stack([cv2.Laplacian(gray, cv2.CV_32F) for gray in gray_stack])
    )
    sharpness /= np.percentile(sharpness, 95, axis=(1, 2), keepdims=True) + 1e-6
    weights = valid.astype(np.float32) * (0.35 + np.clip(sharpness, 0.0, 2.0))
    weight_sum = weights.sum(axis=0)
    fused = (stack * weights[..., None]).sum(axis=0) / np.maximum(weight_sum[..., None], 1e-6)
    no_observation = weight_sum <= 0
    fused[no_observation] = reference[no_observation]
    valid_count = valid.sum(axis=0).astype(np.uint8)
    dispersion = np.sqrt(np.mean(np.square(stack - median[None, ...]), axis=(0, 3))) / 48.0
    uncertainty = np.clip(dispersion + 0.65 * no_observation, 0.0, 1.0).astype(np.float32)
    return BurstFusionResult(
        rgb=np.clip(fused, 0, 255).astype(np.uint8),
        valid_count=valid_count,
        uncertainty=uncertainty,
        transforms=transforms,
        rejected_frames=rejected,
    )


def recover_burst(
    sources: list[str | Path | np.ndarray | Image.Image],
    *,
    mode: str = "camera",
    top_k: int = 3,
    return_debug: bool = False,
    config: RecoverConfig | None = None,
    **recover_options: object,
) -> RecoverResult:
    """Align and fuse a burst, then recover structures from real observed pixels."""

    settings = config or RecoverConfig()
    if not 2 <= len(sources) <= 12:
        raise InvalidInputError("Burst must contain between 2 and 12 frames")

    total_pixels = 0
    for source in sources:
        width, height = probe_image_size(
            source,
            max_pixels=settings.max_pixels,
            max_file_bytes=settings.max_file_bytes,
        )
        total_pixels += width * height
        if total_pixels > settings.max_burst_total_pixels:
            raise InvalidInputError(
                f"Burst has {total_pixels:,} total pixels; allowed maximum is "
                f"{settings.max_burst_total_pixels:,}. Reduce frame count, crop, or downsample."
            )

    decoded = []
    decoded_total_pixels = 0
    for source in sources:
        item = decode_image(
            source,
            max_pixels=settings.max_pixels,
            max_file_bytes=settings.max_file_bytes,
            max_icc_profile_bytes=settings.max_icc_profile_bytes,
            analysis_max_side=settings.analysis_max_side,
        )
        decoded_total_pixels += int(item.input_info["width"]) * int(
            item.input_info["height"]
        )
        if decoded_total_pixels > settings.max_burst_total_pixels:
            raise InvalidInputError(
                "Burst dimensions changed while decoding and now exceed the aggregate "
                f"{settings.max_burst_total_pixels:,}-pixel limit"
            )
        decoded.append(item)
    total_pixels = decoded_total_pixels
    fusion = fuse_burst([item.rgb for item in decoded])
    # Local import avoids an API/burst module cycle.
    from .api import recover

    result = recover(
        fusion.rgb,
        mode=mode,
        top_k=top_k,
        return_debug=return_debug,
        config=settings,
        **recover_options,
    )
    recoverable = float(np.mean(fusion.valid_count > 0))
    result.quality.recoverable_fraction = recoverable
    result.input_info["burst"] = {
        "frame_count": len(sources),
        "aligned_count": len(fusion.transforms),
        "rejected_frames": fusion.rejected_frames,
        "recoverable_fraction": recoverable,
        "input_total_pixels": total_pixels,
        "total_pixel_limit": settings.max_burst_total_pixels,
        "transforms_to_reference": [transform.tolist() for transform in fusion.transforms],
    }
    result.preprocess.insert(0, "burst_alignment_and_robust_fusion")
    if return_debug:
        result.debug_artifacts["burst_valid_count"] = (
            fusion.valid_count.astype(np.float32) / max(len(fusion.transforms), 1)
        )
        result.debug_artifacts["burst_uncertainty"] = fusion.uncertainty
    if recoverable < 0.92:
        result.quality.warnings.append("burst_incomplete_coverage")
        for candidate in result.candidates:
            candidate.decision_confidence *= 0.75 + 0.25 * recoverable
    return result
