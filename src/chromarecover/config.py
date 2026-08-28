"""Serializable configuration for the recovery pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RecoverConfig:
    """Parameters with conservative, CPU-friendly defaults."""

    analysis_max_side: int = 384
    cluster_counts: tuple[int, ...] = (2, 3, 4, 5, 6)
    projection_quantiles: tuple[float, ...] = (0.08, 0.12, 0.25, 0.35, 0.65, 0.75, 0.88, 0.92)
    local_sigma_ratio: float = 0.045
    min_occupancy: float = 0.015
    max_occupancy: float = 0.985
    duplicate_iou: float = 0.975
    random_seed: int = 17
    # Full-resolution support masks remain auditable. This bound keeps their worst-case
    # resident set practical on ordinary contributor/desktop machines.
    # 50 MP covers common 48 MP phone captures. The presentation-space path has been
    # measured at about 0.82 GiB peak RSS for 8064x6048 camera input on the reference host.
    max_pixels: int = 50_000_000
    # Reject unexpectedly large compressed inputs before an image decoder allocates pixels.
    # In-memory PIL/NumPy callers have already paid that allocation cost, so this applies to
    # filesystem sources only.
    max_file_bytes: int = 100 * 1024 * 1024
    # ICC data is independently bounded before LittleCMS parses it. Typical display and
    # camera profiles are far smaller; oversized metadata is ignored, not treated as pixels.
    max_icc_profile_bytes: int = 4 * 1024 * 1024
    # Burst fusion keeps registered observations plus several float work arrays. Bound the
    # aggregate independently of the per-frame limit so a valid set of individually safe
    # images cannot exhaust memory when stacked. 80 MP covers six typical 12 MP frames;
    # larger phone bursts should be cropped or downsampled before fusion.
    max_burst_total_pixels: int = 80_000_000
    # Continuous evidence and overlays are presentation artifacts, not ranking inputs.
    presentation_max_side: int = 2_048
    max_kmeans_samples: int = 40_000
    kmeans_attempts: int = 2
    # Conservative alpha threshold; replace only after calibration on source-group-held-out data.
    confidence_threshold: float = 0.74
    ambiguity_margin: float = 0.035

    def __post_init__(self) -> None:
        if self.analysis_max_side < 64:
            raise ValueError("analysis_max_side must be at least 64")
        if not self.cluster_counts or any(count < 2 or count > 12 for count in self.cluster_counts):
            raise ValueError("cluster_counts must contain values from 2 through 12")
        if not self.projection_quantiles or any(
            quantile <= 0 or quantile >= 1 for quantile in self.projection_quantiles
        ):
            raise ValueError("projection_quantiles must be strictly between 0 and 1")
        if self.local_sigma_ratio <= 0:
            raise ValueError("local_sigma_ratio must be positive")
        if not 0 < self.min_occupancy < self.max_occupancy < 1:
            raise ValueError("occupancy limits must satisfy 0 < min < max < 1")
        if not 0 < self.duplicate_iou <= 1:
            raise ValueError("duplicate_iou must be in (0, 1]")
        if (
            self.max_pixels <= 0
            or self.max_file_bytes <= 0
            or self.max_icc_profile_bytes <= 0
            or self.max_burst_total_pixels <= 0
            or self.max_kmeans_samples <= 0
        ):
            raise ValueError("file, pixel, burst and sample limits must be positive")
        if self.presentation_max_side < 64:
            raise ValueError("presentation_max_side must be at least 64")
        if self.kmeans_attempts < 1 or self.kmeans_attempts > 10:
            raise ValueError("kmeans_attempts must be between 1 and 10")
        if not 0 < self.confidence_threshold < 1:
            raise ValueError("confidence_threshold must be strictly between 0 and 1")
        if self.ambiguity_margin < 0:
            raise ValueError("ambiguity_margin cannot be negative")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
