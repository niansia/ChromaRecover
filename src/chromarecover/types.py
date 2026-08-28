"""Stable result types shared by the Python API and CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class QualityReport:
    blur: float
    clipping: float
    glare: float
    color_spread: float
    perspective: float = 0.0
    shadow: float = 0.0
    fold: float = 0.0
    moire: float = 0.0
    banding: float = 0.0
    invalid_fraction: float = 0.0
    texture_support: float = 1.0
    recoverable_fraction: float = 1.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blur": round(float(self.blur), 6),
            "clipping": round(float(self.clipping), 6),
            "glare": round(float(self.glare), 6),
            "color_spread": round(float(self.color_spread), 6),
            "perspective": round(float(self.perspective), 6),
            "shadow": round(float(self.shadow), 6),
            "fold": round(float(self.fold), 6),
            "moire": round(float(self.moire), 6),
            "banding": round(float(self.banding), 6),
            "invalid_fraction": round(float(self.invalid_fraction), 6),
            "texture_support": round(float(self.texture_support), 6),
            "recoverable_fraction": round(float(self.recoverable_fraction), 6),
            "warnings": list(self.warnings),
        }


@dataclass
class Candidate:
    id: str
    rank: int
    mask: np.ndarray
    structure_score: float
    structure_confidence: float
    capture_confidence: float
    decision_confidence: float
    source_hypothesis: str
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    overlay: np.ndarray | None = None
    structure_mask: np.ndarray | None = None
    evidence_map: np.ndarray | None = None
    transform: dict[str, Any] = field(default_factory=dict)
    semantics: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_confidence(self) -> float:
        """Backward-compatible alias for decision confidence."""

        return self.decision_confidence

    @overall_confidence.setter
    def overall_confidence(self, value: float) -> None:
        self.decision_confidence = value

    def summary(self, paths: dict[str, str | None] | None = None) -> dict[str, Any]:
        paths = paths or {}
        return {
            "id": self.id,
            "rank": self.rank,
            "mask_path": paths.get("mask"),
            "structure_mask_path": paths.get("structure_mask"),
            "evidence_map_path": paths.get("evidence_map"),
            "overlay_path": paths.get("overlay"),
            "structure_score": round(float(self.structure_score), 6),
            "structure_confidence": round(float(self.structure_confidence), 6),
            "capture_confidence": round(float(self.capture_confidence), 6),
            "decision_confidence": round(float(self.decision_confidence), 6),
            "overall_confidence": round(float(self.overall_confidence), 6),
            "source_hypothesis": self.source_hypothesis,
            "metrics": {key: round(float(value), 6) for key, value in self.metrics.items()},
            "warnings": list(self.warnings),
            "transform": self.transform,
            "semantics": self.semantics,
        }


@dataclass
class RecoverResult:
    status: str
    candidates: list[Candidate]
    quality: QualityReport
    input_info: dict[str, Any]
    preprocess: list[str]
    timing_ms: dict[str, float]
    algorithm_version: str
    config_fingerprint: str
    config: dict[str, Any]
    runtime: dict[str, str]
    original: np.ndarray = field(repr=False)
    semantics: dict[str, Any] = field(default_factory=dict)
    debug_artifacts: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def to_dict(
        self,
        paths: dict[str, dict[str, str | None]] | None = None,
        debug_paths: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        paths = paths or {}
        return {
            "schema_version": "0.5",
            "algorithm_version": self.algorithm_version,
            "config_fingerprint": self.config_fingerprint,
            "config": self.config,
            "runtime": self.runtime,
            "status": self.status,
            "input": self.input_info,
            "quality": self.quality.to_dict(),
            "semantics": self.semantics,
            "candidates": [
                candidate.summary(paths.get(candidate.id))
                for candidate in self.candidates
            ],
            "debug": debug_paths or {},
            "preprocess": list(self.preprocess),
            "timing_ms": {key: round(float(value), 3) for key, value in self.timing_ms.items()},
        }

    def save(self, output_dir: str | Path) -> Path:
        """Write masks, overlays and result.json without storing extra private debug data."""

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        paths: dict[str, dict[str, str | None]] = {}
        for candidate in self.candidates:
            mask_name = f"mask_{candidate.rank:02d}.png"
            structure_name = f"structure_{candidate.rank:02d}.png"
            evidence_name = f"evidence_{candidate.rank:02d}.png"
            overlay_name = f"overlay_{candidate.rank:02d}.png"
            Image.fromarray((candidate.mask.astype(np.uint8) * 255), mode="L").save(
                destination / mask_name
            )
            overlay = candidate.overlay if candidate.overlay is not None else self.original
            Image.fromarray(overlay.astype(np.uint8), mode="RGB").save(destination / overlay_name)
            structure_path: str | None = None
            if candidate.structure_mask is not None:
                Image.fromarray(
                    candidate.structure_mask.astype(np.uint8) * 255, mode="L"
                ).save(destination / structure_name)
                structure_path = structure_name
            evidence_path: str | None = None
            if candidate.evidence_map is not None:
                Image.fromarray(
                    np.round(np.clip(candidate.evidence_map, 0, 1) * 255).astype(np.uint8),
                    mode="L",
                ).save(destination / evidence_name)
                evidence_path = evidence_name
            paths[candidate.id] = {
                "mask": mask_name,
                "structure_mask": structure_path,
                "evidence_map": evidence_path,
                "overlay": overlay_name,
            }

        debug_paths: dict[str, str] = {}
        if self.debug_artifacts:
            debug_directory = destination / "debug"
            debug_directory.mkdir(parents=True, exist_ok=True)
            for name, artifact in self.debug_artifacts.items():
                filename = f"{name}.png"
                values = np.asarray(artifact)
                if values.dtype == bool:
                    image = Image.fromarray(values.astype(np.uint8) * 255, mode="L")
                elif values.ndim == 2:
                    if np.issubdtype(values.dtype, np.floating):
                        values = np.round(np.clip(values, 0, 1) * 255).astype(np.uint8)
                    image = Image.fromarray(values.astype(np.uint8), mode="L")
                else:
                    image = Image.fromarray(values.astype(np.uint8), mode="RGB")
                image.save(debug_directory / filename)
                debug_paths[name] = f"debug/{filename}"

        result_path = destination / "result.json"
        result_path.write_text(
            json.dumps(self.to_dict(paths, debug_paths), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return result_path
