"""Evaluate camera geometry on the external CC BY 4.0 SmartDoc sample dataset."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from chromarecover import recover
from chromarecover.geometry import detect_content_quad, detect_document_quad


def _ground_truth(path: Path) -> dict[int, np.ndarray]:
    frames: dict[int, np.ndarray] = {}
    for frame in ET.parse(path).findall(".//frame"):
        if frame.attrib.get("rejected") == "true":
            continue
        points = {
            point.attrib["name"]: (float(point.attrib["x"]), float(point.attrib["y"]))
            for point in frame.findall("point")
        }
        if {"tl", "tr", "br", "bl"} <= points.keys():
            frames[int(frame.attrib["index"])] = np.float32(
                [points[name] for name in ("tl", "tr", "br", "bl")]
            )
    return frames


def _quad_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection, _polygon = cv2.intersectConvexConvex(
        left.astype(np.float32), right.astype(np.float32)
    )
    union = cv2.contourArea(left) + cv2.contourArea(right) - intersection
    return float(intersection / union) if union > 0 else 0.0


def _sample_indices(frame_ids: list[int], count: int) -> list[int]:
    positions = np.linspace(0, len(frame_ids) - 1, min(count, len(frame_ids)), dtype=int)
    return [frame_ids[int(position)] for position in positions]


def evaluate(dataset: Path, frames_per_video: int, recover_count: int) -> dict[str, object]:
    videos = sorted((dataset / "input_sample").rglob("*.avi"))
    ground_truth_files = {path.stem.removesuffix(".gt"): path for path in dataset.rglob("*.gt.xml")}
    if not videos:
        raise ValueError(f"No SmartDoc AVI files found under {dataset}")

    ious: list[float] = []
    normalized_errors: list[float] = []
    misses = 0
    sampled_frames: list[dict[str, object]] = []
    recovery_smokes: list[dict[str, object]] = []
    remaining_recovery = recover_count
    recovered_groups: set[str] = set()
    for video in videos:
        ground_truth_path = ground_truth_files.get(video.stem)
        if ground_truth_path is None:
            raise ValueError(f"No ground truth found for {video.name}")
        truth_by_frame = _ground_truth(ground_truth_path)
        frame_ids = _sample_indices(sorted(truth_by_frame), frames_per_video)
        capture = cv2.VideoCapture(str(video))
        try:
            for frame_id in frame_ids:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
                readable, bgr = capture.read()
                if not readable:
                    raise ValueError(f"Could not read {video.name} frame {frame_id}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                detected = detect_document_quad(rgb)
                detector = "edge_quad"
                if detected is None:
                    detected = detect_content_quad(rgb)
                    detector = "content_quad"
                truth = truth_by_frame[frame_id]
                record: dict[str, object] = {
                    "source_group_id": video.stem,
                    "frame": frame_id,
                    "detector": detector if detected is not None else "abstained",
                }
                if detected is None:
                    misses += 1
                else:
                    iou = _quad_iou(detected, truth)
                    height, width = rgb.shape[:2]
                    normalized_error = float(
                        np.sqrt(np.mean(np.sum(np.square(detected - truth), axis=1)))
                        / np.hypot(height, width)
                    )
                    ious.append(iou)
                    normalized_errors.append(normalized_error)
                    record.update(
                        quad_iou=round(iou, 6),
                        normalized_corner_rmse=round(normalized_error, 6),
                    )
                sampled_frames.append(record)

                if remaining_recovery > 0 and video.stem not in recovered_groups:
                    result = recover(
                        rgb,
                        mode="camera-paper",
                        top_k=1,
                        auto_dewarp=False,
                    )
                    recovery_smokes.append(
                        {
                            "source_group_id": video.stem,
                            "frame": frame_id,
                            "status": result.status,
                            "candidate_count": len(result.candidates),
                            "geometry_source": result.input_info["geometry"]["source"],
                            "warnings": result.quality.warnings,
                        }
                    )
                    remaining_recovery -= 1
                    recovered_groups.add(video.stem)
        finally:
            capture.release()

    total = len(sampled_frames)
    detection_rate = (total - misses) / total
    return {
        "dataset": "ICDAR2015-SmartDoc-Challenge1-sample",
        "license": "CC-BY-4.0",
        "source_url": "https://zenodo.org/records/1230218",
        "sample_archive_md5": "1ee5b7c290d707bd51c59f0b1c1a36f5",
        "citation": "Burie et al., ICDAR 2015 Competition on Smartphone Document Capture",
        "purpose": "camera geometry and quality only; not recovery-confidence calibration",
        "source_groups": len(videos),
        "sampled_frames": total,
        "detection_rate": detection_rate,
        "median_quad_iou": float(np.median(ious)) if ious else 0.0,
        "maximum_normalized_corner_rmse": max(normalized_errors, default=1.0),
        "frames": sampled_frames,
        "recovery_smokes": recovery_smokes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Path to the extracted sampleDataset folder")
    parser.add_argument("--frames-per-video", type=int, default=3)
    parser.add_argument("--recover-count", type=int, default=3)
    parser.add_argument("--assert-thresholds", action="store_true")
    arguments = parser.parse_args()
    if arguments.frames_per_video < 1 or arguments.recover_count < 0:
        parser.error("frame counts must be non-negative and frames-per-video must be positive")

    report = evaluate(arguments.dataset, arguments.frames_per_video, arguments.recover_count)
    print(json.dumps(report, indent=2))
    if arguments.assert_thresholds and not (
        float(report["detection_rate"]) >= 0.90
        and float(report["median_quad_iou"]) >= 0.80
        and float(report["maximum_normalized_corner_rmse"]) <= 0.04
    ):
        raise SystemExit("SmartDoc geometry thresholds failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
