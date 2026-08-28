"""Small deterministic engineering regression suite, not a research benchmark."""

from __future__ import annotations

import argparse
import json

import numpy as np

from chromarecover import recover
from chromarecover.config import RecoverConfig
from chromarecover.synthetic import make_chromatic_pattern


def iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def evaluate(count: int, size: int) -> dict[str, object]:
    confidence_threshold = RecoverConfig().confidence_threshold
    recoveries: list[float] = []
    structured_statuses: list[str] = []
    negative_confidences: list[float] = []
    negative_statuses: list[str] = []

    for seed in range(count):
        case = make_chromatic_pattern(size=size, seed=seed, structured=True)
        result = recover(case.image, mode="digital", top_k=3)
        assert case.mask is not None
        recoveries.append(max(iou(candidate.mask, case.mask) for candidate in result.candidates))
        structured_statuses.append(result.status)

        negative = make_chromatic_pattern(size=size, seed=seed, structured=False)
        negative_result = recover(negative.image, mode="digital", top_k=3)
        negative_confidences.append(
            negative_result.best.overall_confidence if negative_result.best else 0.0
        )
        negative_statuses.append(negative_result.status)

    return {
        "generator": "dots-v1",
        "source_group_split": "one seed is one source_group_id; no train/test use",
        "cases_per_class": count,
        "structured": {
            "median_top3_iou": float(np.median(recoveries)),
            "minimum_top3_iou": float(np.min(recoveries)),
            "top3_iou_at_least_0_70": float(np.mean(np.asarray(recoveries) >= 0.70)),
            "ok_rate": structured_statuses.count("ok") / count,
        },
        "no_structure": {
            "abstention_rate": negative_statuses.count("uncertain") / count,
            "maximum_candidate_confidence": float(np.max(negative_confidences)),
            "production_confidence_threshold": confidence_threshold,
            "confidence_at_least_threshold_rate": float(
                np.mean(np.asarray(negative_confidences) >= confidence_threshold)
            ),
            "confidence_at_least_0_8_rate": float(
                np.mean(np.asarray(negative_confidences) >= 0.8)
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--assert-thresholds", action="store_true")
    arguments = parser.parse_args()
    report = evaluate(arguments.count, arguments.size)
    print(json.dumps(report, indent=2))
    if arguments.assert_thresholds:
        structured = report["structured"]
        no_structure = report["no_structure"]
        assert isinstance(structured, dict) and isinstance(no_structure, dict)
        passed = (
            (arguments.size <= 128 or structured["top3_iou_at_least_0_70"] >= 0.8)
            and no_structure["abstention_rate"] >= 0.95
            and no_structure["confidence_at_least_threshold_rate"] == 0.0
        )
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
