"""Matched-histogram polygon-mosaic structure and false-positive gate."""

from __future__ import annotations

import argparse
import json

import numpy as np

from chromarecover import recover
from chromarecover.synthetic import make_polygon_mosaic


def iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def evaluate(count: int, size: int) -> dict[str, object]:
    recoveries: list[float] = []
    negative_statuses: list[str] = []
    negative_confidences: list[float] = []
    negative_semantic_accepts = 0
    for seed in range(40, 40 + count):
        case = make_polygon_mosaic(size=size, seed=seed, structured=True)
        result = recover(case.image, mode="digital", top_k=5)
        assert case.mask is not None
        recoveries.append(
            max(
                iou(candidate.structure_mask, case.mask)
                for candidate in result.candidates
                if candidate.structure_mask is not None
            )
        )

        negative = make_polygon_mosaic(size=size, seed=seed, structured=False)
        negative_result = recover(
            negative.image, mode="digital", top_k=3, semantics="auto"
        )
        negative_statuses.append(negative_result.status)
        negative_confidences.append(
            negative_result.best.decision_confidence if negative_result.best else 0.0
        )
        negative_semantic_accepts += bool(negative_result.semantics.get("accepted"))

    return {
        "generator": "polygon-mosaic-v1",
        "matched_histogram": True,
        "source_group_split": "one seed is one source_group_id; no train/test use",
        "cases_per_class": count,
        "structured": {
            "median_top5_structure_iou": float(np.median(recoveries)),
            "minimum_top5_structure_iou": float(np.min(recoveries)),
            "structure_iou_at_least_0_25": float(np.mean(np.asarray(recoveries) >= 0.25)),
        },
        "no_structure": {
            "abstention_rate": negative_statuses.count("uncertain") / count,
            "maximum_decision_confidence": float(np.max(negative_confidences)),
            "semantic_accept_count": negative_semantic_accepts,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=4)
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
            structured["median_top5_structure_iou"] >= 0.28
            and structured["structure_iou_at_least_0_25"] >= 0.75
            and no_structure["abstention_rate"] == 1.0
            and no_structure["maximum_decision_confidence"] < 0.74
            and no_structure["semantic_accept_count"] == 0
        )
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
