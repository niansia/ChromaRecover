"""Evaluate mask-first digit semantics on external ColorBlindnessEval images."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from chromarecover import recover


def _read_rows(path: Path) -> list[dict[str, object]]:
    try:
        from pyarrow import parquet
    except ImportError as error:  # pragma: no cover - optional external benchmark dependency
        raise SystemExit("Install pyarrow to run the external parquet benchmark") from error
    return parquet.read_table(path).to_pylist()


def _decode_image(value: object) -> np.ndarray:
    if isinstance(value, dict) and value.get("bytes") is not None:
        return np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))
    if isinstance(value, dict) and value.get("path"):
        return np.asarray(Image.open(value["path"]).convert("RGB"))
    raise ValueError("Unsupported parquet image representation")


def evaluate(path: Path, count: int, offset: int) -> dict[str, object]:
    rows = _read_rows(path)
    selected = rows[offset : offset + count]
    if not selected:
        raise ValueError("The requested dataset slice is empty")
    records: list[dict[str, object]] = []
    accepted_count = 0
    accepted_correct = 0
    candidate_count = 0
    for row in selected:
        expected = str(row["number"])
        result = recover(_decode_image(row["image"]), semantics="auto", top_k=3)
        hypothesis = result.semantics.get("hypothesis")
        accepted = bool(result.semantics.get("accepted"))
        correct = accepted and hypothesis == expected
        accepted_count += int(accepted)
        accepted_correct += int(correct)
        candidate_count += int(result.best is not None)
        records.append(
            {
                "filename": row["filename"],
                "source_group_id": f"{row['font']}:{row['color_config']}",
                "expected": expected,
                "status": result.status,
                "accepted": accepted,
                "hypothesis": hypothesis,
                "correct": correct,
            }
        )
    total = len(selected)
    source_groups = sorted({record["source_group_id"] for record in records})
    return {
        "dataset": "MM-Hallu/ColorBlindnessEval",
        "license_metadata": "Apache-2.0",
        "source_url": "https://huggingface.co/datasets/MM-Hallu/ColorBlindnessEval",
        "purpose": "external digit-semantics evaluation; no mask ground truth",
        "slice": {"offset": offset, "count": total},
        "source_groups": source_groups,
        "candidate_rate": candidate_count / total,
        "acceptance_rate": accepted_count / total,
        "exact_accuracy_including_abstention": accepted_correct / total,
        "conditional_accuracy_when_accepted": (
            accepted_correct / accepted_count if accepted_count else 1.0
        ),
        "false_acceptance_rate": (accepted_count - accepted_correct) / total,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--assert-thresholds", action="store_true")
    arguments = parser.parse_args()
    if arguments.count < 1 or arguments.offset < 0:
        parser.error("count must be positive and offset cannot be negative")
    report = evaluate(arguments.dataset, arguments.count, arguments.offset)
    print(json.dumps(report, indent=2))
    if arguments.assert_thresholds:
        passed = (
            float(report["false_acceptance_rate"]) <= 0.02
            and float(report["acceptance_rate"]) >= 0.30
            and float(report["conditional_accuracy_when_accepted"]) >= 0.98
        )
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
