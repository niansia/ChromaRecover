"""Adversarial smooth-field and achromatic-pattern rejection gate."""

from __future__ import annotations

import argparse
import json

import numpy as np

from chromarecover import RecoverConfig, recover


def evaluate(count: int, size: int) -> dict[str, float | int]:
    generator = np.random.default_rng(42)
    yy, xx = np.mgrid[:size, :size]
    x_normalized = xx / (size - 1)
    y_normalized = yy / (size - 1)
    threshold = RecoverConfig().confidence_threshold
    ok_count = 0
    confident_count = 0
    semantic_accept_count = 0
    maximum_confidence = 0.0

    for _case_index in range(count):
        base = generator.uniform(90, 190, 3)
        horizontal = generator.uniform(-70, 70, 3)
        vertical = generator.uniform(-50, 50, 3)
        amplitude = float(generator.uniform(2, 15))
        period = int(generator.integers(2, 8))
        phase = int(generator.integers(0, period))
        texture = amplitude * (
            2 * (((xx + yy + phase) % period) < period / 2).astype(np.float32) - 1
        )
        image = np.clip(
            base
            + horizontal * x_normalized[..., None]
            + vertical * y_normalized[..., None]
            + texture[..., None],
            0,
            255,
        ).astype(np.uint8)
        result = recover(image, mode="digital", top_k=3, semantics="auto")
        confidence = result.best.overall_confidence if result.best else 0.0
        maximum_confidence = max(maximum_confidence, confidence)
        ok_count += result.status == "ok"
        confident_count += confidence >= threshold
        semantic_accept_count += bool(result.semantics.get("accepted"))

    return {
        "cases": count,
        "ok_count": ok_count,
        "confidence_at_least_threshold_count": confident_count,
        "semantic_accept_count": semantic_accept_count,
        "maximum_confidence": maximum_confidence,
        "production_confidence_threshold": threshold,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--assert-thresholds", action="store_true")
    arguments = parser.parse_args()
    report = evaluate(arguments.count, arguments.size)
    print(json.dumps(report, indent=2))
    if arguments.assert_thresholds:
        return (
            0
            if report["ok_count"] == 0
            and report["confidence_at_least_threshold_count"] == 0
            and report["semantic_accept_count"] == 0
            else 1
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
