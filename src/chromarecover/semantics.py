"""Optional, evidence-gated digit recognition after mask recovery.

This module never creates or edits recovery masks. It reads a candidate's continuous
structure evidence and reports a bounded shape hypothesis with explicit alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DigitMatch:
    label: str
    confidence: float
    margin: float
    bbox: tuple[int, int, int, int]
    alternatives: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 6),
            "margin": round(float(self.margin), 6),
            "bbox_xyxy": list(self.bbox),
            "alternatives": [
                {"label": label, "score": round(float(score), 6)}
                for label, score in self.alternatives
            ],
        }


@dataclass(frozen=True)
class DigitSequence:
    hypothesis: str | None
    confidence: float
    accepted: bool
    characters: tuple[DigitMatch, ...]
    method: str = "multifont_topology_chamfer_v1"

    def to_dict(self) -> dict[str, Any]:
        # Public ``hypothesis`` means an accepted semantic result. Low-confidence template
        # labels remain inspectable as character alternatives but are not promoted into the
        # same field as a usable answer.
        return {
            "kind": "digit_sequence",
            "hypothesis": self.hypothesis if self.accepted else None,
            "confidence": round(float(self.confidence), 6),
            "accepted": self.accepted,
            "reason": "accepted_shape_match" if self.accepted else "below_acceptance_threshold",
            "method": self.method,
            "characters": [character.to_dict() for character in self.characters],
        }


def _active_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], values.astype(bool), [False]))
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(left), int(right)) for left, right in zip(changes[::2], changes[1::2])]


def _split_wide_runs(
    runs: list[tuple[int, int]], projection: np.ndarray, glyph_height: int
) -> list[tuple[int, int]]:
    """Split touching glyphs at local projection valleys using only coarse aspect priors."""

    output: list[tuple[int, int]] = []
    expected_width = max(4.0, glyph_height * 0.72)
    for left, right in runs:
        width = right - left
        part_count = int(np.clip(round(width / expected_width), 1, 8))
        if part_count == 1 or width < glyph_height * 1.08:
            output.append((left, right))
            continue
        boundaries = [left]
        part_width = width / part_count
        for index in range(1, part_count):
            expected = left + index * part_width
            radius = max(2, round(part_width * 0.22))
            search_left = max(boundaries[-1] + 2, round(expected) - radius)
            search_right = min(right - 2, round(expected) + radius + 1)
            if search_right <= search_left:
                boundary = round(expected)
            else:
                boundary = search_left + int(np.argmin(projection[search_left:search_right]))
            boundaries.append(boundary)
        boundaries.append(right)
        output.extend(
            (start, end) for start, end in pairwise(boundaries) if end > start
        )
    return output


def _normalize_glyph(values: np.ndarray, height: int = 64, width: int = 48) -> np.ndarray:
    glyph = np.asarray(values, dtype=np.float32)
    if glyph.size == 0 or float(glyph.max()) <= 1e-6:
        return np.zeros((height, width), dtype=np.float32)
    glyph = glyph / float(glyph.max())
    active = glyph >= max(0.10, float(np.percentile(glyph[glyph > 0], 35)))
    ys, xs = np.nonzero(active)
    if len(xs) < 4:
        return np.zeros((height, width), dtype=np.float32)
    crop = glyph[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    scale = min((height - 8) / crop.shape[0], (width - 8) / crop.shape[1])
    resized_width = max(1, round(crop.shape[1] * scale))
    resized_height = max(1, round(crop.shape[0] * scale))
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width), dtype=np.float32)
    y0 = (height - resized_height) // 2
    x0 = (width - resized_width) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return canvas


@lru_cache(maxsize=1)
def _digit_templates() -> dict[str, tuple[np.ndarray, ...]]:
    fonts = (
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    )
    bank: dict[str, list[np.ndarray]] = {str(digit): [] for digit in range(10)}
    for digit, digit_templates in bank.items():
        for font in fonts:
            for thickness in (2, 3, 4, 5):
                rendered = np.zeros((96, 72), dtype=np.uint8)
                scale = 2.55
                (text_width, text_height), _baseline = cv2.getTextSize(
                    digit, font, scale, thickness
                )
                origin = ((72 - text_width) // 2, (96 + text_height) // 2)
                cv2.putText(
                    rendered,
                    digit,
                    origin,
                    font,
                    scale,
                    255,
                    thickness,
                    cv2.LINE_AA,
                )
                normalized = _normalize_glyph(rendered.astype(np.float32) / 255.0)
                digit_templates.append(normalized)
    return {digit: tuple(templates) for digit, templates in bank.items()}


def _hole_count(binary: np.ndarray) -> int:
    contours, hierarchy = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return 0
    minimum_area = binary.size * 0.008
    return sum(
        parent >= 0 and cv2.contourArea(contours[index]) >= minimum_area
        for index, (_next, _previous, _child, parent) in enumerate(hierarchy[0])
    )


def _shape_score(observed: np.ndarray, template: np.ndarray) -> float:
    observed_soft = cv2.GaussianBlur(observed, (0, 0), 1.2)
    template_soft = cv2.GaussianBlur(template, (0, 0), 1.2)
    observed_soft /= float(observed_soft.max()) + 1e-6
    template_soft /= float(template_soft.max()) + 1e-6

    observed_zero = observed_soft - observed_soft.mean()
    template_zero = template_soft - template_soft.mean()
    correlation = float(
        np.sum(observed_zero * template_zero)
        / (np.linalg.norm(observed_zero) * np.linalg.norm(template_zero) + 1e-6)
    )
    correlation = 0.5 + 0.5 * correlation

    observed_binary = observed_soft >= 0.22
    template_binary = template_soft >= 0.22
    template_distance = cv2.distanceTransform(
        (~template_binary).astype(np.uint8), cv2.DIST_L2, 3
    )
    observed_distance = cv2.distanceTransform(
        (~observed_binary).astype(np.uint8), cv2.DIST_L2, 3
    )
    forward = float(template_distance[observed_binary].mean()) if observed_binary.any() else 64.0
    backward = float(observed_distance[template_binary].mean()) if template_binary.any() else 64.0
    chamfer = float(np.exp(-7.0 * (forward + backward) / 128.0))
    soft_iou = float(
        np.minimum(observed_soft, template_soft).sum()
        / (np.maximum(observed_soft, template_soft).sum() + 1e-6)
    )
    # A higher level preserves holes that soft stroke halos may merge. Topology is useful
    # for 0/6/8/9 but remains only one bounded term because sparse mosaic samples can break
    # otherwise valid loops.
    observed_holes = _hole_count(observed_soft >= 0.42)
    template_holes = _hole_count(template_soft >= 0.42)
    hole_penalty = 0.10 * min(abs(observed_holes - template_holes), 2)
    return float(np.clip(0.50 * correlation + 0.30 * chamfer + 0.20 * soft_iou - hole_penalty, 0, 1))


def _match_digit(values: np.ndarray, bbox: tuple[int, int, int, int]) -> DigitMatch:
    observed = _normalize_glyph(values)
    label_scores: list[tuple[str, float]] = []
    for label, templates in _digit_templates().items():
        label_scores.append((label, max(_shape_score(observed, template) for template in templates)))
    label_scores.sort(key=lambda item: item[1], reverse=True)
    best_label, best_score = label_scores[0]
    margin = best_score - label_scores[1][1]
    absolute = np.clip((best_score - 0.48) / 0.30, 0.0, 1.0)
    distinctiveness = np.clip(margin / 0.075, 0.0, 1.0)
    confidence = float(absolute * (0.45 + 0.55 * distinctiveness))
    return DigitMatch(
        label=best_label,
        confidence=confidence,
        margin=float(margin),
        bbox=bbox,
        alternatives=tuple(label_scores[:3]),
    )


def _has_unresolved_six_eight(characters: list[DigitMatch]) -> bool:
    return any(
        character.label == "8"
        and len(character.alternatives) > 1
        and character.alternatives[1][0] in {"6", "9"}
        and character.margin < 0.075
        for character in characters
    )


def recognize_digit_sequence(evidence_map: np.ndarray) -> DigitSequence:
    """Recognize a horizontal sequence of 1-8 digits from continuous structure evidence."""

    evidence = np.clip(np.asarray(evidence_map, dtype=np.float32), 0.0, 1.0)
    if evidence.ndim != 2 or min(evidence.shape) < 20 or float(evidence.max()) < 0.12:
        return DigitSequence(None, 0.0, False, ())
    evidence = cv2.GaussianBlur(evidence, (0, 0), max(0.7, min(evidence.shape) * 0.003))
    evidence /= float(evidence.max()) + 1e-6

    row_projection = evidence.sum(axis=1)
    active_rows = row_projection >= 0.10 * float(row_projection.max())
    row_runs = _active_runs(active_rows)
    if not row_runs:
        return DigitSequence(None, 0.0, False, ())
    y0, y1 = max(row_runs, key=lambda run: (run[1] - run[0], row_projection[slice(*run)].sum()))
    if y1 - y0 < max(8, round(evidence.shape[0] * 0.12)):
        return DigitSequence(None, 0.0, False, ())

    row = evidence[y0:y1]
    column_projection = row.sum(axis=0)
    column_projection /= float(column_projection.max()) + 1e-6
    # Gaussian evidence has long non-zero tails. A 15% projection cut separates adjacent
    # glyphs while remaining permissive for narrow "1" strokes and sparse mosaic cells.
    active_columns = (column_projection >= 0.15).astype(np.uint8)[None, ...]
    close_width = max(1, round(evidence.shape[1] * 0.004))
    active_columns = cv2.morphologyEx(
        active_columns, cv2.MORPH_CLOSE, np.ones((1, close_width), np.uint8)
    )[0].astype(bool)
    minimum_width = max(4, round(evidence.shape[1] * 0.025))
    runs = [run for run in _active_runs(active_columns) if run[1] - run[0] >= minimum_width]
    runs = _split_wide_runs(runs, column_projection, y1 - y0)
    if not 1 <= len(runs) <= 8:
        return DigitSequence(None, 0.0, False, ())

    characters: list[DigitMatch] = []
    for x0, x1 in runs:
        padding = max(1, round((x1 - x0) * 0.06))
        left = max(0, x0 - padding)
        right = min(evidence.shape[1], x1 + padding)
        characters.append(_match_digit(evidence[y0:y1, left:right], (left, y0, right, y1)))

    hypothesis = "".join(character.label for character in characters)
    confidence = float(np.mean([character.confidence for character in characters]))
    confidence *= float(np.min([0.65 + 0.35 * character.confidence for character in characters]))
    unresolved_six_eight = _has_unresolved_six_eight(characters)
    accepted = (
        confidence >= 0.52
        and all(character.confidence >= 0.30 for character in characters)
        and not unresolved_six_eight
    )
    return DigitSequence(hypothesis, confidence, accepted, tuple(characters))


def semantic_consensus(candidate_semantics: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [item for item in candidate_semantics if item.get("accepted")]
    if not accepted:
        return {
            "mode": "auto",
            "kind": "digit_sequence",
            "hypothesis": None,
            "confidence": 0.0,
            "accepted": False,
            "reason": "no_confident_shape_match",
        }
    weights: dict[str, float] = {}
    for item in accepted:
        hypothesis = str(item["hypothesis"])
        weights[hypothesis] = weights.get(hypothesis, 0.0) + float(item["confidence"])
    winner, winner_weight = max(weights.items(), key=lambda item: item[1])
    total = sum(weights.values())
    support = sum(item["hypothesis"] == winner for item in accepted)
    winner_items = [item for item in accepted if item["hypothesis"] == winner]
    agreement = winner_weight / max(total, 1e-6)
    mean_confidence = winner_weight / support
    consensus_confidence = float(mean_confidence * agreement * min(1.0, 0.55 + 0.30 * support))
    decision_visual_gate = max(
        float(item.get("candidate_decision_confidence", 0.0)) for item in accepted
    ) >= 0.74
    source_families = {
        str(item.get("candidate_source_family", "unknown")) for item in winner_items
    }
    maximum_capture_confidence = max(
        float(item.get("candidate_capture_confidence", 0.0)) for item in winner_items
    )
    cross_family_minimum_margin = min(
        (
            float(character.get("margin", 0.0))
            for item in winner_items
            for character in item.get("characters", [])
        ),
        default=0.0,
    )
    top = candidate_semantics[0] if candidate_semantics else {}
    # Strong agreement between independently generated visual families can validate the
    # readable shape without changing the recovery mask, rank or uncertain status. This is
    # deliberately stricter than ordinary semantic consensus and still needs non-trivial
    # pre-status capture support.
    cross_family_visual_gate = (
        support >= 3
        and len(source_families) >= 2
        and agreement >= 0.85
        and consensus_confidence >= 0.55
        and maximum_capture_confidence >= 0.48
        and cross_family_minimum_margin >= 0.04
        and not bool(top.get("candidate_weak_structure_evidence", True))
        and not bool(top.get("candidate_quality_blocked", True))
    )
    visual_gate = decision_visual_gate or cross_family_visual_gate
    high_single = support == 1 and mean_confidence >= 0.82 and decision_visual_gate
    trusted_top = (
        top.get("accepted") is True
        and top.get("hypothesis") == winner
        and float(top.get("confidence", 0.0)) >= 0.70
        and float(top.get("candidate_decision_confidence", 0.0)) >= 0.74
    )
    is_accepted = visual_gate and (
        (support >= 2 and agreement >= 0.60 and consensus_confidence >= 0.48)
        or high_single
        or trusted_top
    )
    return {
        "mode": "auto",
        "kind": "digit_sequence",
        "hypothesis": winner if is_accepted else None,
        "confidence": round(consensus_confidence, 6),
        "accepted": is_accepted,
        "supporting_candidates": support,
        "supporting_source_families": len(source_families),
        "maximum_capture_confidence": round(maximum_capture_confidence, 6),
        "cross_family_minimum_margin": round(cross_family_minimum_margin, 6),
        "top_has_weak_structure_evidence": bool(
            top.get("candidate_weak_structure_evidence", True)
        ),
        "top_has_blocking_quality_warning": bool(
            top.get("candidate_quality_blocked", True)
        ),
        "candidate_agreement": round(float(agreement), 6),
        "reason": (
            "cross_family_shape_validation"
            if cross_family_visual_gate and not decision_visual_gate
            else "trusted_top_shape_match"
            if trusted_top and support < 2
            else "cross_candidate_shape_consensus"
            if is_accepted
            else "weak_visual_evidence"
            if not visual_gate
            else "semantic_ambiguity"
        ),
    }
