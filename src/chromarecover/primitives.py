"""Contour primitives and lightweight spatial-graph grouping.

This stage deliberately proposes additional masks instead of replacing color hypotheses.
The ranker therefore remains able to abstain when graph grouping is not useful.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .hypotheses import MaskHypothesis


@dataclass(frozen=True)
class Primitive:
    label: int
    area: int
    x: int
    y: int
    width: int
    height: int
    center_x: float
    center_y: float
    fill: float


def extract_primitives(mask: np.ndarray) -> tuple[np.ndarray, list[Primitive]]:
    """Turn a pixel mask into connected contour primitives."""

    binary = mask.astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    image_area = mask.size
    minimum_area = max(3, round(image_area * 0.00004))
    primitives: list[Primitive] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < minimum_area or area > image_area * 0.70:
            continue
        primitives.append(
            Primitive(
                label=label,
                area=area,
                x=x,
                y=y,
                width=width,
                height=height,
                center_x=float(centroids[label, 0]),
                center_y=float(centroids[label, 1]),
                fill=float(area / max(width * height, 1)),
            )
        )
    return labels, primitives


def _component_groups(primitives: list[Primitive], shape: tuple[int, int]) -> list[list[int]]:
    """Build a graph from proximity and baseline compatibility."""

    height, width = shape
    adjacency: list[set[int]] = [set() for _ in primitives]
    for left_index, left in enumerate(primitives):
        for right_index in range(left_index + 1, len(primitives)):
            right = primitives[right_index]
            vertical_distance = abs(left.center_y - right.center_y)
            horizontal_distance = abs(left.center_x - right.center_x)
            vertical_limit = max(0.035 * height, 1.35 * max(left.height, right.height))
            horizontal_limit = max(0.06 * width, 3.0 * max(left.width, right.width))
            size_ratio = max(left.area, right.area) / max(min(left.area, right.area), 1)
            if (
                vertical_distance <= vertical_limit
                and horizontal_distance <= horizontal_limit
                and size_ratio <= 10.0
            ):
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

    groups: list[list[int]] = []
    unseen = set(range(len(primitives)))
    while unseen:
        seed = unseen.pop()
        group = [seed]
        stack = [seed]
        while stack:
            current = stack.pop()
            neighbors = adjacency[current] & unseen
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
            group.extend(neighbors)
        groups.append(group)
    return groups


def group_primitives(mask: np.ndarray) -> np.ndarray | None:
    """Return the most coherent multi-primitive graph, or abstain."""

    labels, primitives = extract_primitives(mask)
    if not 2 <= len(primitives) <= 96:
        return None
    groups = _component_groups(primitives, mask.shape)
    viable = [group for group in groups if 2 <= len(group) <= 24]
    if not viable:
        return None

    def group_score(group: list[int]) -> float:
        nodes = [primitives[index] for index in group]
        total_area = sum(node.area for node in nodes)
        y_values = np.array([node.center_y for node in nodes], dtype=np.float32)
        alignment = np.exp(-float(np.std(y_values)) / max(0.08 * mask.shape[0], 1.0))
        return float(total_area * (0.65 + 0.35 * alignment) * min(len(nodes), 8) ** 0.35)

    best = max(viable, key=group_score)
    selected_labels = [primitives[index].label for index in best]
    grouped = np.isin(labels, selected_labels)
    change = np.mean(grouped != mask)
    occupancy = float(grouped.mean())
    if change < 0.002 or not 0.01 <= occupancy <= 0.75:
        return None
    return grouped


def augment_with_primitive_graph(
    hypotheses: list[MaskHypothesis], *, limit: int = 16, attempt_limit: int = 32
) -> list[MaskHypothesis]:
    """Append bounded graph-grouped proposals from promising component masks."""

    family_queues: dict[str, list[MaskHypothesis]] = {}
    for hypothesis in hypotheses:
        family_queues.setdefault(hypothesis.family, []).append(hypothesis)
    # Round-robin keeps the bounded stage multi-family instead of spending the entire CPU
    # budget on whichever color representation happened to be generated first.
    routed: list[MaskHypothesis] = []
    depth = 0
    while len(routed) < min(attempt_limit, len(hypotheses)):
        added = False
        for queue in family_queues.values():
            if depth < len(queue):
                routed.append(queue[depth])
                added = True
                if len(routed) >= attempt_limit:
                    break
        if not added:
            break
        depth += 1

    additions: list[MaskHypothesis] = []
    for hypothesis in routed:
        if len(additions) >= limit:
            break
        grouped = group_primitives(hypothesis.mask)
        if grouped is None:
            continue
        additions.append(
            MaskHypothesis(
                mask=grouped,
                source=f"{hypothesis.source}:primitive_graph",
                family=f"{hypothesis.family}_primitive_graph",
            )
        )
    return hypotheses + additions
