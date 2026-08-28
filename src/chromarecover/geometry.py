"""Camera/digital geometry normalization with explicit inverse mapping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from .exceptions import InvalidInputError


@dataclass
class GeometryResult:
    rgb: np.ndarray
    forward_matrix: np.ndarray
    inverse_matrix: np.ndarray
    perspective_score: float = 0.0
    steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    sampling_map_x: np.ndarray | None = field(default=None, repr=False)
    sampling_map_y: np.ndarray | None = field(default=None, repr=False)


def _order_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    # Angle order is clockwise in image coordinates. Rotate it to start at the visual
    # top-left; unlike the classic sum/difference heuristic this does not duplicate a point
    # for symmetric diamond-shaped quadrilaterals.
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0).astype(np.float32, copy=False)


def _validate_document_corners(
    values: Sequence[Sequence[float]], image_width: int, image_height: int
) -> np.ndarray:
    try:
        points = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError("document_corners must contain four numeric x,y points") from exc
    if points.shape != (4, 2):
        raise InvalidInputError("document_corners must have shape (4, 2)")
    if not np.all(np.isfinite(points)):
        raise InvalidInputError("document_corners must contain only finite coordinates")

    margin = max(2.0, 0.10 * max(image_width, image_height))
    if (
        np.any(points[:, 0] < -margin)
        or np.any(points[:, 0] > image_width - 1 + margin)
        or np.any(points[:, 1] < -margin)
        or np.any(points[:, 1] > image_height - 1 + margin)
    ):
        raise InvalidInputError(
            "document_corners must stay within the image or its 10% coordinate margin"
        )

    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    pairwise[np.diag_indices(4)] = np.inf
    minimum_separation = max(2.0, 0.01 * min(image_width, image_height))
    if float(pairwise.min()) < minimum_separation:
        raise InvalidInputError("document_corners must be four distinct, separated points")

    ordered = _order_corners(points)
    contour = ordered.reshape(-1, 1, 2)
    area = abs(float(cv2.contourArea(contour)))
    minimum_area = max(16.0, image_width * image_height * 0.0001)
    if not cv2.isContourConvex(contour) or area < minimum_area:
        raise InvalidInputError("document_corners must form a non-degenerate convex quadrilateral")

    quad_width, quad_height = _quad_dimensions(ordered)
    if min(quad_width, quad_height) < minimum_separation:
        raise InvalidInputError("document_corners produce an unusably small quadrilateral")
    return ordered


def _quad_dimensions(corners: np.ndarray) -> tuple[float, float]:
    top_left, top_right, bottom_right, bottom_left = corners
    width = max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left),
    )
    height = max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right),
    )
    return float(width), float(height)


def _perspective_score(corners: np.ndarray) -> float:
    top_left, top_right, bottom_right, bottom_left = corners
    top = np.linalg.norm(top_right - top_left)
    bottom = np.linalg.norm(bottom_right - bottom_left)
    left = np.linalg.norm(bottom_left - top_left)
    right = np.linalg.norm(bottom_right - top_right)
    width_skew = abs(top - bottom) / max(top, bottom, 1e-6)
    height_skew = abs(left - right) / max(left, right, 1e-6)
    return float(np.clip(max(width_skew, height_skew) * 2.0, 0.0, 1.0))


def detect_document_quad(rgb: np.ndarray) -> np.ndarray | None:
    """Find a confident large planar quadrilateral, otherwise abstain."""

    height, width = rgb.shape[:2]
    scale = min(1.0, 1000.0 / max(height, width))
    preview = (
        cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else rgb
    )
    gray = cv2.cvtColor(preview, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    preview_area = float(preview.shape[0] * preview.shape[1])

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        area_ratio = cv2.contourArea(contour) / max(preview_area, 1.0)
        if area_ratio < 0.18:
            break
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        corners = _order_corners(polygon.reshape(4, 2) / scale)
        quad_width, quad_height = _quad_dimensions(corners)
        if min(quad_width, quad_height) < 96:
            continue
        # A quad equal to the image boundary provides no useful rectification evidence.
        if area_ratio > 0.96:
            continue
        return corners
    return None


def detect_content_quad(rgb: np.ndarray) -> np.ndarray | None:
    """Fallback for borderless media using contrast from the outer-frame background."""

    height, width = rgb.shape[:2]
    scale = min(1.0, 900.0 / max(height, width))
    preview = (
        cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else rgb
    )
    lab = cv2.cvtColor(preview.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    border_width = max(3, round(min(preview.shape[:2]) * 0.025))
    border = np.concatenate(
        [
            lab[:border_width].reshape(-1, 3),
            lab[-border_width:].reshape(-1, 3),
            lab[:, :border_width].reshape(-1, 3),
            lab[:, -border_width:].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    image_area = float(preview.shape[0] * preview.shape[1])
    minimum_side = min(preview.shape[:2])
    candidates: list[tuple[float, np.ndarray]] = []

    # A fixed multiple of border variation fails when one border contains a reflection or
    # desk object. Quantiles of the full distance field instead propose several bounded
    # foreground areas; geometry then decides whether any proposal is document-like.
    for quantile in (82.0, 86.0, 90.0, 92.0):
        threshold = float(np.percentile(distance, quantile))
        if threshold < 4.0:
            continue
        # Inclusive comparison matters for low-noise phone captures where a nearly uniform
        # page creates a large plateau exactly at the selected percentile.
        content = (distance >= threshold).astype(np.uint8)
        for kernel_ratio in (0.012, 0.020, 0.030):
            kernel_size = max(5, round(minimum_side * kernel_ratio) | 1)
            connected = cv2.morphologyEx(
                content,
                cv2.MORPH_CLOSE,
                np.ones((kernel_size, kernel_size), np.uint8),
            )
            connected = cv2.morphologyEx(
                connected, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
            )
            contours, _hierarchy = cv2.findContours(
                connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
                contour_area = cv2.contourArea(contour)
                area_ratio = contour_area / max(image_area, 1.0)
                if not 0.04 <= area_ratio <= 0.94:
                    continue
                box = cv2.boxPoints(cv2.minAreaRect(contour))
                box_area = cv2.contourArea(box)
                rectangularity = contour_area / max(box_area, 1.0)
                if rectangularity < 0.62:
                    continue
                edge_margin = min(
                    float(box[:, 0].min()),
                    float(box[:, 1].min()),
                    float(preview.shape[1] - 1 - box[:, 0].max()),
                    float(preview.shape[0] - 1 - box[:, 1].max()),
                )
                border_safety = float(
                    np.clip(edge_margin / max(minimum_side * 0.03, 1.0), 0.0, 1.0)
                )
                score = (
                    0.60 * rectangularity
                    + 0.25 * np.sqrt(area_ratio)
                    + 0.15 * border_safety
                )
                candidates.append((float(score), box))

    for score, box in sorted(candidates, key=lambda item: -item[0]):
        if score < 0.62:
            break
        corners = _order_corners(box / scale)
        quad_width, quad_height = _quad_dimensions(corners)
        if min(quad_width, quad_height) >= 96:
            return corners
    return None


def _detect_curved_page_maps(
    rgb: np.ndarray, max_side: int
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Estimate an axis-aligned page ribbon from observable curved top/bottom edges."""

    height, width = rgb.shape[:2]
    scale = min(1.0, 900.0 / max(height, width))
    preview = (
        cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else rgb
    )
    lab = cv2.cvtColor(preview.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    border_width = max(3, round(min(preview.shape[:2]) * 0.025))
    border = np.concatenate(
        [
            lab[:border_width].reshape(-1, 3),
            lab[-border_width:].reshape(-1, 3),
            lab[:, :border_width].reshape(-1, 3),
            lab[:, -border_width:].reshape(-1, 3),
        ]
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    threshold = max(6.0, float(np.percentile(np.linalg.norm(border - background, axis=1), 95)) * 2.8)
    page = (distance > threshold).astype(np.uint8)
    close_size = max(9, round(min(preview.shape[:2]) * 0.055) | 1)
    page = cv2.morphologyEx(
        page, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8)
    )
    contours, _hierarchy = cv2.findContours(page, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < preview.shape[0] * preview.shape[1] * 0.18:
        return None
    rectangle = cv2.boxPoints(cv2.minAreaRect(contour))
    rectangularity = cv2.contourArea(contour) / max(cv2.contourArea(rectangle), 1.0)
    if rectangularity >= 0.91:
        return None
    filled = np.zeros(page.shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 1, -1)
    columns = np.flatnonzero(filled.any(axis=0))
    if len(columns) < preview.shape[1] * 0.45:
        return None
    x0, x1 = int(columns[0]), int(columns[-1])
    x_values = np.arange(x0, x1 + 1)
    top = np.empty(len(x_values), dtype=np.float32)
    bottom = np.empty(len(x_values), dtype=np.float32)
    for index, x in enumerate(x_values):
        rows = np.flatnonzero(filled[:, x])
        if len(rows):
            top[index], bottom[index] = rows[0], rows[-1]
        elif index:
            top[index], bottom[index] = top[index - 1], bottom[index - 1]
        else:
            return None
    sigma = max(2.0, len(x_values) * 0.018)
    top = cv2.GaussianBlur(top.reshape(1, -1), (0, 0), sigma).ravel()
    bottom = cv2.GaussianBlur(bottom.reshape(1, -1), (0, 0), sigma).ravel()
    design = np.stack([np.ones(len(x_values)), x_values], axis=1)
    top_line = design @ np.linalg.lstsq(design, top, rcond=None)[0]
    bottom_line = design @ np.linalg.lstsq(design, bottom, rcond=None)[0]
    page_height = max(float(np.median(bottom - top)), 1.0)
    curvature = float(
        max(np.std(top - top_line), np.std(bottom - bottom_line)) / page_height
    )
    if not 0.008 <= curvature <= 0.28:
        return None

    source_width = (x1 - x0 + 1) / scale
    source_height = page_height / scale
    output_scale = min(1.0, max_side / max(source_width, source_height))
    output_width = max(2, round(source_width * output_scale))
    output_height = max(2, round(source_height * output_scale))
    destination_x = np.linspace(x0, x1, output_width, dtype=np.float32)
    top_resampled = np.interp(destination_x, x_values, top) / scale
    bottom_resampled = np.interp(destination_x, x_values, bottom) / scale
    source_x = destination_x / scale
    vertical = np.linspace(0.0, 1.0, output_height, dtype=np.float32)[:, None]
    map_x = np.broadcast_to(source_x[None, :], (output_height, output_width)).copy()
    map_y = top_resampled[None, :] + vertical * (
        bottom_resampled - top_resampled
    )[None, :]
    return map_x.astype(np.float32), map_y.astype(np.float32), curvature


def normalize_geometry(
    rgb: np.ndarray,
    *,
    mode: str,
    max_side: int,
    roi: tuple[int, int, int, int] | None = None,
    document_corners: Sequence[Sequence[float]] | None = None,
    auto_rectify: bool = True,
    auto_dewarp: bool = True,
) -> GeometryResult:
    height, width = rgb.shape[:2]
    if roi is not None and document_corners is not None:
        raise ValueError("Pass either roi or document_corners, not both")
    warnings: list[str] = []
    steps: list[str] = []
    corners: np.ndarray | None = None
    source = "full_image"

    if document_corners is not None:
        corners = _validate_document_corners(document_corners, width, height)
        source = "user_corners"
    elif roi is not None:
        x0, y0, x1, y1 = roi
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise InvalidInputError(
                "roi must satisfy 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height"
            )
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise InvalidInputError("roi must be at least 2 by 2 pixels")
        corners = np.float32([[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]])
        source = "user_roi"
    if corners is None and mode == "camera" and auto_rectify:
        dense_maps = _detect_curved_page_maps(rgb, max_side) if auto_dewarp else None
        if dense_maps is not None:
            map_x, map_y, curvature = dense_maps
            normalized = cv2.remap(
                rgb,
                map_x,
                map_y,
                interpolation=cv2.INTER_AREA,
                borderMode=cv2.BORDER_REPLICATE,
            )
            steps.append("geometry_dense_ribbon_dewarp")
            return GeometryResult(
                rgb=normalized,
                forward_matrix=np.eye(3, dtype=np.float64),
                inverse_matrix=np.eye(3, dtype=np.float64),
                perspective_score=curvature,
                steps=steps,
                warnings=warnings,
                metadata={
                    "source": "auto_dense_ribbon",
                    "analysis_width": int(normalized.shape[1]),
                    "analysis_height": int(normalized.shape[0]),
                    "forward_matrix": None,
                    "inverse_matrix": None,
                    "corners": None,
                    "dense_mapping": True,
                    "curvature_score": curvature,
                },
                sampling_map_x=map_x,
                sampling_map_y=map_y,
            )
        corners = detect_document_quad(rgb)
        if corners is not None:
            source = "auto_quad"
        else:
            corners = detect_content_quad(rgb)
            if corners is not None:
                source = "auto_content_quad"
            else:
                warnings.append("perspective_not_rectified")

    if corners is None:
        scale = min(1.0, max_side / max(width, height))
        output_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        forward = np.array(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        normalized = cv2.resize(
            rgb,
            output_size,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        if scale < 1.0:
            steps.append(f"analysis_resize:{output_size[0]}x{output_size[1]}")
        score = 0.0
    else:
        rect_width, rect_height = _quad_dimensions(corners)
        scale = min(1.0, max_side / max(rect_width, rect_height))
        output_size = (
            max(1, round(rect_width * scale)),
            max(1, round(rect_height * scale)),
        )
        destination = np.float32(
            [
                [0, 0],
                [output_size[0] - 1, 0],
                [output_size[0] - 1, output_size[1] - 1],
                [0, output_size[1] - 1],
            ]
        )
        forward = cv2.getPerspectiveTransform(corners.astype(np.float32), destination).astype(
            np.float64
        )
        determinant = float(np.linalg.det(forward))
        condition = float(np.linalg.cond(forward))
        if (
            not np.all(np.isfinite(forward))
            or not np.isfinite(condition)
            or abs(determinant) < 1e-12
            or condition > 1e12
        ):
            raise InvalidInputError("document_corners produce a singular perspective transform")
        normalized = cv2.warpPerspective(
            rgb,
            forward,
            output_size,
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_REPLICATE,
        )
        score = _perspective_score(corners)
        steps.append(f"geometry_rectify:{source}")

    try:
        inverse = np.linalg.inv(forward)
    except np.linalg.LinAlgError as exc:
        raise InvalidInputError("geometry transform is singular") from exc
    return GeometryResult(
        rgb=normalized,
        forward_matrix=forward,
        inverse_matrix=inverse,
        perspective_score=score,
        steps=steps,
        warnings=warnings,
        metadata={
            "source": source,
            "analysis_width": int(normalized.shape[1]),
            "analysis_height": int(normalized.shape[0]),
            "forward_matrix": forward.tolist(),
            "inverse_matrix": inverse.tolist(),
            "corners": corners.tolist() if corners is not None else None,
        },
    )


def inverse_map_mask(mask: np.ndarray, result: GeometryResult, output_shape: tuple[int, int]) -> np.ndarray:
    output_height, output_width = output_shape
    if result.sampling_map_x is not None and result.sampling_map_y is not None:
        resized = cv2.resize(
            mask.astype(np.uint8),
            (result.sampling_map_x.shape[1], result.sampling_map_x.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        x = np.clip(np.rint(result.sampling_map_x[resized]).astype(int), 0, output_width - 1)
        y = np.clip(np.rint(result.sampling_map_y[resized]).astype(int), 0, output_height - 1)
        output = np.zeros((output_height, output_width), dtype=np.uint8)
        output[y, x] = 1
        return cv2.dilate(output, np.ones((3, 3), np.uint8)).astype(bool)
    return cv2.warpPerspective(
        mask.astype(np.uint8),
        result.inverse_matrix,
        (output_width, output_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def inverse_map_evidence(
    evidence: np.ndarray,
    result: GeometryResult,
    output_shape: tuple[int, int],
    *,
    coordinate_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Map continuous evidence into an original or scaled-original coordinate space.

    ``coordinate_shape`` is the full-resolution coordinate system used by the stored inverse
    geometry. Passing a smaller ``output_shape`` then produces a bounded presentation map
    directly, without first allocating a full-resolution float32 field.
    """

    output_height, output_width = output_shape
    coordinate_height, coordinate_width = coordinate_shape or output_shape
    scale_x = (output_width - 1) / max(coordinate_width - 1, 1)
    scale_y = (output_height - 1) / max(coordinate_height - 1, 1)
    if result.sampling_map_x is not None and result.sampling_map_y is not None:
        resized = cv2.resize(
            np.asarray(evidence, dtype=np.float32),
            (result.sampling_map_x.shape[1], result.sampling_map_x.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        x = np.clip(
            np.rint(result.sampling_map_x * scale_x).astype(int), 0, output_width - 1
        )
        y = np.clip(
            np.rint(result.sampling_map_y * scale_y).astype(int), 0, output_height - 1
        )
        output = np.zeros((output_height, output_width), dtype=np.float32)
        np.maximum.at(output.ravel(), (y * output_width + x).ravel(), resized.ravel())
        map_height, map_width = result.sampling_map_x.shape
        fill_radius = max(
            1,
            round(
                0.55
                * max(output_width / max(map_width, 1), output_height / max(map_height, 1))
            ),
        )
        kernel_size = min(31, fill_radius * 2 + 1)
        return cv2.dilate(
            output, np.ones((kernel_size, kernel_size), np.uint8)
        ).astype(np.float32)
    presentation_scale = np.array(
        [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return cv2.warpPerspective(
        np.asarray(evidence, dtype=np.float32),
        presentation_scale @ result.inverse_matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)
