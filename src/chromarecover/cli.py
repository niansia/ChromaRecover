"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from ._version import __version__
from .api import recover
from .burst import recover_burst
from .exceptions import ChromaRecoverError


def _parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        coordinates = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1") from exc
    if len(coordinates) != 4:
        raise argparse.ArgumentTypeError("ROI must contain four comma-separated integers")
    return coordinates


def _parse_corners(value: str) -> list[list[float]]:
    try:
        corners = [
            [float(coordinate.strip()) for coordinate in point.split(",")]
            for point in value.split(";")
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("corners must be x,y;x,y;x,y;x,y") from exc
    if len(corners) != 4 or any(len(point) != 2 for point in corners):
        raise argparse.ArgumentTypeError("corners must contain four x,y points")
    if not all(math.isfinite(coordinate) for point in corners for coordinate in point):
        raise argparse.ArgumentTypeError("corners must contain only finite coordinates")
    if len({tuple(point) for point in corners}) != 4:
        raise argparse.ArgumentTypeError("corners must contain four distinct points")
    return corners


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chromarecover",
        description="Recover spatial structure carried by subtle color differences.",
    )
    parser.add_argument("image", nargs="?", help="PNG, JPEG or WebP input")
    parser.add_argument(
        "--burst",
        nargs="+",
        metavar="FRAME",
        help="fuse 2-12 photographs before recovery",
    )
    parser.add_argument("--out", "-o", type=Path, default=Path("chromarecover-result"))
    parser.add_argument(
        "--mode",
        choices=("auto", "digital", "camera", "camera-paper", "camera-screen"),
        default="digital",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--semantics",
        choices=("none", "auto"),
        default="none",
        help="optionally recognize an evidence-gated digit sequence after recovery",
    )
    parser.add_argument("--roi", type=_parse_roi, help="camera crop: x0,y0,x1,y1")
    parser.add_argument(
        "--corners",
        type=_parse_corners,
        help='camera document corners: "x,y;x,y;x,y;x,y"',
    )
    parser.add_argument("--no-auto-rectify", action="store_true")
    parser.add_argument("--no-auto-dewarp", action="store_true")
    parser.add_argument("--debug", action="store_true", help="save intermediate evidence maps")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.image) == bool(args.burst):
        parser.error("provide either one input image or --burst FRAME FRAME [...]")
    if args.burst and len(args.burst) < 2:
        parser.error("--burst requires at least two frames")
    try:
        options = {
            "mode": args.mode,
            "top_k": args.top_k,
            "semantics": args.semantics,
            "return_debug": args.debug,
            "roi": args.roi,
            "document_corners": args.corners,
            "auto_rectify": not args.no_auto_rectify,
            "auto_dewarp": not args.no_auto_dewarp,
        }
        if args.burst:
            result = recover_burst(args.burst, **options)
        else:
            result = recover(args.image, **options)
        result_path = result.save(args.out)
    except (ChromaRecoverError, ValueError) as exc:
        print(f"chromarecover: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": result.status,
                "structure_confidence": (
                    result.best.structure_confidence if result.best else None
                ),
                "capture_confidence": result.best.capture_confidence if result.best else None,
                "decision_confidence": result.best.decision_confidence if result.best else None,
                "semantics": result.semantics,
                "best_confidence": result.best.overall_confidence if result.best else None,
                "result": str(result_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
