"""Regenerate the deterministic README demo."""

from pathlib import Path

from PIL import Image

from chromarecover import recover
from chromarecover.synthetic import make_polygon_mosaic


def main() -> int:
    destination = Path("examples/digital")
    destination.mkdir(parents=True, exist_ok=True)
    case = make_polygon_mosaic(size=480, text="820", seed=42, structured=True)
    Image.fromarray(case.image).save(destination / "demo_input.png")
    Image.fromarray((case.mask.astype("uint8") * 255), mode="L").save(
        destination / "demo_ground_truth.png"
    )
    result = recover(case.image, mode="digital", top_k=3)
    result.save(destination / "demo_result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
