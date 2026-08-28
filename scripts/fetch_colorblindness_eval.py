"""Download the external Apache-2.0 ColorBlindnessEval parquet shard."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

DATA_URL = (
    "https://huggingface.co/datasets/MM-Hallu/ColorBlindnessEval/resolve/main/"
    "data-00000-of-00001.parquet?download=true"
)
DATA_SHA256 = "96c26f2ea1e5d465b8029b103208d844a4c6841cb1ee74cafe81f23b15fba2ae"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("tmp/external-datasets/colorblindness-eval"),
    )
    arguments = parser.parse_args()
    destination = arguments.destination
    data_path = destination / "data-00000-of-00001.parquet"
    destination.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        print(f"Downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, data_path)
    checksum = _sha256(data_path)
    if checksum != DATA_SHA256:
        raise SystemExit(
            f"Checksum mismatch for {data_path}: expected {DATA_SHA256}, got {checksum}"
        )
    print(data_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
