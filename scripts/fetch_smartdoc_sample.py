"""Download and safely extract the official CC BY 4.0 SmartDoc sample."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

ARCHIVE_URL = "https://zenodo.org/records/1230218/files/sampleDataset.tar.gz?download=1"
ARCHIVE_SHA256 = "90d1a64f476ffe290ebbddf4337e108d471b9c25420551223f447db972844a9a"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_members(archive: tarfile.TarFile, destination: Path) -> list[tarfile.TarInfo]:
    destination = destination.resolve()
    members: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if destination != target and destination not in target.parents:
            raise ValueError(f"Unsafe archive path: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"Unsupported archive member: {member.name}")
        members.append(member)
    return members


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("tmp/external-datasets/smartdoc-sample"),
    )
    arguments = parser.parse_args()
    destination = arguments.destination
    archive_path = destination / "sampleDataset.tar.gz"
    extracted_path = destination / "extracted"
    destination.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        print(f"Downloading {ARCHIVE_URL}")
        urllib.request.urlretrieve(ARCHIVE_URL, archive_path)
    checksum = _sha256(archive_path)
    if checksum != ARCHIVE_SHA256:
        raise SystemExit(
            f"Checksum mismatch for {archive_path}: expected {ARCHIVE_SHA256}, got {checksum}"
        )

    extracted_path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extracted_path, members=_safe_members(archive, extracted_path))
    print(extracted_path / "sampleDataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
