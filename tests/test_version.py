import re
from pathlib import Path

from chromarecover import __version__


def test_package_metadata_and_runtime_version_match() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)

    assert match is not None
    assert match.group(1) == __version__
