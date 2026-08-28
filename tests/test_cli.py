from pathlib import Path

import pytest
from PIL import Image

from chromarecover.cli import main
from chromarecover.synthetic import make_chromatic_pattern


def test_cli_end_to_end(tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    output = tmp_path / "result"
    Image.fromarray(make_chromatic_pattern(size=128).image).save(source)

    assert main([str(source), "--out", str(output), "--top-k", "2"]) == 0
    assert (output / "result.json").is_file()
    assert (output / "mask_01.png").is_file()


def test_cli_burst_and_debug(tmp_path: Path) -> None:
    case = make_chromatic_pattern(size=128, seed=11)
    sources: list[str] = []
    for index in range(3):
        source = tmp_path / f"frame-{index}.png"
        frame = case.image.copy()
        frame[25:60, 12 + 30 * index : 34 + 30 * index] = 255
        Image.fromarray(frame).save(source)
        sources.append(str(source))
    output = tmp_path / "burst-result"

    assert (
        main(
            [
                "--burst",
                *sources,
                "--mode",
                "camera-screen",
                "--no-auto-rectify",
                "--debug",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "debug" / "burst_uncertainty.png").is_file()


def test_cli_rejects_oversized_burst_before_opening_frames(capsys) -> None:
    missing = [f"missing-{index}.png" for index in range(13)]

    assert main(["--burst", *missing]) == 2
    error = capsys.readouterr().err
    assert "between 2 and 12" in error
    assert "does not exist" not in error


def test_cli_rejects_nonfinite_corners_during_argument_parsing(capsys) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "missing.png",
                "--corners",
                "nan,0;10,0;10,10;0,10",
            ]
        )

    error = capsys.readouterr().err
    assert "finite coordinates" in error
    assert "does not exist" not in error
