import json
from pathlib import Path


def test_golden_fixture_inventory_is_source_grouped_and_large_enough() -> None:
    path = Path(__file__).parent / "fixtures" / "golden_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cases = manifest["cases"]

    assert len(cases) == 20
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case.get("source_group_id") for case in cases)
    assert {case["family"] for case in cases} >= {
        "dots-v1",
        "polygon-mosaic-v1",
    }
    assert all("path" not in case for case in cases)
