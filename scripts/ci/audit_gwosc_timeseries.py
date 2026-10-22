"""Collect and audit GWOSC-dependent tests inherited from upstream GWpy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from importlib.metadata import version
from pathlib import Path

import pytest
from gwpy.timeseries.tests import test_timeseries as gwpy_timeseries_tests

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests" / "docs" / "gwosc_timeseries_inventory.json"
UPSTREAM_CLASS = "TestTimeSeries"
UPSTREAM_TEST_PATH = "tests/timeseries/test_timeseries.py"
LIVE_FIXTURES = frozenset(
    {"gw150914", "gw150914_16384", "gw150914_h1_32", "gw150914_l1_32"}
)
MARKED_LIVE_FIXTURES = frozenset({"gw150914", "gw150914_16384"})
KNOWN_GWOSC_FIXTURES = LIVE_FIXTURES
DIRECT_LIVE_TESTS = frozenset(
    {
        "test_fetch_open_data",
        "test_fetch_open_data_error",
        "test_find",
        "test_find_best_frametype_in_find",
        "test_find_datafind_httperror",
        "test_find_datafind_runtimeerror",
        "test_get_datafind",
    }
)
DIRECT_GWOSC_TESTS = frozenset(
    {
        "test_fetch_open_data",
        "test_fetch_open_data_error",
        "test_get_gwosc_kwargs",
    }
)


def _is_upstream_item(item: pytest.Item) -> bool:
    return (
        str(item.fspath).endswith(UPSTREAM_TEST_PATH)
        and item.cls is not None
        and item.cls.__name__ == UPSTREAM_CLASS
    )


def _category(item: pytest.Item) -> str | None:
    if not _is_upstream_item(item):
        return None
    if item.originalname in DIRECT_GWOSC_TESTS:
        return "A"
    fixtures = set(item.fixturenames)
    if fixtures.intersection(LIVE_FIXTURES):
        return "C"
    if item.originalname in DIRECT_LIVE_TESTS:
        return "A"
    return None


class CollectionAudit:
    def __init__(self) -> None:
        self.total_collected = 0
        self.all_ids: list[str] = []
        self.inherited_test_ids: list[str] = []
        self.items: list[dict[str, object]] = []
        self.unknown_fixture_ids: list[str] = []
        self.unknown_network_ids: list[str] = []

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        self.total_collected = len(items)
        self.all_ids = sorted(item.nodeid for item in items)
        for item in items:
            if (
                str(item.fspath).endswith(".py")
                and "/tests/timeseries/" in str(item.fspath)
                and item.get_closest_marker("network") is not None
                and _category(item) is None
            ):
                self.unknown_network_ids.append(item.nodeid)
            if not _is_upstream_item(item):
                continue
            self.inherited_test_ids.append(item.nodeid)

            fixtures = set(item.fixturenames)
            unknown_fixtures = {
                name
                for name in fixtures
                if name.startswith("gw150914") and name not in KNOWN_GWOSC_FIXTURES
            }
            if unknown_fixtures:
                self.unknown_fixture_ids.append(
                    f"{item.nodeid}: {', '.join(sorted(unknown_fixtures))}"
                )

            category = _category(item)
            if category is not None:
                reason = (
                    "Upstream GWOSC/DataFind live request"
                    if category == "A"
                    else "Upstream calculation/I/O case obtains its input from a live GWOSC fixture; equivalent offline GWexpy regressions remain required"
                )
                self.items.append(
                    {
                        "nodeid": item.nodeid,
                        "category": category,
                        "fixtures": sorted(fixtures.intersection(KNOWN_GWOSC_FIXTURES)),
                        "reason": reason,
                        "upstream_gwosc_live": item.get_closest_marker(
                            "upstream_gwosc_live"
                        )
                        is not None,
                    }
                )


def upstream_source_sha256() -> str:
    path = Path(gwpy_timeseries_tests.__file__).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_candidate_cases(
    current: list[dict[str, object]], expected: list[dict[str, object]]
) -> None:
    """Fail closed if IDs or their audited fixture/classification metadata drift."""
    current_sorted = sorted(current, key=lambda row: str(row["nodeid"]))
    expected_sorted = sorted(expected, key=lambda row: str(row["nodeid"]))
    if current_sorted != expected_sorted:
        raise SystemExit(
            "GWOSC candidate inventory metadata mismatch; audit fixture, "
            "category, reason, and marker changes before updating the manifest"
        )


def audit(output: Path, write_baseline: bool = False) -> int:
    collector = CollectionAudit()
    result = pytest.main(
        ["--collect-only", "-qq", "tests/timeseries"],
        plugins=[collector],
    )
    if result != pytest.ExitCode.OK:
        raise SystemExit(f"pytest collection failed with exit code {result}")

    gwpy_version = version("gwpy")
    live_marker_ids = sorted(
        entry["nodeid"] for entry in collector.items if entry["upstream_gwosc_live"]
    )
    live_marker_set = set(live_marker_ids)
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    expected_source_sha = os.environ.get("EXPECTED_SOURCE_SHA")
    if expected_source_sha and source_sha != expected_source_sha:
        raise SystemExit(
            f"audit source SHA {source_sha} does not match expected SHA {expected_source_sha}"
        )
    report = {
        "schema": "gwexpy-gwosc-timeseries-audit-v2",
        "source_sha": source_sha,
        "gwpy_version": gwpy_version,
        "gwpy_upstream_test_timeseries_sha256": upstream_source_sha256(),
        "total_collected": collector.total_collected,
        "inherited_upstream_test_count": len(collector.inherited_test_ids),
        "candidate_dependency_count": len(collector.items),
        "live_marker_count": len(live_marker_ids),
        "required_gate_count": collector.total_collected - len(live_marker_ids),
        "non_candidate_count": collector.total_collected - len(collector.items),
        "all_ids": collector.all_ids,
        "inherited_upstream_test_ids": sorted(collector.inherited_test_ids),
        "required_ids": sorted(set(collector.all_ids) - live_marker_set),
        "live_marker_ids": live_marker_ids,
        "candidate_cases": sorted(collector.items, key=lambda row: str(row["nodeid"])),
        "unknown_fixture_ids": sorted(collector.unknown_fixture_ids),
        "unknown_network_ids": sorted(collector.unknown_network_ids),
        "category_counts": {
            category: sum(1 for row in collector.items if row["category"] == category)
            for category in ("A", "B", "C")
        },
    }

    if write_baseline:
        expected = {
            "schema": "gwexpy-gwosc-timeseries-candidate-inventory-v2",
            "gwpy_versions": ["4.0.1", "4.0.2"],
            "upstream_test_timeseries_sha256": upstream_source_sha256(),
            "candidate_cases": report["candidate_cases"],
            "live_marker_ids": live_marker_ids,
            "inherited_upstream_test_ids": report["inherited_upstream_test_ids"],
        }
        MANIFEST.write_text(
            json.dumps(expected, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if gwpy_version not in expected["gwpy_versions"]:
            raise SystemExit(f"GWpy {gwpy_version} is not in the audited matrix")
        if upstream_source_sha256() != expected["upstream_test_timeseries_sha256"]:
            raise SystemExit(
                "upstream GWpy test module changed; audit it before updating the inventory"
            )
        _validate_candidate_cases(collector.items, expected["candidate_cases"])
        if live_marker_ids != expected["live_marker_ids"]:
            added = sorted(live_marker_set - set(expected["live_marker_ids"]))
            missing = sorted(set(expected["live_marker_ids"]) - live_marker_set)
            raise SystemExit(
                f"GWOSC marker inventory mismatch; added={added!r}, missing={missing!r}"
            )
        expected_inherited_ids = set(expected["inherited_upstream_test_ids"])
        current_inherited_ids = set(collector.inherited_test_ids)
        if current_inherited_ids != expected_inherited_ids:
            added = sorted(current_inherited_ids - expected_inherited_ids)
            missing = sorted(expected_inherited_ids - current_inherited_ids)
            raise SystemExit(
                "inherited GWpy test inventory mismatch; "
                f"added={added!r}, missing={missing!r}"
            )

    if collector.unknown_fixture_ids:
        raise SystemExit(
            f"unclassified GWOSC fixtures: {collector.unknown_fixture_ids!r}"
        )
    if collector.unknown_network_ids:
        raise SystemExit(
            f"unclassified timeseries network tests: {collector.unknown_network_ids!r}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"gwpy={gwpy_version} collected={collector.total_collected} "
        f"live_fixture_candidates={len(collector.items)} "
        f"live_marker={len(live_marker_ids)} required_gate={collector.total_collected - len(live_marker_ids)} "
        f"non_candidates={collector.total_collected - len(collector.items)} "
        f"source_sha={report['source_sha']}"
    )
    for entry in sorted(collector.items, key=lambda row: str(row["nodeid"])):
        print(f"{entry['category']} {entry['nodeid']} ({', '.join(entry['fixtures'])})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    return audit(args.output, write_baseline=args.write_baseline)


if __name__ == "__main__":
    raise SystemExit(main())
