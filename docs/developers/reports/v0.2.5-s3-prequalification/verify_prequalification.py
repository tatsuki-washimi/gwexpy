import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MATRIX = ROOT.parent / "2026-09-27-public-io-cross-format-post-fix-matrix.json"
FINDINGS = [
    f
    for f in json.loads(MATRIX.read_text())["findings"]
    if f["post_fix_status"] == "NOT_REQUALIFIED"
]
BLOCKED = {
    f["finding_id"]: f for f in FINDINGS if f["baseline_finding_status"] == "BLOCKED"
}
BLOCKED_REFS = {
    "NC-MATRIX-008": [33],
    "NC-MATRIX-009": [30, 32],
    "ZARR-DTYPE-003": [2, 6, 17],
    "TDMS-TIME-007": [157],
    "TDMS-TIME-008": [163],
    "TDMS-UNIT-001": [145],
    "GBD-COUNT-001": [43],
    "GBD-LEGACY-001": [],
    "ATS-TRUNC-001": [5],
    "HDF5-DISCOVERY-TS-DICT-GROUP-001": [4],
    "HDF5-HIST-DATASET-001": [25, 29],
    "OBSPY-DUP-BASE-001": [1],
}


def load(artifact, command):
    root = ROOT / "raw" / artifact
    path = root / f"{command}.stdout.jsonl"
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ], hashlib.sha256(path.read_bytes()).hexdigest()


def result(artifact, finding):
    fid = finding["finding_id"]
    command = finding["baseline_command_id"]
    rows, digest = load(artifact, command)
    refs = []

    def row(i):
        refs.append(i)
        return rows[i]

    try:
        if fid == "NC-ROUTE-001":
            x = row(2)
            assert x["route"] == "matrix-explicit-nc"
            assert x["result"]["shape"] == [2, 2, 3]
            assert x["result"]["rows"] == ["r0", "r1"] and x["result"]["cols"] == [
                "c0",
                "c1",
            ]
            assert x["result"]["values"][1][1] == [22.0] * 3
            # Supplement the historical probe with six public read routes.
            extra = [
                json.loads(line)
                for line in (ROOT / "raw" / f"nc-routes-{artifact}.jsonl")
                .read_text()
                .splitlines()
                if line.strip()
            ]
            assert len(extra) == 6
            for observed in extra:
                y = observed["result"]
                assert (
                    y["type"]
                    == {
                        "single": "TimeSeries",
                        "dict": "TimeSeriesDict",
                        "matrix": "TimeSeriesMatrix",
                    }[observed["kind"]]
                )
                values = y["values"]
                if observed["kind"] == "dict":
                    assert y["keys"] == ["signal"]
                    values = values["signal"]
                elif observed["kind"] == "matrix":
                    assert y["shape"] == [1, 1, 3]
                    values = values[0][0]
                assert values == [9007199254740993, -9007199254740995, 17]
            assert {x["route"] for x in extra} == {"explicit-nc", "auto"}
        elif fid == "ZARR-DTYPE-004":
            for i in (21, 22, 23):
                x = row(i)
                y = x["result"]["series"]["signal"] if i == 22 else x["result"]
                assert y["dtype"] == "float64"
                assert (y["real"] if i != 23 else y["real"][0][0]) == [1.1, -2.2, 3.3]
                assert (
                    y["unit"] == "V" and y["t0"] == 1234567890.25 and y["dt"] == 0.0625
                )
        elif fid == "OPTIONAL-001":
            x = row(0)
            assert (
                x["dependencies"]["zarr"] is None
                and x["dependencies"]["xarray"] is None
                and x["dependencies"]["netCDF4"] is None
            )
            for i in range(1, 25):
                x = row(i)
                assert x["error_type"] == "ImportError" and "Install" in x["error"]
        elif fid in ("TDMS-TIME-006", "TDMS-TIME-009"):
            x = row(145 if fid.endswith("006") else 146)["actual"]
            assert (
                x["value"] == [11, 13, 17] and x["dt"] == 0.25 and x["dtype"] == "int16"
            )
            assert x["t0"] == (1388199863.0 if fid.endswith("006") else 1234567890.0)
        elif fid == "TDMS-OPTIONAL-001":
            for i in (1, 2, 3):
                x = row(i)
                assert (
                    x["actual"]["error_type"] == "ImportError"
                    and "nptdms" in x["actual"]["error"].lower()
                )
        elif fid == "GBD-CONTROL-001":
            x = row(5)["actual"]["channels"]["CH1"]
            y = row(7)["actual"]["channels"]["CH1"]
            assert x["value"] == y["value"] == [1.0, -2.0, 5.0]
            assert x["unit"] == "V" and x["dt"] == 0.1
            assert x["t0"] - y["t0"] == 32400.0
        elif fid == "GBD-HEADER-002":
            for i in (19, 21, 23):
                assert row(i)["actual"]["error_type"] == "ValueError"
        elif fid == "GBD-TRUNC-001":
            for i in (49, 51, 53):
                assert row(i)["actual"]["error_type"] == "ValueError"
        elif fid == "GBD-DIGITAL-001":
            x = row(57)["actual"]["channels"]
            assert x["Alarm"]["value"] == [0.0, 1.0, 1.0]
            assert x["CH1"]["value"] == [1.0, -2.0, 5.0]
        elif fid == "ATS-CONTROL-001":
            x = row(1)["actual"]
            assert x["value"] == [0.5, -1.0, 1.5] and x["dt"] == 0.25
            assert x["unit"] == "V" and x["t0"] == 1388199863.0
        elif fid == "ATS-RATE-001":
            for i in (9, 13):
                assert row(i)["actual"]["error_type"] == "ValueError"
        elif fid == "ATS-EPOCH-001":
            x = row(2)["actual"]
            assert (
                x["value"] == [0.5, -1.0, 1.5]
                and x["dt"] == 0.25
                and x["t0"] == 1234567890.0
            )
        elif fid == "MATRIX-CONVERSION-001":
            for i in (0, 1):
                x = row(i)
                assert "unit=, channel=None" in x["cells"][0][0]
        elif fid in (
            "HDF5-DISCOVERY-TS-LIST-001",
            "HDF5-DISCOVERY-FS-001",
            "HDF5-DISCOVERY-SPEC-001",
            "HDF5-DISCOVERY-HIST-001",
        ):
            idx = {
                "HDF5-DISCOVERY-TS-LIST-001": 6,
                "HDF5-DISCOVERY-FS-001": 14,
                "HDF5-DISCOVERY-SPEC-001": 22,
                "HDF5-DISCOVERY-HIST-001": 30,
            }[fid]
            x = row(idx)
            assert x["authority"] == "C2_discovery" and x["shape"] == "List"
            y = x["mutant"]["result"]
            assert y["keys"] == ["0", "1"]
            assert [y["entries"][k]["metadata"]["name"] for k in y["keys"]] == [
                "A",
                "C",
            ]
        elif fid == "HDF5-DISCOVERY-TS-DICT-DATASET-001":
            x = row(2)
            assert (
                x["authority"] == "C2_discovery"
                and x["shape"] == "Dict"
                and x["layout_requested"] == "dataset"
            )
            assert x["mutant"]["exception"] == "TypeError"
        elif fid == "OBSPY-DUP-001":
            lengths = {"contiguous": 8, "gap": 10, "overlap_conflict": 6}
            for i in (1, 2, 3):
                x = row(i)
                y = x["public"]["result"]
                assert (
                    list(y) == ["XX.TEST..BHZ"]
                    and len(y["XX.TEST..BHZ"]["values"]) == lengths[x["variant"]]
                )
            assert row(4)["public"]["exception"] == "TypeError"
        elif fid == "SDB-KW-001":
            x = row(5)["reads"]
            assert x["direct_default"]["result"] == x["direct_epoch"]["result"]
            assert x["registry_default"]["result"] == x["registry_epoch"]["result"]
            assert x["direct_default"]["result"]["outTemp"]["t0"] == 1384035218.0
        elif fid == "WIN-KW-001":
            x = row(6)["reads"]
            assert x["direct_default"]["result"] == x["direct_epoch"]["result"]
            assert x["registry_default"]["result"] == x["registry_epoch"]["result"]
            assert x["direct_default"]["result"]["...0001"]["values"] == [10, 11]
        elif fid == "WIN-OPTIONAL-001":
            x = row(3)["reads"]
            assert x["direct_default"]["exception"] == "ImportError"
            assert x["registry_auto_epoch"]["exception"] == "IORegistryError"
        elif fid in ("WAV-CONTROL-001", "AUDIO-CONTROL-001"):
            x = row(7 if fid.startswith("WAV") else 8)
            y = x["reads"]["dict_default"]["result"]
            assert list(y) == ["channel_0", "channel_1"]
            assert y["channel_0"]["dt"] == y["channel_1"]["dt"] == 0.000125
            assert y["channel_0"]["t0"] == y["channel_1"]["t0"] == 0.0
            scale = 1 if fid.startswith("WAV") else 32768
            assert y["channel_0"]["values"] == [
                v / scale for v in [1000, 2000, 3000, 4000, 5000]
            ]
            assert y["channel_1"]["values"] == [
                v / scale for v in [-2000, -1000, 0, 1000, 2000]
            ]
        elif fid == "AUDIO-OPTIONAL-001":
            wav = row(4)["reads"]
            flac = row(5)["reads"]
            assert wav["dict_default"]["result"]["channel_0"]["values"] == [
                1000,
                2000,
                3000,
                4000,
                5000,
            ]
            assert wav["dict_extract_metadata"]["warnings"]
            assert flac["dict_default"]["exception"] == "ImportError"
        else:
            raise AssertionError(f"unhandled case: {fid}")
        return (
            "PASS",
            refs,
            "Case-specific public-result assertion passed on installed artifact.",
            digest,
        )
    except (AssertionError, KeyError, IndexError, TypeError) as e:
        return "FAIL", refs, f"{type(e).__name__}: {e}", digest


entries = []
for f in FINDINGS:
    fid = f["finding_id"]
    entry = {
        "finding_id": fid,
        "baseline_status": f["baseline_finding_status"],
        "command_id": f["baseline_command_id"],
    }
    if fid in BLOCKED:
        entry["status"] = "BLOCKED"
        entry["reason"] = f["baseline_expected"]
        entry["artifact_checks"] = {}
        for artifact in ("wheel", "sdist"):
            rows, digest = load(artifact, f["baseline_command_id"])
            indices = BLOCKED_REFS[fid]
            assert all(0 <= index < len(rows) for index in indices)
            entry["artifact_checks"][artifact] = {
                "status": "BLOCKED",
                "row_indices": indices,
                "raw_sha256": digest,
            }
        entry["observations"] = (
            "The referenced rows characterize the current result; the historical fixture/authority gap remains open."
        )
    else:
        checks = {}
        for artifact in ("wheel", "sdist"):
            status, refs, note, digest = result(artifact, f)
            checks[artifact] = {
                "status": status,
                "row_indices": refs,
                "note": note,
                "raw_sha256": digest,
            }
        entry["artifact_checks"] = checks
        entry["status"] = (
            "PASS"
            if all(v["status"] == "PASS" for v in checks.values())
            else (
                "FAIL"
                if any(v["status"] == "FAIL" for v in checks.values())
                else "NEEDS_HARNESS"
            )
        )
    entries.append(entry)
assert len(entries) == 38 and len({x["finding_id"] for x in entries}) == 38
assert set(BLOCKED_REFS) == set(BLOCKED)
out = {
    "schema": "gwexpy-v025-audit-prequalification-v1",
    "source_sha": "159a338081e1fca2c77030cabe160a458217563b",
    "historical_matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
    "artifact_sha256": {
        "wheel": "96e1f72e68404c718ad6f881119ede983782d469db6d146141659eac21bd469d",
        "sdist": "515bb91a960e6dfc6f85537bcf4775fa6cbf7e1c60c14141608593160f2d2f7d",
    },
    "status_counts": {
        s: sum(x["status"] == s for x in entries)
        for s in ("PASS", "FAIL", "NEEDS_HARNESS", "BLOCKED")
    },
    "release_gate": "OPEN",
    "findings": entries,
}
path = ROOT / "prequalification-38.json"
path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(out["status_counts"], sort_keys=True))
print(path)
