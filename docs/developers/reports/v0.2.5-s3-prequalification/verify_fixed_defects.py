"""Check installed-artifact JUnit evidence for the 36 fixed audit findings."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MATRIX = ROOT.parent / "2026-09-27-public-io-cross-format-post-fix-matrix.json"


def junit_node(case):
    parts = case.attrib["classname"].split(".")
    assert parts[:2] == ["tests", "io"]
    path = "/".join(parts[:2]) + "/" + parts[2] + ".py"
    name = case.attrib["name"].split("[", 1)[0]
    return "::".join((path, *parts[3:], name))


def check_junit(path, required):
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    assert len(suites) == 1
    suite = suites[0]
    assert all(
        int(suite.attrib[field]) == 0 for field in ("errors", "failures", "skipped")
    )
    cases = list(suite.iter("testcase"))
    assert len(cases) == int(suite.attrib["tests"])
    observed = defaultdict(list)
    for case in cases:
        assert not any(
            case.find(tag) is not None for tag in ("error", "failure", "skipped")
        )
        observed[junit_node(case)].append(case)
    assert all(observed[node] for node in required), sorted(required - observed.keys())
    assert set(observed) == required, sorted(observed.keys() - required)
    return {"testcases": len(cases), "regression_nodes": len(observed)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-junit", type=Path, required=True)
    parser.add_argument("--sdist-junit", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    findings = [
        finding
        for finding in json.loads(MATRIX.read_text())["findings"]
        if finding["post_fix_status"] == "FIX_VERIFIED"
    ]
    assert len(findings) == 36
    ids = [finding["finding_id"] for finding in findings]
    assert len(set(ids)) == 36
    required = {node for finding in findings for node in finding["regression_tests"]}
    assert len(required) == 23
    results = {
        "wheel": check_junit(args.wheel_junit, required),
        "sdist": check_junit(args.sdist_junit, required),
    }
    assert results["wheel"] == results["sdist"]
    if args.summary is not None:
        summary = json.loads(args.summary.read_text())
        assert summary["finding_ids"] == ids
        assert summary["regression_nodes"] == sorted(required)
        for artifact, junit in (
            ("wheel", args.wheel_junit),
            ("sdist", args.sdist_junit),
        ):
            evidence = summary["artifacts"][artifact]
            assert (
                hashlib.sha256(junit.read_bytes()).hexdigest()
                == evidence["junit_sha256"]
            )
            assert results[artifact] == {
                "regression_nodes": evidence["regression_nodes"],
                "testcases": evidence["testcases_passed"],
            }
            assert evidence["testcases_skipped"] == 0
    print(json.dumps({"finding_count": len(ids), "artifacts": results}, sort_keys=True))


if __name__ == "__main__":
    main()
