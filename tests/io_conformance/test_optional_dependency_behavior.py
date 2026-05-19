from __future__ import annotations

from tests.io_conformance.contract import (
    CONTRACT_PATH,
    ContractCIJobId,
    CoverageStatus,
    MissingDependencyPolicy,
    load_public_io_contract,
)


def _formats_by_canonical(contract: dict[str, object]) -> dict[str, dict[str, object]]:
    return {entry["canonical"]: entry for entry in contract["formats"]}


def test_public_optional_dependency_formats_do_not_use_skip_policy() -> None:
    contract = load_public_io_contract(CONTRACT_PATH)
    formats = _formats_by_canonical(contract)

    for entry in formats.values():
        if not entry["optional_dependencies"]:
            continue
        if entry["public_api"]["read"]:
            assert (
                entry["v3"]["missing_dependency_policy"]["read"]
                != MissingDependencyPolicy.SKIP.value
            )
        if entry["canonical"] != "wav" and entry["public_api"]["write"]:
            assert (
                entry["v3"]["missing_dependency_policy"]["write"]
                != MissingDependencyPolicy.SKIP.value
            )


def test_optional_dependency_formats_use_optional_ci_jobs_except_wav() -> None:
    contract = load_public_io_contract(CONTRACT_PATH)
    formats = _formats_by_canonical(contract)

    wav = formats["wav"]
    assert wav["v3"]["coverage_status"] == CoverageStatus.BLOCKING.value
    assert wav["v3"]["ci_jobs"] == [ContractCIJobId.BASE.value]
    assert (
        wav["v3"]["missing_dependency_policy"]["read"]
        == MissingDependencyPolicy.WARNS_AND_SKIPS_OPTIONAL_METADATA.value
    )
    assert (
        wav["v3"]["missing_dependency_policy"]["write"]
        == MissingDependencyPolicy.SKIP.value
    )

    for canonical, entry in formats.items():
        if canonical == "wav" or not entry["optional_dependencies"]:
            continue
        if not (entry["public_api"]["read"] or entry["public_api"]["write"]):
            continue

        assert entry["v3"]["ci_jobs"]
        assert ContractCIJobId.BASE.value not in entry["v3"]["ci_jobs"]
