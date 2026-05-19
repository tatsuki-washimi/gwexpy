from __future__ import annotations

from tests.io_conformance.contract import (
    CONTRACT_PATH,
    ScenarioName,
    ScenarioStatus,
    load_public_io_contract,
)
from tests.io_conformance.reporting import summarize_blocking_rows
from tests.io_conformance.scenarios import expand_contract_scenarios


def test_blocking_summary_uses_expanded_contract_rows() -> None:
    contract = load_public_io_contract(CONTRACT_PATH)
    rows = expand_contract_scenarios(contract)

    summary = summarize_blocking_rows(rows)
    blocking_rows = [
        row for row in rows if row["scenario"] == ScenarioName.BLOCKING.value
    ]
    expected_blocked = sum(
        row["status"] == ScenarioStatus.BLOCKED.value for row in blocking_rows
    )

    assert summary["blocking_blocked"] == expected_blocked
    assert summary["blocking_total"] == len(blocking_rows)
    assert summary["blocking_display"] == (
        f"{summary['blocking_blocked']} blocked / {summary['blocking_total']} total"
    )
