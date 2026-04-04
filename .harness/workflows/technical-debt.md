---
name: technical-debt
description: GWexpy 技術的負債消化ワークフロー。backlog から安全な着手単位へ分解し、優先度付け・実装・検証・記録まで管理する。
trigger: manual
---

# GWexpy Technical Debt Workflow

Use this workflow to systematically address items in `docs_internal/archive/plans/improvement_tasks_backlog.md`.

## Step 1: Debt Intake
1. Select a task from the backlog or identify a new debt.
2. Document the affected files, the current issue, and the risk of leaving it as is.
3. Define the desired state.

## Step 2: Prioritization
- **P0**: Critical for security, data integrity, or core physics.
- **P1**: Essential for usability, documentation, or CI stability.
- **P2**: Maintenance and non-breaking improvements.

## Step 3: Slice into Phases
- Break large tasks into PR-sized chunks (e.g., "Step 1: Refactor", "Step 2: Add Tests").
- Each phase must result in a stable, passing build.

## Step 4: Role Assignment
- Choose the expert agent for the task (e.g., `exception-auditor` for broad catches).
- Assign a "Physics Judge" model for final validation of any physical logic.

## Step 5: Implement and Verify
1. Run `ruff`, `mypy`, and `pytest`.
2. For UI/GUI changes, manually verify on a headless virtual display if needed.
3. Check for documentation and example notebook drift.

## Step 6: Recording Outcomes
1. Update `docs_internal/archive/plans/improvement_tasks_backlog.md` (mark as complete or updated).
2. Record changes in `CHANGELOG.md`.
3. Generate an `evidence-pack` or audit manifest for the PR.

## Exit Criteria
- The specific technical debt item is resolved or moved to a better state.
- Documentation and tests are updated to prevent regression.
