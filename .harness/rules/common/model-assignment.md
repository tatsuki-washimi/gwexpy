# GWexpy Model Assignment Rules

To maximize reasoning quality and minimize tokens/quota, use the right LLM model for each GWexpy project task.

## Task Categories

- **High-Reasoning (Physics Judge)**: Reviewing physical logic, algorithm correctness, and final PR checks.
- **Mid-Reasoning (Coding specialist)**: Core implementation, data structure changes, refactoring.
- **Action-Heavy (Runner)**: Running tests, fixing minor lints, cleaning up documentation formatting.

## Preferred Assignment Patterns

- **Physics Review**: Use the highest-reasoning model available (Opus tier or equivalent).
- **Core Implementation**: Use a balanced coding-specialist model (Sonnet tier or equivalent).
- **Routine Fixes**: Use a lightweight, fast model (Haiku tier or equivalent).

## Historical Examples

- **Roadmap Planning (@v0.1.0b1)**: Used high-reasoning (Opus tier) for decomposition.
- **HHT Debugging**: Used coding-specialist (Sonnet tier) for loop logic and high-reasoning (Opus tier) for physics validation.
- **Unit Test Generation**: Used lightweight (Haiku tier) for boilerplate.

## Escalation Rules

- Any automated change failing `ruff --check` or `pytest` twice should be escalated.
- Metadata preservation failures in `ScalarField` or `VectorField` must be reviewed by the "Physics Judge" role.

## Anti-Patterns

- Using a high-cost/heavy model for repetitive shell command execution.
- Allowing a low-reasoning runner to merge changes into `gwexpy/fields/` without review.
