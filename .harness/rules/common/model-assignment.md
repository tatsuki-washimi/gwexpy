# GWexpy Model Assignment Rules

To maximize reasoning quality and minimize tokens/quota, use the right LLM model for each GWexpy project task.

## Task Categories

- **High-Reasoning (Physics Judge)**: Reviewing physical logic, algorithm correctness, and final PR checks.
- **Mid-Reasoning (Coding specialist)**: Core implementation, data structure changes, refactoring.
- **Action-Heavy (Runner)**: Running tests, fixing minor lints, cleaning up documentation formatting.

## Preferred Assignment Patterns

- **Physics Review**: Use the most capable model (e.g., Claude 3.5 Opus/Sonnet or Gemini 3 Flash / 1.5 Pro).
- **Core Implementation**: Sonnet or Gemini 1.5/3 Pro/Flash.
- **Routine Fixes**: Haiku or Gemini 1.5 Flash.

## Historical Examples

- **Roadmap Planning (@v0.1.0b1)**: Used Opus/Sonnet for decomposition.
- **HHT Debugging**: Used Sonnet for loop logic and Opus for physics validation.
- **Unit Test Generation**: Used Flash/Haiku for boilerplate.

## Escalation Rules

- Any automated change failing `ruff --check` or `pytest` twice should be escalated.
- Metadata preservation failures in `ScalarField` or `VectorField` must be reviewed by the "Physics Judge" role.

## Anti-Patterns

- Using a high-cost/heavy model for repetitive shell command execution.
- Allowing a low-reasoning runner to merge changes into `gwexpy/fields/` without review.
