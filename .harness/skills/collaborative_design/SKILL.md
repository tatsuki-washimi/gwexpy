---
name: collaborative_design
description: API設計、大規模リファクタリング、物理ロジックの選択など、実装前に方針そのものをユーザーと対話で詰める時に使う。計画書作成の標準入口はsetup_plan
---

# Collaborative Design & Planning

This skill is a narrow guardrail for design-first conversations. Use it when the primary task is to align on policy, API shape, or scientific assumptions before any implementation plan is finalized.

## Core Principle: Conversation First

When high-level design decisions (API design, selection of complex physical logic, large-scale refactoring) are required, the AI prioritizes making a **proposal** and obtaining user approval above all else.

### 1. Pre-implementation "Review" Process
- **Presentation of Choices**: Clearly state trade-offs such as Proposal A (Batch processing/Simple) vs. Proposal B (Extraction-based/Memory-efficient), and present the pros and cons to the user.
- **Verification of Physical/Mathematical Validity**: Discuss with the user whether the equations or normalizations are physically correct before implementation.
- **User Experience (UX) of the API**: Align beforehand on how the user will call the code (e.g., `field.psd()` vs. `compute_psd(field)`).

### 2. Trigger for Starting Work (Explicit Approval)
- While discussing plans or designs, **do not modify the project's source code**, no matter how confident you are.
- Hand off to `setup_plan` once the discussion has converged enough to write an implementation plan.
- Only use coding tools (e.g., `write_to_file`) after receiving affirmative instructions from the user, such as "Proceed," "Go," or "Approved."

### 3. Permitted "Pre-work"
Non-destructive tasks to enhance the accuracy of discussion and analysis can be performed in parallel.
- **Analysis of Existing Code**: Understand the current state using `view_file` or `grep_search`.
- **Prototype Verification**: Create and execute temporary verification scripts (without modifying the project core).
- **Skill Updates**: Solidify knowledge gained through dialogue into agent skills.

## Workflow
1.  **Listen**: Deeply understand the user's requirements and the underlying intent.
2.  **Propose**: Present plan or design proposals in Markdown format.
3.  **Refine**: Receive user feedback and revise/improve the proposal until satisfied.
4.  **Hand Off**: Once the direction is accepted, move to `setup_plan` for the written implementation plan.
