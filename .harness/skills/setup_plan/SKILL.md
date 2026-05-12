---
name: setup_plan
description: 新しい作業を始める時に使う標準の計画作成入口。要求を分解し、必要ならcollaborative_designを前段に挟みつつ、モデル・スキル・工数込みの実装計画書を保存・更新する。複数エージェント/worker数/model/effort/レビューワー確認が求められる場合は global multi-agent-planning-gate を参照する
---

# Setup Plan Workflow

This is the default entry point for turning a request into a written implementation plan.

## Instructions

1. **Requirement Understanding and Decomposition**:
    * Read and interpret chat requests or provided plans (Markdown), decomposing them into specific work steps.
    * Identify classes, methods, and test items that require implementation.
    * If the task is still at the policy-discussion stage, call `collaborative_design` first and return here after alignment.
    * If the user asks for multi-agent planning, worker count, model/effort recommendations, parallelization, or reviewer confirmation, first consult the global `multi-agent-planning-gate` skill, then return here with a plan that explicitly includes the critical path, parallel lanes, write scopes, model/effort, reviewer gate, and verification.

2. **AI Assistant Strategy Formulation**:
    * Call the following skills in order to analyze task characteristics:
        * `suggest_next`: Propose the optimal LLM and auxiliary skills based on task difficulty.
        * `estimate_effort`: Predict the required time and quota consumption.

3. **Creation or Update of Plan**:
    * Create or update a timestamped detailed plan (`.md`) in the `docs/developers/plans/` directory.
    * Include the following sections:
        * **Objectives & Goals**
        * **Detailed Roadmap (by Phase)**
        * **Testing & Verification Plan**
        * **Models, Recommended Skills, and Effort Estimates** (reflecting results of step 2)
        * If the multi-agent gate was triggered, also include **Critical Path**, **Parallel Lanes**, **Write Scopes**, **Reviewer Gate**, and **Integration Verification**.

4. **Await Model Selection**:
    * Present the analysis results to the user and **wait for user approval** (selection of LLM model or plan approval) before proceeding to the next step.

5. **Confirmation for Continuation**:
    * Upon obtaining approval, proceed to execute the first phase or ask for further instructions.
