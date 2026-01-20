---
name: estimate_effort
description: 作業計画から必要な時間とLLMクオータ（トークン消費量）の見積もりを算出し、コストパフォーマンスを評価する
---

# Estimate Effort

This skill analyzes a proposed implementation plan or task description to estimate the time required for completion and the approximate LLM quota consumption.

## Instructions

1. **Task Decomposition**:
    * Break down the task into phases such as "Research", "Implementation", "Testing", and "Refactoring".
    * Assess the difficulty of each phase (Low / Medium / High).

2. **Time Estimation**:
    * Estimate the total time (in minutes) required for the assistant to think and execute, based on task scale and complexity.

3. **Quota Estimation**:
    * Predict token consumption based on the following criteria:
        * **High**: Large-scale file creation/replacement, multiple research calls (`search_web`, `grep_search`), complex logic discussions.
        * **Medium**: Small fixes to existing code, standard test execution.
        * **Low**: Reading only, simple renaming, short response generation.
    * Account for variations between models (e.g., Gemini Pro 1.5 vs. Flash).

4. **Efficiency Evaluation**:
    * Evaluate the Return on Investment (ROI) by comparing predicted results with costs (time/quota).
    * Propose "lower-cost alternative approaches" if necessary.

5. **Reporting**:
    * Generate an estimation report in the following format:
        * **Estimated Total Time**: XX minutes
        * **Estimated Quota Consumption**: [Low / Medium / High]
        * **Breakdown**: Estimates per step
        * **Concerns**: Uncertain elements that could cause significant time/quota overruns.
