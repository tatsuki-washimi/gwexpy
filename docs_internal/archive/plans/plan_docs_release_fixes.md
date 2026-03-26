---
type: plan
title: Documentation Fixes for Public Release
status: approved
created: 2026-01-26
author: Antigravity
---

# Documentation Fixes Plan for Public Release

This plan outlines the steps to address the documentation issues identified in `reports/公開に向けたドキュメント修正事項一覧.md` to ensure a high-quality public release of `gwexpy`.

## Objectives

1. **Eliminate User-Facing Warnings**: Suppress internal library warnings (FutureWarning, UserWarning, LAL-related) in source code to keep tutorials clean.
2. **Optimize Output Logs**: Truncate excessively long output logs in notebooks.
3. **Language Refinement**: Fix mixed language usage in documentation titles (Use English for English pages, Japanese for Japanese pages).

## Task Breakdown

### Phase 1: Critical Cleanup (Warnings & Logs)

Address issues that degrade the user experience immediately.

* **Suppress Warnings (Source Code)**:
  * **Action**: Add `warnings.filterwarnings("ignore", ...)` in relevant `gwexpy` source modules or `__init__.py`.
  * **Target Warnings**:
    * `sklearn` `FutureWarning` ('force_all_finite' renamed...).
    * `TensorFlow` / `Protobuf` `UserWarning`.
    * `LAL` related warnings (if safe to ignore).
    * `control` `FutureWarning` ("fresp attribute is deprecated").

* **Truncate Outputs (Notebooks)**:
  * **Target**: `docs/ja/guide/tutorials/intro_frequencyseries.ipynb`
    * **Action**: Modify `print(frd_sys)` to prevent hundreds of lines of output. Use string slicing (`str(frd_sys)[:1000] + "..."`) or summary view.

### Phase 2: Documentation Standardization (Language)

Standardize the language of titles and headers.

* **English Pages**:
  * Ensure all non-tutorial pages (e.g., Developers Guide, Reports list) use English titles.
  * *Note*: Tutorial contents remain in Japanese for now (user decision).

* **Japanese Pages**:
  * Ensure titles such as "ScalarField Physics Review" in Japanese indexes are translated to Japanese (e.g., "ScalarField 物理レビュー").
  * Keep code terms (ASCII) as is.

### (Deferred) Phase 3: Content Expansion

* *Deferred*: Converting/Translating tutorials to English is postponed.

## Execution Steps

1. **Source Code Modification**: Edit `gwexpy` source files to suppress warnings globally or locally within modules.
2. **Notebook Modification**: Edit `intro_frequencyseries.ipynb` to truncate logs.
3. **Docs Modification**: Rename/Translate titles in `docs/ja/developers/index.rst` (and related `.md` files) and `docs/developers/index.rst`.
4. **Verification**: Run `build_docs` to verify fixes.

## Estimated Effort

* Phase 1: 20 mins
* Phase 2: 30 mins
