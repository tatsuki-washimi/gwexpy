# Section 14-10 Physics-Comment Remediation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve section 14-10 by rewriting tutorial and case-study notebook comments so they explain the physical rationale behind analysis steps, while promoting valuable legacy notebooks from `examples/` into the public docs source tree without preserving duplicate maintenance paths.

**Architecture:** Treat `docs/web/{en,ja}/user_guide/tutorials/*.ipynb` as the canonical source for published notebooks, and treat `examples/` as a staging/incubation area for legacy or runnable artifacts. Add a lightweight developer rubric for “physics-intent comments”, apply it to a focused set of existing public notebooks, then promote selected legacy notebooks into the public tree and rewrite their comments during promotion instead of after the fact.

**Tech Stack:** Jupyter notebooks (`.ipynb` JSON), nbsphinx/Sphinx docs, bilingual docs (`ja` / `en`), Markdown/reStructuredText, `conda run -n gwexpy`

---

## Findings

- The current public docs behave as though `docs/web/{en,ja}/user_guide/tutorials/*.ipynb` are canonical:
  - tutorial index pages tell users to download notebooks from `docs/web/.../tutorials/`
  - `docs/conf.py` builds nbsphinx pages from `docs/`
  - public quickstart/getting-started pages link directly to those notebook paths
- `examples/README.md` still says the opposite and claims `examples/` is the source of truth. That is now inaccurate and will reintroduce drift if section 14-10 is fixed in both places independently.
- The three overlapping case studies already diverged materially between legacy and public copies:
  - `case_transfer_function.ipynb`: similarity about `0.56`
  - `case_active_damping.ipynb`: similarity about `0.71`
  - `case_noise_budget.ipynb`: similarity about `0.41`
- The highest-value public notebooks for section 14-10 are the ones where code comments narrate API mechanics but often skip the detector or signal-processing reason for the step:
  - `docs/web/en/user_guide/tutorials/intro_timeseries.ipynb`
  - `docs/web/en/user_guide/tutorials/segment_asd_pipeline.ipynb`
  - `docs/web/en/user_guide/tutorials/case_transfer_function.ipynb`
  - `docs/web/en/user_guide/tutorials/case_active_damping.ipynb`
  - `docs/web/en/user_guide/tutorials/case_noise_budget.ipynb`
- Valuable legacy notebooks worth promoting because they cover physically meaningful workflows that are not yet first-class in public docs:
  - `examples/case-studies/case_lockin_detection.ipynb`
  - `examples/case-studies/case_wiener_filter.ipynb`
  - `examples/case-studies/case_signal_extraction.ipynb`
  - `examples/case-studies/case_coupling_analysis.ipynb`
  - `examples/advanced-methods/tutorial_Control_00_DiscretizationBasics.ipynb`
  - `examples/advanced-methods/tutorial_Control_01_Basics.ipynb`
  - `examples/advanced-methods/tutorial_Control_02_Modeling.ipynb`
- Not every notebook needs physics-heavy commentary. Plotting, interop, and pure container walkthroughs still need concise code comments, but the rubric should apply only when code cells encode a physical analysis choice:
  - filter/window/band selection
  - channel mixing or projection
  - model parameterization
  - assumptions about stationarity, resonance, damping, coherence, or calibration

## Scope Decisions

- Do not edit both `examples/` and `docs/web/.../tutorials/` for the same published notebook.
- Do not promote low-value or redundant legacy notebooks in the same pass, especially:
  - legacy duplicates of `case_transfer_function`, `case_active_damping`, `case_noise_budget`
  - generic interop or plotting notebooks already covered publicly
- Do not introduce a new docs taxonomy section in this pass. Reuse the current tutorial/examples grouping unless a promoted notebook clearly belongs in the existing `examples/index.rst` gallery instead of the tutorial index.
- Do not execute notebooks during the fix unless a notebook’s JSON metadata or static outputs require regeneration. Section 14-10 is fundamentally a content/annotation fix, not an execution fix.

## Target Published Set

Promote the following legacy notebooks into the public docs source tree:

### Tutorials to add to `tutorials/index.rst`

- `advanced_control_discretization.ipynb`
  - Source seed: `examples/advanced-methods/tutorial_Control_00_DiscretizationBasics.ipynb`
  - Proposed title: `Control Analysis: Discretization Basics`
- `advanced_control_basics.ipynb`
  - Source seed: `examples/advanced-methods/tutorial_Control_01_Basics.ipynb`
  - Proposed title: `Control Analysis: Resonance and Feedback Basics`
- `advanced_control_modeling.ipynb`
  - Source seed: `examples/advanced-methods/tutorial_Control_02_Modeling.ipynb`
  - Proposed title: `Control Analysis: Plant Modeling from Measured Response`

### Case studies to add to `examples/index.rst`

- `case_lockin_detection.ipynb`
  - Source seed: `examples/case-studies/case_lockin_detection.ipynb`
  - Proposed title: `Lock-in Detection: Recovering Weak AM/FM Structure`
- `case_wiener_filter.ipynb`
  - Source seed: `examples/case-studies/case_wiener_filter.ipynb`
  - Proposed title: `Wiener Filtering: Coherent Noise Subtraction`
- `case_signal_extraction.ipynb`
  - Source seed: `examples/case-studies/case_signal_extraction.ipynb`
  - Proposed title: `Signal Extraction: Weak Signal Recovery from Colored Noise`
- `case_coupling_analysis.ipynb`
  - Source seed: `examples/case-studies/case_coupling_analysis.ipynb`
  - Proposed title: `Coupling Analysis: Estimating Transfer Paths Between Channels`

## Physics-Comment Rubric

Create a short durable rubric and apply it consistently.

Required properties for comments that describe an analysis step:

1. State the physical role of the signal or channel.
2. State why this transform/filter/window/model is chosen here.
3. State what artifact, ambiguity, or failure mode the step is preventing.
4. Prefer detector or measurement vocabulary over API narration.
5. Avoid comments that only restate the next function call.

Example replacements to use as templates during rewrite:

```python
# Bad: Calculate ASD for each segment
# Good: Estimate ASD per science-valid segment so stationary noise floors can be compared without smearing transient glitches into the average.

# Bad: Closed-loop system
# Good: Close the feedback loop here to check whether modal damping suppresses the suspension resonance without amplifying cross-coupled rigid-body motion.

# Bad: 2. Low-pass Filter
# Good: Low-pass after mixing removes the 2*f_c image term so the slowly varying envelope and phase drift remain as baseband observables.
```

### Task 1: Establish Notebook Ownership and the Comment Rubric

**Files:**
- Create: `docs/developers/guides/notebook_physics_comment_rubric.md`
- Modify: `docs/NOTEBOOK_POLICY.md`
- Modify: `examples/README.md`

- [ ] **Step 1: Write the rubric document**

Create `docs/developers/guides/notebook_physics_comment_rubric.md` with these sections:

```md
# Notebook Physics Comment Rubric

## When to add physics-intent comments
- Filters, windows, whitening, segmentation, projection, fitting, coherence, transfer functions, damping, demodulation, calibration

## What good comments explain
1. Physical role of the signal/channel
2. Why the method or parameter is chosen
3. What analysis risk it controls

## What to avoid
- Commenting obvious syntax
- Repeating function names
- Long textbook paragraphs inside code cells

## Rewrite examples
- ASD example
- lock-in example
- modal damping example
```

- [ ] **Step 2: Update notebook policy to define source ownership**

Revise `docs/NOTEBOOK_POLICY.md` so it says:

```md
- Published notebook pages are maintained in `docs/web/{en,ja}/user_guide/tutorials/`.
- `examples/` is for legacy runnable notebooks, incubation, and non-published artifacts.
- When a legacy notebook is promoted, it must be copied/adapted into the public docs tree and then linked from the relevant index.
- Section-14-10 style comment rewrites must follow `docs/developers/guides/notebook_physics_comment_rubric.md`.
```

- [ ] **Step 3: Rewrite `examples/README.md` to remove the canonical-source conflict**

Replace the current “source of truth” claim with language like:

```md
- Public documentation notebooks live under `docs/web/{en,ja}/user_guide/tutorials/`.
- `examples/` contains legacy notebooks, promotion candidates, and runnable support material.
- If a notebook exists in both places, the public docs copy is authoritative for published content.
```

### Task 2: Rewrite Priority Existing Public Notebooks

**Files:**
- Modify: `docs/web/en/user_guide/tutorials/intro_timeseries.ipynb`
- Modify: `docs/web/ja/user_guide/tutorials/intro_timeseries.ipynb`
- Modify: `docs/web/en/user_guide/tutorials/segment_asd_pipeline.ipynb`
- Modify: `docs/web/ja/user_guide/tutorials/segment_asd_pipeline.ipynb`
- Modify: `docs/web/en/user_guide/tutorials/case_transfer_function.ipynb`
- Modify: `docs/web/ja/user_guide/tutorials/case_transfer_function.ipynb`
- Modify: `docs/web/en/user_guide/tutorials/case_active_damping.ipynb`
- Modify: `docs/web/ja/user_guide/tutorials/case_active_damping.ipynb`
- Modify: `docs/web/en/user_guide/tutorials/case_noise_budget.ipynb`
- Modify: `docs/web/ja/user_guide/tutorials/case_noise_budget.ipynb`

- [ ] **Step 1: Rewrite `intro_timeseries` comments around signal meaning**

Target cells/comments such as:

```python
# Calculate the envelope
# Instantaneous frequency
# Calculate transfer function
# Cross-correlation (xcorr)
```

Rewrite them to explain:

- why the analytic signal isolates amplitude/phase modulation
- why instantaneous frequency is meaningful for chirps and drifting lines
- why transfer/correlation are used to ask whether one channel can explain another

- [ ] **Step 2: Rewrite `segment_asd_pipeline` comments around stationarity**

Replace comments like:

```python
# Crop to each segment span
# Calculate ASD for each segment
```

with comments that explain:

- segments isolate intervals with shared operating state
- per-segment ASD prevents non-stationary periods from biasing the baseline

- [ ] **Step 3: Rewrite `case_transfer_function` comments around measured response physics**

Improve comments to explain:

- why coherence is checked before trusting the transfer function
- why fitting is cropped near resonance instead of over the full band
- why `Q` and resonance frequency matter physically for suspension or actuator behavior

- [ ] **Step 4: Rewrite `case_active_damping` comments around modal control**

Improve comments to explain:

- why coordinates are transformed from sensor/actuator space into modal space
- why diagonal modal filters reduce cross-coupling
- why the closed-loop comparison is the relevant test for damping performance

- [ ] **Step 5: Rewrite `case_noise_budget` comments around contribution bookkeeping**

Improve comments to explain:

- why auxiliary channels stand in for candidate physical noise sources
- why coherence/projection estimate explanatory contribution rather than proof of causality
- why quadrature summation is used for approximately independent contributions

- [ ] **Step 6: Mirror each rewrite in Japanese with the same physical meaning**

Validation rule:

- The JA notebook should not be a literal word-for-word translation if detector terminology reads awkwardly.
- The JA and EN versions must, however, encode the same physical rationale and same cautionary notes.

### Task 3: Promote Legacy Control Tutorials into the Public Tutorial Set

**Files:**
- Create: `docs/web/en/user_guide/tutorials/advanced_control_discretization.ipynb`
- Create: `docs/web/ja/user_guide/tutorials/advanced_control_discretization.ipynb`
- Create: `docs/web/en/user_guide/tutorials/advanced_control_basics.ipynb`
- Create: `docs/web/ja/user_guide/tutorials/advanced_control_basics.ipynb`
- Create: `docs/web/en/user_guide/tutorials/advanced_control_modeling.ipynb`
- Create: `docs/web/ja/user_guide/tutorials/advanced_control_modeling.ipynb`
- Modify: `docs/web/en/user_guide/tutorials/index.rst`
- Modify: `docs/web/ja/user_guide/tutorials/index.rst`

- [ ] **Step 1: Copy each legacy control notebook into the public tree with normalized filenames**

Use the legacy notebooks only as seeds:

```text
examples/advanced-methods/tutorial_Control_00_DiscretizationBasics.ipynb
examples/advanced-methods/tutorial_Control_01_Basics.ipynb
examples/advanced-methods/tutorial_Control_02_Modeling.ipynb
```

Do not preserve the old `tutorial_Control_00_*` naming in the public docs.

- [ ] **Step 2: Rewrite comments during promotion**

For each promoted notebook, make sure comments explain:

- discretization: why ZOH vs Tustin changes phase and controller realizability
- basics: why resonance frequency and `Q` set the control difficulty
- modeling: why measured transfer functions are fit near physically dominant poles/zeros

- [ ] **Step 3: Add bilingual titles and metadata**

Add the usual notebook header/title cells and ensure both EN/JA copies include:

- consistent title in `Feature: Task` style
- difficulty / estimated time / audience metadata if the surrounding notebook set uses those fields
- first-cell CI classification tags per `docs/NOTEBOOK_POLICY.md`

- [ ] **Step 4: Add the new tutorials to both tutorial indexes**

Place them under `IV. Advanced Signal Processing` in EN and `IV. 高度な信号処理` in JA unless, during implementation, the content clearly needs a separate follow-up taxonomy change.

Suggested bullets:

```rst
- :doc:`Control Analysis: Discretization Basics <advanced_control_discretization>` ...
- :doc:`Control Analysis: Resonance and Feedback Basics <advanced_control_basics>` ...
- :doc:`Control Analysis: Plant Modeling from Measured Response <advanced_control_modeling>` ...
```

Also update each hidden toctree.

### Task 4: Promote Legacy Physics-Rich Case Studies into the Public Gallery

**Files:**
- Create: `docs/web/en/user_guide/tutorials/case_lockin_detection.ipynb`
- Create: `docs/web/ja/user_guide/tutorials/case_lockin_detection.ipynb`
- Create: `docs/web/en/user_guide/tutorials/case_wiener_filter.ipynb`
- Create: `docs/web/ja/user_guide/tutorials/case_wiener_filter.ipynb`
- Create: `docs/web/en/user_guide/tutorials/case_signal_extraction.ipynb`
- Create: `docs/web/ja/user_guide/tutorials/case_signal_extraction.ipynb`
- Create: `docs/web/en/user_guide/tutorials/case_coupling_analysis.ipynb`
- Create: `docs/web/ja/user_guide/tutorials/case_coupling_analysis.ipynb`
- Modify: `docs/web/en/examples/index.rst`
- Modify: `docs/web/ja/examples/index.rst`

- [ ] **Step 1: Promote the four case studies with public-doc filenames unchanged**

Use these legacy notebooks as seeds:

```text
examples/case-studies/case_lockin_detection.ipynb
examples/case-studies/case_wiener_filter.ipynb
examples/case-studies/case_signal_extraction.ipynb
examples/case-studies/case_coupling_analysis.ipynb
```

- [ ] **Step 2: Rewrite comments so each notebook teaches the physical question**

Required rewrite focus:

- `case_lockin_detection`
  - explain carrier, sideband, and baseband separation
  - explain why mixing + low-pass recovers slow modulation buried under a strong carrier
- `case_wiener_filter`
  - explain why auxiliary witnesses can predict coherent noise but not irreducible noise
  - explain why the matrix inverse is solving a multichannel correlation problem
- `case_signal_extraction`
  - explain why whitening/filtering improve detectability of weak structure in colored noise
  - explain what kind of false features over-filtering can create
- `case_coupling_analysis`
  - explain that estimated coupling is a transfer-path model, not proof of mechanism
  - explain why frequency dependence matters for diagnosing where coupling enters the plant

- [ ] **Step 3: Add the promoted notebooks to the public case-study gallery**

Add concise cards/bullets to `docs/web/{en,ja}/examples/index.rst` with:

- problem statement
- approach summary
- key APIs

Suggested EN labels:

```rst
- Lock-in Detection: Recovering Weak AM/FM Structure
- Wiener Filtering: Coherent Noise Subtraction
- Signal Extraction: Weak Signal Recovery from Colored Noise
- Coupling Analysis: Estimating Transfer Paths Between Channels
```

Also update the hidden toctree in both languages.

### Task 5: Audit Remaining Legacy Notebooks and Prevent Drift

**Files:**
- Modify if needed: `examples/README.md`
- Check only: remaining `examples/basic-new-methods/*.ipynb`
- Check only: remaining `examples/case-studies/*.ipynb`
- Check only: remaining `examples/advanced-methods/*.ipynb`

- [ ] **Step 1: Mark duplicated legacy notebooks as non-authoritative**

At minimum, document that these legacy files are not the published edit targets:

```text
examples/case-studies/case_transfer_function.ipynb
examples/case-studies/case_active_damping.ipynb
examples/case-studies/case_noise_budget.ipynb
```

- [ ] **Step 2: Leave non-promoted notebooks for a later pass**

Explicitly defer, rather than half-fix:

- `case_trend_analysis.ipynb`
- `case_bootstrap_spectral.ipynb`
- `case_response_analysis.ipynb`
- `tutorial_Control_03_Design.ipynb`
- `tutorial_ShortTimeLaplaceTransformation.ipynb`

### Task 6: Validate, Build, and Close the Audit Item

**Files:**
- Modify: `docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md`

- [ ] **Step 1: Validate notebook JSON after every edited/promoted file**

Run a focused JSON parse check in the required environment:

```bash
conda run -n gwexpy python -c "import json, pathlib; [json.loads(path.read_text()) for path in pathlib.Path('docs/web').rglob('*.ipynb')]"
```

Expected:

- no JSON parse failures
- no truncated notebook edits

- [ ] **Step 2: Build the docs without notebook execution**

Run:

```bash
conda run -n gwexpy sphinx-build -b html -D nbsphinx_execute=never docs docs/_build/html
```

Expected:

- Sphinx build succeeds
- new tutorial pages render in EN/JA
- `tutorials/index.html` and `examples/index.html` show the promoted entries

- [ ] **Step 3: Spot-check the public pages for rubric compliance**

Review at least these generated pages:

- `intro_timeseries`
- `segment_asd_pipeline`
- `case_transfer_function`
- `case_active_damping`
- `case_noise_budget`
- one promoted control tutorial
- one promoted case study

Acceptance check:

- comments explain physical intent where analysis choices occur
- comments do not become long prose blocks that interrupt notebook flow

- [ ] **Step 4: Update the internal audit report**

Change `14-10` from `要確認` to a resolved state only after the public notebook rewrites and index integration land.

Suggested note:

```md
✅修正済み（公開 tutorial / case-study notebook のコードコメントに、解析手順の物理的意図・選定理由・防ぎたい失敗モードを追記。併せて legacy examples から有用 notebook を公開 docs へ昇格し、正本管理を `docs/web/.../tutorials/` へ統一）
```

- [ ] **Step 5: Commit**

```bash
git add docs/developers/guides/notebook_physics_comment_rubric.md \
        docs/NOTEBOOK_POLICY.md \
        examples/README.md \
        docs/web/en/user_guide/tutorials \
        docs/web/ja/user_guide/tutorials \
        docs/web/en/examples/index.rst \
        docs/web/ja/examples/index.rst \
        docs_internal/analysis/webpage/統合監査レポート_計214件の指摘.md
git commit -m "docs: add physics-intent notebook comments"
```

## Success Criteria

- Public notebook ownership is unambiguous: published notebooks are edited in `docs/web/{en,ja}/user_guide/tutorials/`.
- A durable rubric exists for future notebook comment rewrites.
- Priority public notebooks now explain the physical reason behind filters, transforms, segmentation, fitting, and noise-projection steps.
- At least three control/tutorial notebooks and four physics-rich case studies are promoted from legacy `examples/` into the public docs tree.
- EN and JA public indexes expose the promoted notebooks symmetrically.
- `examples/README.md` no longer contradicts the docs build or download paths.
- Section `14-10` can be closed with a concrete remediation note rather than `要確認`.
