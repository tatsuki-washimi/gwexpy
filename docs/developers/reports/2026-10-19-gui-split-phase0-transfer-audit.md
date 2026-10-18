# Phase 0 — GWexpy GUI extraction completeness / deletion gate audit

**Date:** 2026-10-19 (JST). **Status:** source-traceability audit performed; cross-repo fix PR opened and limited CI passed; **pre-deletion transfer acceptance is still pending**.

## Immutable bases and methodology

- GWexpy extraction source **G0**: [`61fc5dc7453ae7165dc711f13951625552eca88d`](https://github.com/tatsuki-washimi/gwexpy/commit/61fc5dc7453ae7165dc711f13951625552eca88d) (provenance recorded in [#645](https://github.com/tatsuki-washimi/gwexpy/issues/645); split 2026-07-11).
- GWexpy active branch snapshot **G1**: [`10311d8dcebd8938522a4e07d78fb8dfa73b3145`](https://github.com/tatsuki-washimi/gwexpy/commit/10311d8dcebd8938522a4e07d78fb8dfa73b3145).
- pyaggui active branch snapshot **P1**: [`cd715f72ed9ce75dd25fe0743bf3dd8f05c3601b`](https://github.com/tatsuki-washimi/pyaggui/commit/cd715f72ed9ce75dd25fe0743bf3dd8f05c3601b).
- GitHub recursive trees were returned with `truncated=false`: G0 2181 entries, G1 10676, P1 77. We mapped **every tracked G1 file** under `gwexpy/gui/`, `tests/gui/`, `docs/developers/design/gui/`, plus `tests/e2e/test_gui_smoke.py` to the corresponding pyaggui path. We separately inspected `tests/nds/test_gui_nds_smoke.py` and later-added `tests/plot/test_gui_plotting_normalize_series.py`.
- GitHub compare `G0...G1` reports 946 commits but only the first 300 changed files: its file list is **truncated** and is **not** the audit inventory. Full recursive Git tree blob SHA comparison supplies the file-level population and change detection. Manual content inspection was performed on the 14 G1 files modified since G0, plus the standalone smoke/normalization tests, loader and design documentation. SHA equality is exact byte equality; differing SHA **does not imply a defect or missing functionality**.

## Counts / coverage

| Area | G1 files | Byte-identical to P1 target | Different blob | No direct target |
| --- | ---: | ---: | ---: | ---: |
| `gwexpy/gui/` | 32 | 19 | 12 | 1 |
| `tests/gui/` | 21 | 3 | 18 | 0 |
| GUI design docs | 2 | 1 | 1 | 0 |
| `tests/e2e/test_gui_smoke.py` → `tests/gui/test_gui_smoke.py` | 1 | 0 | 1 | 0 |
| **Mapped total** | **56** | **23** | **32** | **1** |

Across G0→G1, **14 mapped files changed** (the remaining 42 are byte-identical to the pre-extraction source). G0 contained **57** in-scope tracked files, G1 contains **58**; G1 added `tests/plot/test_gui_plotting_normalize_series.py` after the split; there are no G0-only deleted paths within this scope.

## Reviewed post-extraction changes (14 files)

| Source area | Findings | Transfer disposition |
| --- | --- | --- |
| `gwexpy/gui/plotting/normalize.py` | PR [#795](https://github.com/tatsuki-washimi/gwexpy/pull/795) changed ambiguous NumPy truth-value use to explicit `is None` fallback; P1 still has old expression. This is a real fix absent from P1. | **Ported in [pyaggui PR #3](https://github.com/tatsuki-washimi/pyaggui/pull/3)** together with six targeted regressions (five transferred cases plus empty-array). Python 3.11 and 3.12 headless workflow jobs succeeded. **Wait for review/merge.** |
| `gwexpy/gui/loaders/loaders.py` and `gwexpy/gui/README.md` | FFL reader-input support claim corrected in G1 docs. pyaggui loader still describes `.ffl` as supported, although its explicit extension map lacks an FFL entry. This is a documentation alignment, not evidence of the outcome of every possible GWpy reader. | Loader docstring adjusted in pyaggui PR #3; pyaggui standalone user-guide/README text needs its own editorial review rather than byte-for-byte old README transfer. |
| `gwexpy/gui/data_sources.py`, `nds/audio_thread.py`, `nds/buffer.py`, `nds/util.py`, `ui/_tab_builders.py`, `ui/excitation_manager.py`, `ui/plot_renderer.py` | Observed G0→G1 edits were insertion of blank lines/import formatting; no changed executable expressions in those compared deltas. | No code port indicated by these **specific G0→G1 edits**. P1 may intentionally diverge for independent refactoring; inspect any future behavioral merge separately. |
| `tests/gui/integration/test_pyautogui_smoke.py`, `test_channel_browser_dialog.py`, `test_gui_main_window.py`, `test_main_window_channel_browser.py` | G0→G1 differences were line wraps/formatting of already existing test statements and function declarations. | P1 already contains counterpart tests. Functional test parity was **not established** by blob comparison; pyaggui GUI-suite execution belongs to its own CI review. |

## Unmapped, indirect or separately owned test/doc assets

| GWexpy G1 source | Target / observation | State |
| --- | --- | --- |
| `tests/plot/test_gui_plotting_normalize_series.py` | G1-only file with **five** targeted regressions; pyaggui [PR #3](https://github.com/tatsuki-washimi/pyaggui/pull/3) creates six (adds empty-array case), outside `tests/gui/` to avoid autouse `qtbot` fixtures. | **Transfer PR created and targeted tests green; pending merge.** |
| `tests/nds/test_gui_nds_smoke.py` | Live NDS GUI smoke verifies connection/acquisition/plot. No same-purpose test was found in pyaggui tree; `nds_backend` and `GWEXPY_NDS_CHANNELS` were not found in pyaggui repository code search. Existing pyaggui GUI data backend tests use stubs; those **do not establish live NDS parity**. | **Outstanding preservation decision:** port/modernize an opt-in live NDS smoke with fixture and controlled network configuration, or explicitly accept its removal with documented rationale. This is **not covered** by PR #3. |
| `tests/e2e/test_gui_smoke.py` | Corresponding `tests/gui/test_gui_smoke.py` exists in pyaggui but carries a module-level skip; same skip existed in GWexpy G1. | Static transfer verified. **Runtime GUI smoke remains skipped in both**, so no positive integration pass is established. |
| `gwexpy/gui/README.md` | No `src/pyaggui/README.md`; pyaggui root `README.md` and `docs/user_guide_{en,ja}.md` are independent canonical documentation candidates. | Canonical location exists, but **manual content completeness review remains** (including split-specific stale `gwexpy.gui`/path text). |
| `docs/developers/design/gui/` | Both files have `docs/design/` counterparts. `implementation_map.md` differs in namespace text, `reference_analysis.md` has identical blob SHA. | Current design ownership available in pyaggui; update residual pre-split nomenclature in its current docs as a separate documentation fix. Historical GWexpy audit artifacts remain retained. |

## Separate GWexpy Studio boundary (not pyaggui)

- Located [GWexpy Studio](https://github.com/tatsuki-washimi/gwexpy-studio) and [GWexpy Studio Dev](https://github.com/tatsuki-washimi/gwexpy-studio-dev) as distinct repositories.
- `gwexpy-studio/pyproject.toml` declares `gwexpy==0.2.0` and a `gwexpy_studio.ui.app:main` entry point; `gwexpy-studio-dev` declares `gwexpy>=0.1.14,<0.2.0`. These are different development baselines and **neither should be silently used to choose a v0.2.6 dependency range**.
- GitHub repository code search for literal `gwexpy.gui` found no matches in either Studio repository. This is **search evidence, not exhaustive AST/runtime import verification**. Before deletion, review active code and run Studio's own dependency-boundary tests where applicable.

## Actual pyaggui transfer work and CI

[pyaggui PR #3](https://github.com/tatsuki-washimi/pyaggui/pull/3) is **Draft and unmerged**. Its initial head is `331b28985f2d2f4472b62a518db91caddcae7e74`.

| File | Change |
| --- | --- |
| `src/pyaggui/plotting/normalize.py` | Explicit `None` fallback (GWexpy #765-C1) |
| `tests/test_plotting_normalize_series.py` | 6 NumPy-only regression tests from pure module via `runpy.run_path`; avoids requiring a Qt QApplication |
| `src/pyaggui/loaders/loaders.py` | FFL reader-input description correction |
| `.github/workflows/normalize-headless.yml` | Targeted tests on Python 3.11 and 3.12 (isolated NumPy/pytest CI, not a full GUI suite) |

CI: [run 39139855457](https://github.com/tatsuki-washimi/pyaggui/actions/runs/39139855457) and [run 39139826628](https://github.com/tatsuki-washimi/pyaggui/actions/runs/39139826628) both concluded **success**; each reports both 3.11 and 3.12 jobs successful. This only validates the isolated normalization tests and workflow, not Qt/NDS/fresh-package installation.

## Phase 0 decision and deletion preconditions

**Verified completed:** source path population, immutable hash crosswalk for 56 mapped files, static review of 14 post-extraction changes, identification of C1 gap, creation of pyaggui transfer PR #3, and targeted cross-version CI pass. FFL documentation correction prepared. Independent Studio repository identifiers and package declarations inspected.

**Not yet accepted:** pyaggui PR #3 review/merge; handling of unmatched real-NDS smoke and its fixture; current standalone pyaggui docs completeness; full clean wheel/sdist + Qt/GUI integration and supported GWexpy version contract; runtime Studio boundary check. These are concrete gates for [#645](https://github.com/tatsuki-washimi/gwexpy/issues/645), not grounds to discard prior completed static audit.

**Recommendation:** do **not** delete `gwexpy/gui/` yet. First accept/merge the targeted pyaggui transfer and decide preservation or explicit deferral of live-NDS smoke. Once those are documented, proceed with #645 PR A/B deletion/CI. Keep GWexpy `gwexpy.plot`, generic NDS backend, `gwexpy.cli.qtransform`, and DiagGUI XML I/O; their source is distinct from the legacy GUI.

## Full path-to-path SHA crosswalk

SHA strings below are complete blob IDs; `G0→G1` indicates content change since extraction; `G1↔P1` indicates byte equivalence only. A different SHA can reflect import namespace or formatter changes, rather than a missing port.

| GWexpy source | pyaggui destination | G0→G1 | G1↔P1 | G0 blob | G1 blob | P1 blob |
| --- | --- | --- | --- | --- | --- | --- |
| `docs/developers/design/gui/implementation_map.md` | `docs/design/implementation_map.md` | unchanged | different blob | `6b811d73b48679ae166583382df7fd023d4619f9` | `6b811d73b48679ae166583382df7fd023d4619f9` | `bdbb927532744bcbcd76625f44b3f3d5a016a55d` |
| `docs/developers/design/gui/reference_analysis.md` | `docs/design/reference_analysis.md` | unchanged | same blob | `2f65972c1929d17b59a2e9f8b5bbc0df301e6712` | `2f65972c1929d17b59a2e9f8b5bbc0df301e6712` | `2f65972c1929d17b59a2e9f8b5bbc0df301e6712` |
| `gwexpy/gui/__init__.py` | `src/pyaggui/__init__.py` | unchanged | same blob | `8aee47e624eee4489d166faf0a3cd62accd0e3e1` | `8aee47e624eee4489d166faf0a3cd62accd0e3e1` | `8aee47e624eee4489d166faf0a3cd62accd0e3e1` |
| `gwexpy/gui/__main__.py` | `src/pyaggui/__main__.py` | unchanged | same blob | `bb3bc0f4c0661bad41dc494d00bc1480dd41dc47` | `bb3bc0f4c0661bad41dc494d00bc1480dd41dc47` | `bb3bc0f4c0661bad41dc494d00bc1480dd41dc47` |
| `gwexpy/gui/AGENTS.md` | `src/pyaggui/AGENTS.md` | unchanged | same blob | `00bf57a7d601aaf1b93f6b2c4dcca74b3528e69d` | `00bf57a7d601aaf1b93f6b2c4dcca74b3528e69d` | `00bf57a7d601aaf1b93f6b2c4dcca74b3528e69d` |
| `gwexpy/gui/channel_handling.md` | `src/pyaggui/channel_handling.md` | unchanged | same blob | `403f2bb74ca29c7a8f34686d004520cdf50a65d4` | `403f2bb74ca29c7a8f34686d004520cdf50a65d4` | `403f2bb74ca29c7a8f34686d004520cdf50a65d4` |
| `gwexpy/gui/CLAUDE.md` | `src/pyaggui/CLAUDE.md` | unchanged | same blob | `055fd85fc240b4916ecf8ae60cb5e00df7bbd7e6` | `055fd85fc240b4916ecf8ae60cb5e00df7bbd7e6` | `055fd85fc240b4916ecf8ae60cb5e00df7bbd7e6` |
| `gwexpy/gui/data_sources.py` | `src/pyaggui/data_sources.py` | changed | different blob | `7e006d78ae6b72c17bacaa51ff2e69ad4598f773` | `00af8778d80c44177956504aeda407645a919daf` | `7e006d78ae6b72c17bacaa51ff2e69ad4598f773` |
| `gwexpy/gui/engine.py` | `src/pyaggui/engine.py` | unchanged | same blob | `da4586629ffa0465741a64c232144581899c9ee1` | `da4586629ffa0465741a64c232144581899c9ee1` | `da4586629ffa0465741a64c232144581899c9ee1` |
| `gwexpy/gui/excitation/generator.py` | `src/pyaggui/excitation/generator.py` | unchanged | same blob | `88142eee3b267235a72a5dd96efba4b306f5d1c2` | `88142eee3b267235a72a5dd96efba4b306f5d1c2` | `88142eee3b267235a72a5dd96efba4b306f5d1c2` |
| `gwexpy/gui/excitation/params.py` | `src/pyaggui/excitation/params.py` | unchanged | same blob | `f33be827b3982ae10fe45840fd305c72cd51d33b` | `f33be827b3982ae10fe45840fd305c72cd51d33b` | `f33be827b3982ae10fe45840fd305c72cd51d33b` |
| `gwexpy/gui/loaders/__init__.py` | `src/pyaggui/loaders/__init__.py` | unchanged | same blob | `bdec2fc8c46f03040519d76757f9e7c26dec2a31` | `bdec2fc8c46f03040519d76757f9e7c26dec2a31` | `bdec2fc8c46f03040519d76757f9e7c26dec2a31` |
| `gwexpy/gui/loaders/loaders.py` | `src/pyaggui/loaders/loaders.py` | changed | different blob | `6df8209b7e33737a1de8f41129b874c49307c255` | `653a47e3e7b04797fab2b1d4099b43590744377d` | `0abc4280a73c26a14853d314d8d7ae8c9fd0485d` |
| `gwexpy/gui/loaders/products.py` | `src/pyaggui/loaders/products.py` | unchanged | same blob | `21ca38d680ddca7d46373452459cebea2c5c82c2` | `21ca38d680ddca7d46373452459cebea2c5c82c2` | `21ca38d680ddca7d46373452459cebea2c5c82c2` |
| `gwexpy/gui/nds/__init__.py` | `src/pyaggui/nds/__init__.py` | unchanged | same blob | `bdec2fc8c46f03040519d76757f9e7c26dec2a31` | `bdec2fc8c46f03040519d76757f9e7c26dec2a31` | `bdec2fc8c46f03040519d76757f9e7c26dec2a31` |
| `gwexpy/gui/nds/audio_thread.py` | `src/pyaggui/nds/audio_thread.py` | changed | different blob | `e3bea87a212ef3d9bd54c532ac2b9dd2bed21eb6` | `c12c88aaeb269b5b5e86eda154b30d8f6e22a70a` | `e3bea87a212ef3d9bd54c532ac2b9dd2bed21eb6` |
| `gwexpy/gui/nds/buffer.py` | `src/pyaggui/nds/buffer.py` | changed | different blob | `0b343760e68579412d568127ca30723e3fae9497` | `ad3cfb7972a17376cae1a194edc91a167cfbdc5f` | `0b343760e68579412d568127ca30723e3fae9497` |
| `gwexpy/gui/nds/cache.py` | `src/pyaggui/nds/cache.py` | unchanged | same blob | `04b4016846bca5bf0e7bcc645a1de69cba411f2e` | `04b4016846bca5bf0e7bcc645a1de69cba411f2e` | `04b4016846bca5bf0e7bcc645a1de69cba411f2e` |
| `gwexpy/gui/nds/nds_thread.py` | `src/pyaggui/nds/nds_thread.py` | unchanged | same blob | `c5788886592ab2639bf88c519931d60fbc3cec10` | `c5788886592ab2639bf88c519931d60fbc3cec10` | `c5788886592ab2639bf88c519931d60fbc3cec10` |
| `gwexpy/gui/nds/sim_thread.py` | `src/pyaggui/nds/sim_thread.py` | unchanged | same blob | `d3b64be10b254b445e415a5b61bc17a4c6ca1e35` | `d3b64be10b254b445e415a5b61bc17a4c6ca1e35` | `d3b64be10b254b445e415a5b61bc17a4c6ca1e35` |
| `gwexpy/gui/nds/util.py` | `src/pyaggui/nds/util.py` | changed | different blob | `3967797b7fd485279fa0053cad97b01ecbd72e5a` | `611bb85d1e0af2cdf464177f22025db1481af8c4` | `3967797b7fd485279fa0053cad97b01ecbd72e5a` |
| `gwexpy/gui/plotting/__init__.py` | `src/pyaggui/plotting/__init__.py` | unchanged | same blob | `bdec2fc8c46f03040519d76757f9e7c26dec2a31` | `bdec2fc8c46f03040519d76757f9e7c26dec2a31` | `bdec2fc8c46f03040519d76757f9e7c26dec2a31` |
| `gwexpy/gui/plotting/normalize.py` | `src/pyaggui/plotting/normalize.py` | changed | different blob | `649c8b7cc84de83d4d00461994de25ead13eff8f` | `6b6addaa8f878a55ab12c81159c1c70a419d157f` | `649c8b7cc84de83d4d00461994de25ead13eff8f` |
| `gwexpy/gui/pyaggui.py` | `src/pyaggui/pyaggui.py` | unchanged | different blob | `f2f3408002b234ba836d75df6363ffdde852d03a` | `f2f3408002b234ba836d75df6363ffdde852d03a` | `574cea931bac39e4a6c2921e4e459e93375e9f41` |
| `gwexpy/gui/README.md` | `src/pyaggui/README.md` | changed | no direct path | `7ff8940bf22739ab47e74928a9462913f76e4e15` | `fb61631c02e86dedfad13c2171906da4133baac2` | `—` |
| `gwexpy/gui/streaming.py` | `src/pyaggui/streaming.py` | unchanged | same blob | `5afa56ad81043b681f317b3c9b612151d0c1e2e0` | `5afa56ad81043b681f317b3c9b612151d0c1e2e0` | `5afa56ad81043b681f317b3c9b612151d0c1e2e0` |
| `gwexpy/gui/ui/__init__.py` | `src/pyaggui/ui/__init__.py` | unchanged | same blob | `bdec2fc8c46f03040519d76757f9e7c26dec2a31` | `bdec2fc8c46f03040519d76757f9e7c26dec2a31` | `bdec2fc8c46f03040519d76757f9e7c26dec2a31` |
| `gwexpy/gui/ui/_tab_builders.py` | `src/pyaggui/ui/_tab_builders.py` | changed | different blob | `74586bdfeaf15bc52ec3171265a27ecadf877539` | `0237f53f16d253961f31041154b8715afb01a477` | `94d1da152eed8bd477bcd27ba4cc3e77882195a9` |
| `gwexpy/gui/ui/channel_browser.py` | `src/pyaggui/ui/channel_browser.py` | unchanged | same blob | `3da873cd683579aba522725ede22014642b96443` | `3da873cd683579aba522725ede22014642b96443` | `3da873cd683579aba522725ede22014642b96443` |
| `gwexpy/gui/ui/excitation_manager.py` | `src/pyaggui/ui/excitation_manager.py` | changed | different blob | `cc359413b09f26f4ce7fb371c747cc1da5dcd309` | `6691a854f8168bbaa92649a9f90d13ba74f0bac4` | `3cdaba8bceca94cd7f5ff569531718f54a6a700e` |
| `gwexpy/gui/ui/graph_panel.py` | `src/pyaggui/ui/graph_panel.py` | unchanged | different blob | `97d14c520289b18709e4dd09f5eaf262ac5a3f1e` | `97d14c520289b18709e4dd09f5eaf262ac5a3f1e` | `a55c0d6a31dfc1c69c750a0731358f32accc4ee9` |
| `gwexpy/gui/ui/main_window.py` | `src/pyaggui/ui/main_window.py` | unchanged | different blob | `3039d939e55e1c2ccb34193737ae278ba3e9bef9` | `3039d939e55e1c2ccb34193737ae278ba3e9bef9` | `955bce36de15616e0c24724ea182554a9c92dafc` |
| `gwexpy/gui/ui/plot_renderer.py` | `src/pyaggui/ui/plot_renderer.py` | changed | different blob | `b0ff9eb37a0167c9a4593862eb147ebc47811991` | `08a32114ef071622fa11eabe0767411dd0906a9a` | `907b183c020d8fccd033e486a449f997f24aee75` |
| `gwexpy/gui/ui/tabs.py` | `src/pyaggui/ui/tabs.py` | unchanged | same blob | `8f1ca21fb3e850dec7b9fa80e7d82e9335b45bdc` | `8f1ca21fb3e850dec7b9fa80e7d82e9335b45bdc` | `8f1ca21fb3e850dec7b9fa80e7d82e9335b45bdc` |
| `tests/e2e/test_gui_smoke.py` | `tests/gui/test_gui_smoke.py` | unchanged | different blob | `1dc576408ccb79ed478f5a573f2c06539d54393f` | `1dc576408ccb79ed478f5a573f2c06539d54393f` | `6280713485a877d2a8bfbda0640ac957fd1ebd72` |
| `tests/gui/archive/README.md` | `tests/gui/archive/README.md` | unchanged | same blob | `c19dd2f12558017fb478453f5fb8b80606bf102d` | `c19dd2f12558017fb478453f5fb8b80606bf102d` | `c19dd2f12558017fb478453f5fb8b80606bf102d` |
| `tests/gui/conftest.py` | `tests/gui/conftest.py` | unchanged | different blob | `3e9fdebcf117ee71a04c5f88a752910c97e0147e` | `3e9fdebcf117ee71a04c5f88a752910c97e0147e` | `9842f0821a956d030a5634a35a6c1f165c490c66` |
| `tests/gui/integration/test_channel_config.py` | `tests/gui/integration/test_channel_config.py` | unchanged | different blob | `8693c3c04716bd1da80484767193fd5c6216612e` | `8693c3c04716bd1da80484767193fd5c6216612e` | `ec993729bb16d02873c653abc3d83a775777aff8` |
| `tests/gui/integration/test_excitation_controls.py` | `tests/gui/integration/test_excitation_controls.py` | unchanged | different blob | `70106d6e8b86b9f21fd352f3d6a2085819a85e1f` | `70106d6e8b86b9f21fd352f3d6a2085819a85e1f` | `fcd03a352a251b9b3978e74ba2106b63d1f46969` |
| `tests/gui/integration/test_main_window_flow.py` | `tests/gui/integration/test_main_window_flow.py` | unchanged | different blob | `78895ffa448e6fce42d5a314b25418344c0ee2f3` | `78895ffa448e6fce42d5a314b25418344c0ee2f3` | `84b80e5c99b5123dba7dd22a71af4048af12a023` |
| `tests/gui/integration/test_measurement_params.py` | `tests/gui/integration/test_measurement_params.py` | unchanged | different blob | `f5e958d5b7d0dc3d435b0b696b5bb8955ce686e4` | `f5e958d5b7d0dc3d435b0b696b5bb8955ce686e4` | `9e4e940dbac58174d0e6458fd7d72c11fbbc3c6c` |
| `tests/gui/integration/test_pyautogui_smoke.py` | `tests/gui/integration/test_pyautogui_smoke.py` | changed | different blob | `756c861846520011ef600f8ac3aab782239211ee` | `a4d99283b4135b934c2e0750b77aaeb2d25f92e3` | `87017e3d88dc27d454faebf7ca3d6ae5a2844fca` |
| `tests/gui/integration/test_simulation_acquisition.py` | `tests/gui/integration/test_simulation_acquisition.py` | unchanged | different blob | `f52209e630017f0127ed0b5b494456ae8ca3fb49` | `f52209e630017f0127ed0b5b494456ae8ca3fb49` | `2b2dd55fd65190b71bbe3139d804cf37281362be` |
| `tests/gui/integration/test_simulation_partial_channels.py` | `tests/gui/integration/test_simulation_partial_channels.py` | unchanged | different blob | `960aa38b076297d637fe71f8e8553c4f79291a97` | `960aa38b076297d637fe71f8e8553c4f79291a97` | `cfc328d133afe9551b18ad9622e5b1aa77c63805` |
| `tests/gui/test_accumulator_delay.py` | `tests/gui/test_accumulator_delay.py` | unchanged | different blob | `2cc9c6420eee700a2ee11bb80b3de67e5818ea73` | `2cc9c6420eee700a2ee11bb80b3de67e5818ea73` | `61b40db393055c28a91394e7e866210198209af6` |
| `tests/gui/test_channel_browser_dialog.py` | `tests/gui/test_channel_browser_dialog.py` | changed | different blob | `52b79ede952ddf4b000d330fe3c7b308fcbd31a4` | `f25f56a3fd4e00491d15a7252fc616a4b6af4b02` | `3c05d74e01120b54ccc36593206936a7cec3772a` |
| `tests/gui/test_excitation_manager.py` | `tests/gui/test_excitation_manager.py` | unchanged | different blob | `7e2f4a4cfb19288299826cd129365a25acbc91a5` | `7e2f4a4cfb19288299826cd129365a25acbc91a5` | `efa08a8e2a6fa6cc2300b60fb16c7aa954c13b98` |
| `tests/gui/test_gui_data_backend.py` | `tests/gui/test_gui_data_backend.py` | unchanged | different blob | `af10800796072026260681dcae5ce4470150588f` | `af10800796072026260681dcae5ce4470150588f` | `109301627663b66f2decebebe53f62766f0779b1` |
| `tests/gui/test_gui_main_window.py` | `tests/gui/test_gui_main_window.py` | changed | different blob | `06952c0702c8b1626f31d5fa2f22d946fb728d67` | `5333a6cbf19ff8e451c051f1da7019cb468f8c68` | `265e7860ce2053fe0b1e03a99101d61c9d4ca01c` |
| `tests/gui/test_gui_payload_contracts.py` | `tests/gui/test_gui_payload_contracts.py` | unchanged | different blob | `13d953e3959fd03d9871584e125d73e4d2dc096e` | `13d953e3959fd03d9871584e125d73e4d2dc096e` | `af0db3bcc4e183d5bfba844fbc1933019320c2bd` |
| `tests/gui/test_gui_result_tab.py` | `tests/gui/test_gui_result_tab.py` | unchanged | different blob | `e3313a45a87d052bca0571cbcd2ab9fde22bf77f` | `e3313a45a87d052bca0571cbcd2ab9fde22bf77f` | `c90b0a95b3319547e2cb8e895396d471a5b6b03f` |
| `tests/gui/test_main_window_channel_browser.py` | `tests/gui/test_main_window_channel_browser.py` | changed | different blob | `ba977b178b5792f5bed4ef5787f473fb0cf41041` | `b912884d6857fe2dc3fef8662c7419aec6114cdb` | `bdc1a6068d8327f00a54c4401137a0da9dc31904` |
| `tests/gui/test_main_window_excitation_integration.py` | `tests/gui/test_main_window_excitation_integration.py` | unchanged | same blob | `31ed1b1e36c6a7d320ff0ece5d586a259f4905f7` | `31ed1b1e36c6a7d320ff0ece5d586a259f4905f7` | `31ed1b1e36c6a7d320ff0ece5d586a259f4905f7` |
| `tests/gui/test_plot_renderer.py` | `tests/gui/test_plot_renderer.py` | unchanged | different blob | `85666fe61d6c51dddb62a3bc18318ecf4cf8dae9` | `85666fe61d6c51dddb62a3bc18318ecf4cf8dae9` | `f4190f34f784122f765751a9612f7587bd71dde4` |
| `tests/gui/test_pyqtgraph_deleted_guard.py` | `tests/gui/test_pyqtgraph_deleted_guard.py` | unchanged | different blob | `f5003dd6be28f2c70f3f9a0835416b10646650ea` | `f5003dd6be28f2c70f3f9a0835416b10646650ea` | `145bca10d3f97ea48f6e029c6de2373ff95a370f` |
| `tests/gui/test_visual_stability.py` | `tests/gui/test_visual_stability.py` | unchanged | same blob | `b9e13deecba0cd452911798fc669558e5a0bb81a` | `b9e13deecba0cd452911798fc669558e5a0bb81a` | `b9e13deecba0cd452911798fc669558e5a0bb81a` |

