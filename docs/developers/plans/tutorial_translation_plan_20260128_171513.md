# Tutorial Translation Plan (Japanese → English)

**Created:** 2026-01-28 17:15:13 JST
**Status:** Planning Phase

## 📊 Current Status

### Japanese Tutorials (Complete)
- **Total:** 19 tutorials, 410 cells
- **Location:** `docs/web/ja/guide/tutorials/*.ipynb`

### English Tutorials (Stubs Only)
- **Total:** 19 placeholder .md files
- **Location:** `docs/web/en/guide/tutorials/*.md`
- **Status:** All marked as "under translation"

## 📈 Tutorial Breakdown by Category

### INTRO系 (Fundamentals) - 6 tutorials, 176 cells
| Tutorial | Cells | Markdown | Code | Priority |
|----------|-------|----------|------|----------|
| `intro_timeseries.ipynb` | 39 | 19 | 20 | ★★★★★ |
| `intro_frequencyseries.ipynb` | 24 | 12 | 12 | ★★★★☆ |
| `intro_plotting.ipynb` | 19 | 9 | 10 | ★★★★☆ |
| `intro_spectrogram.ipynb` | 18 | 10 | 8 | ★★★☆☆ |
| `intro_mapplotting.ipynb` | 8 | 5 | 3 | ★★☆☆☆ |
| `intro_interop.ipynb` | 68 | 38 | 30 | ★★★☆☆ (largest) |

### ADVANCED系 (Applied Methods) - 6 tutorials, 97 cells
| Tutorial | Cells | Markdown | Code |
|----------|-------|----------|------|
| `advanced_arima.ipynb` | 24 | 13 | 11 |
| `advanced_bruco.ipynb` | 17 | 5 | 12 |
| `advanced_correlation.ipynb` | 19 | 7 | 12 |
| `advanced_fitting.ipynb` | 17 | 8 | 9 |
| `advanced_hht.ipynb` | 12 | 7 | 5 |
| `advanced_peak_detection.ipynb` | 8 | 4 | 4 |

### CASE系 (Use Cases) - 3 tutorials, 32 cells
| Tutorial | Cells | Markdown | Code |
|----------|-------|----------|------|
| `case_active_damping.ipynb` | 10 | 5 | 5 |
| `case_noise_budget.ipynb` | 12 | 6 | 6 |
| `case_transfer_function.ipynb` | 10 | 5 | 5 |

### MATRIX系 (Matrix Operations) - 3 tutorials, 63 cells
| Tutorial | Cells | Markdown | Code |
|----------|-------|----------|------|
| `matrix_timeseries.ipynb` | 27 | 14 | 13 |
| `matrix_frequencyseries.ipynb` | 19 | 10 | 9 |
| `matrix_spectrogram.ipynb` | 17 | 9 | 8 |

### FIELD系 (Scalar Fields) - 1 tutorial, 42 cells
| Tutorial | Cells | Markdown | Code |
|----------|-------|----------|------|
| `field_scalar_intro.ipynb` | 42 | 21 | 21 |

## 🎯 Translation Phases

### Phase 1: INTRO (Fundamentals) - HIGHEST PRIORITY
**Goal:** Provide essential tutorials for new users

1. ✅ `intro_timeseries.ipynb` (39 cells) - Most fundamental
2. ✅ `intro_frequencyseries.ipynb` (24 cells)
3. ✅ `intro_plotting.ipynb` (19 cells)
4. ✅ `intro_spectrogram.ipynb` (18 cells)
5. ✅ `intro_mapplotting.ipynb` (8 cells) - Smallest intro
6. ✅ `intro_interop.ipynb` (68 cells) - Comprehensive interoperability guide

**Rationale:** INTRO tutorials are the first touchpoint for users. Completing these enables international users to get started with gwexpy.

### Phase 2: CASE (Real-World Examples)
**Goal:** Show practical applications

- `case_active_damping.ipynb` (10 cells)
- `case_noise_budget.ipynb` (12 cells)
- `case_transfer_function.ipynb` (10 cells)

**Rationale:** Demonstrates gwexpy's value in real control/physics scenarios.

### Phase 3: FIELD (Advanced Data Structures)
**Goal:** Document ScalarField API

- `field_scalar_intro.ipynb` (42 cells)

**Rationale:** ScalarField is a unique feature worth highlighting early.

### Phase 4: ADVANCED (Specialized Methods)
**Goal:** Cover advanced signal processing

- All 6 `advanced_*.ipynb` tutorials (97 cells total)

**Rationale:** Users need fundamentals before diving into advanced methods.

### Phase 5: MATRIX (Multi-Channel Operations)
**Goal:** Document matrix/multi-channel workflows

- All 3 `matrix_*.ipynb` tutorials (63 cells total)

**Rationale:** Matrix operations build on understanding of basic Series types.

## 🔧 Translation Guidelines

### Markdown Cells
- Translate all prose into natural, technical English
- Preserve structure (headers, lists, code blocks)
- Maintain LaTeX equations unchanged
- Keep hyperlinks and cross-references

### Code Cells
- **Code:** Keep unchanged (Python is universal)
- **Comments:** Translate to English
- **Docstrings:** Keep unchanged (they reference source code)
- **Output:** Preserve as-is (no re-execution needed)

### Terminology Consistency
Create a glossary for key terms:
- 時系列 → TimeSeries
- 周波数系列 → FrequencySeries
- スペクトログラム → Spectrogram
- パワースペクトル密度 → Power Spectral Density (PSD)
- 伝達関数 → Transfer Function
- ノイズバジェット → Noise Budget
- etc.

### Quality Assurance Checklist (per tutorial)
- [ ] All markdown cells translated
- [ ] All code comments translated
- [ ] JSON structure valid
- [ ] Notebook metadata preserved
- [ ] No broken internal links
- [ ] Sphinx build passes
- [ ] Notebook executes without errors (optional: use `pytest --nbmake`)
- [ ] Ruff/mypy checks pass

## 📝 Workflow (per tutorial)

1. **Read** Japanese `.ipynb` file
2. **Translate** markdown cells using LLM (GPT-4/Claude)
3. **Translate** code comments
4. **Save** English `.ipynb` to `docs/web/en/guide/tutorials/`
5. **Remove** corresponding placeholder `.md` stub
6. **Test** with `jupyter nbconvert --execute` (optional)
7. **Build** with `sphinx-build` to verify
8. **Commit** with message: `docs(tutorials): translate <name> to English`

## 🚀 Starting Point

**First Tutorial:** `intro_timeseries.ipynb`

- **Why:** Most fundamental tutorial (TimeSeries is the core data type)
- **Size:** 39 cells (manageable for first attempt)
- **Impact:** Immediately enables English-speaking users to use gwexpy

## 📦 Deliverables

For each translated tutorial:
- `docs/web/en/guide/tutorials/<name>.ipynb` (new)
- `docs/web/en/guide/tutorials/<name>.md` (deleted)
- Updated index/toctree if needed
- Git commit with translation summary

## 🔍 Success Metrics

- [ ] All 19 tutorials translated
- [ ] `sphinx-build -nW` passes
- [ ] `sphinx-build -b linkcheck` passes
- [ ] All notebooks listed in English index
- [ ] No "under translation" stubs remain

## 📅 Estimated Effort

- **Phase 1 (INTRO):** 176 cells → ~6-8 hours
- **Phase 2 (CASE):** 32 cells → ~2-3 hours
- **Phase 3 (FIELD):** 42 cells → ~2-3 hours
- **Phase 4 (ADVANCED):** 97 cells → ~4-5 hours
- **Phase 5 (MATRIX):** 63 cells → ~3-4 hours

**Total:** ~17-23 hours of translation + testing

## 🤝 Collaboration Notes

- Use LLM for translation but **always review** for technical accuracy
- Maintain consistent voice and terminology
- Preserve original author intent and teaching style
- Keep code examples identical to ensure reproducibility

## 📚 References

- Japanese tutorials: `docs/web/ja/guide/tutorials/`
- English stubs: `docs/web/en/guide/tutorials/`
- Tutorial index: `docs/web/en/guide/tutorials/index.rst`
- Sphinx config: `docs/conf.py`

---

**Next Action:** Begin translation of `intro_timeseries.ipynb`
