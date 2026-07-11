# デバッグキャンペーン計画 (v0.1.9〜v0.2.0)

- **日付**: 2026-07-04
- **対象**: gwexpy 全体(GUI / CLI は対象外)
- **手法**: バグ修正履歴の傾向分析(Issue 253件・PR 177件・全1410コミット・CHANGELOG) → 3並列調査(テスト密度実測 / 既知バグパターン残存スキャン / 監査ドキュメント・open issue 精査) → コード裏取り → 計画策定
- **ベース**: main `7c6a5ea54`(= v0.1.8)
- **P4-0 status**: **completed with environment-only failures**(coverage 計測は完走・Phase 4 優先度は実測で確定済み。failed は pygmt 未導入による geomap 4件のみで full test green ではない)

## 背景・目的

v0.1.2〜v0.1.8 のバグ修正は「I/O・interop(#450/#452 波)→ attrs 深コピー(#442 波)→ 数値ロバスト性 statistics/spectral/fitting(Phase 1 sweep 波)」と体系的監査で進んできた。本計画はその傾向から「まだ残っていそうなバグ」と「検証不足領域」を4層(既知未修正 / 新規疑義 / Dangerous Defaults / カバレッジギャップ)に整理し、v0.1.9〜v0.2.0 への割り当てを定めるもの。

## P4-0: カバレッジ実測(2026-07-04)

```text
Measured commit: 7c6a5ea540630463845ddc1d5d9a41cdb93cb94d (tag v0.1.8)
Working tree before measurement: not clean — untracked `.codegraph/` のみ(本計画と無関係)
Python: 3.12.12 / pytest: 9.0.3 / coverage: 7.15.0(回避環境, 後述) / pytest-cov: 7.1.0
除外マーカー: gui, nds, network, pyautogui, cvmfs, root
実行形態: CI pr-fast と同じ3チャンク分割(core / io+segments+table / CI除外単体)+ coverage combine
P4-0 status: completed with environment-only failures(回避環境で完走。failed = geomap 4件のみ、環境要因)
```

### 実測結果: モジュール別カバレッジ(全スイート実行・間接込み)

対象7モジュール合計 **81.7%**(15,868 / 19,423 lines)。

| モジュール | カバレッジ | statements |
|---|---:|---:|
| gwexpy/frequencyseries | **68.4%** | 1,506 |
| gwexpy/histogram | 77.8% | 761 |
| gwexpy/timeseries | 80.6% | 8,497 |
| gwexpy/table | 81.3% | 699 |
| gwexpy/analysis | 85.0% | 1,910 |
| gwexpy/types | 85.2% | 4,158 |
| gwexpy/fields | 88.1% | 1,892 |

カバレッジ最下位ファイル(statements≥100、下位15):

| ファイル | カバレッジ | statements |
|---|---:|---:|
| gwexpy/frequencyseries/io/dttxml.py | 28.4% | 116 |
| gwexpy/timeseries/io/zarr_.py | 29.0% | 283 |
| gwexpy/timeseries/io/win.py | 34.0% | 147 |
| gwexpy/frequencyseries/bifrequencymap.py | 36.9% | 312 |
| gwexpy/timeseries/arima.py | 46.3% | 162 |
| gwexpy/histogram/collections.py | 63.9% | 244 |
| gwexpy/timeseries/_interop.py | 66.5% | 191 |
| gwexpy/analysis/bruco.py | 70.7% | 639 |
| gwexpy/timeseries/preprocess.py | 73.6% | 564 |
| gwexpy/table/segment_plot.py | 73.9% | 165 |
| gwexpy/timeseries/collections.py | 74.6% | 654 |
| gwexpy/types/array3d.py | 76.0% | 200 |
| gwexpy/frequencyseries/collections.py | 77.9% | 444 |
| gwexpy/timeseries/_resampling.py | 78.4% | 393 |
| gwexpy/timeseries/_spectral_fourier.py | 79.4% | 214 |

テスト実行結果: chunk A(pr-fast core)**5,994 passed** / 106 skipped(6:09)、chunk B(io+segments+table)**886 passed** / 13 skipped(0:49)、chunk C(CI除外単体)114 passed / **4 failed = `tests/test_geomap.py`(pygmt `GMTCLibNotFound`、オプションバックエンド未導入の環境要因。コードのバグではない)**(0:12)。coverage JSON・生ログはセッション scratchpad に保存(リポジトリ外)。

### 実測結果: 専用テスト密度(補助計測、`tests/<area>` のみ実行)

| モジュール | 専用テストのみ | 全スイート(間接込み) | 差 = 間接カバー分 |
|---|---:|---:|---:|
| frequencyseries | 62.9% | 68.4% | +5.5pt |
| fields | 81.1% | 88.1% | +7.0pt |
| analysis | 80.7% | 85.0% | +4.3pt |
| histogram | 77.8% | 77.8% | **±0**(他スイートから一切叩かれていない) |
| table | 81.0% | 81.3% | +0.3pt |

histogram と table はほぼ専用テストのみで支えられており、回帰の検出は専用スイートの質に完全依存する。

### 環境問題の記録: pytest-cov 実行が SIGABRT した根本原因(確定)

素の pytest-cov 実行は、テスト内容にかかわらず起動直後に SIGABRT(exit 134)した。調査で根本原因を確定:

**根本原因**: 開発環境に科学スタックが二重インストールされている(conda env `main` と `~/.local/lib/python3.12/site-packages` の双方に numpy/astropy/gwpy/regex 等が存在)。pytest-cov がモジュール名形式 `--cov=gwexpy.xxx` を解決する際、coverage が `importlib.util.find_spec()` で **conda env 側にインストール済みの gwexpy コピー**を早期 import し、numpy が二重ロードされ(astropy が "The NumPy module was reloaded" を警告)、ヒープ破壊 `munmap_chunk(): invalid pointer` → **GC 中に SIGABRT**(faulthandler 取得済み。abort 時のスタックは `regex` C 拡張 + dateparser import 中、lal._lal ロード済み)。

切り分けの記録:
- プラグイン説(ligo.skymap OpenMP / pytest-qt / xvfb 等): 単独無効・全無効・`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` いずれも変化なし → **棄却**
- `~/.local` の coverage 7.14.0 と conda pytest の混在: クリーン版 coverage で最小ケースは解消したが重量 import 系テストで再発 → 一因だが本丸ではない
- `PYTHONNOUSERSITE=1`: conda env 単体では astropy/gwpy の版が合わず収集エラー多発(`IORegistryError: 'hdf5.pycbc_live' already defined` / `UnitTypeError`)→ **実開発スタックは `~/.local` 側**であり user site は外せない
- トレーサ切替(`COVERAGE_CORE=sysmon`/`pytrace`)は効果なし → トレーサ実装の問題ではない
- pytest 抜きの `coverage` 単体 + `import gwexpy.histogram` は成功 → gwexpy のバグではない

**完走に使った回避策(ユーザー環境は無変更)**:
1. クリーンな coverage/pytest-cov を scratchpad に `pip install --target` し、`PYTHONPATH` で `~/.local` より優先させる
2. `--cov` を**パス形式**(`--cov=gwexpy/frequencyseries` 等)で指定 — モジュール名解決(`find_spec`)による早期 import 自体を回避

**恒久対策の推奨(未実施・要ユーザー判断)**:
- `~/.local` と conda env の重複スタックを一本化する(coverage 以外でも import 順序依存の不可解なクラッシュの火種)
- conda env 内の非 editable な gwexpy コピーを除去し `pip install -e .` に統一する

**副次的発見(それ自体が計画に有用)**:
- CI の pr-fast ゲートは、1プロセス実行の既知不安定性のため `tests/io/`・`tests/segments/`・`tests/table/`・`tests/timeseries/test_matrix_analysis.py`・`tests/types/test_series_matrix_io.py` 等を除外して実行している。今回の回避環境での3チャンク実行は cov 込みで全チャンク完走した


## 調査で確定した事実(コード裏取り済み)

### A. 既知・未修正バグ(open issue)

| # | 内容 | 状態 |
|---|---|---|
| #481 | GWF read path の gap zero-pad が v0.1.8 の NaN 既定と不整合(`gwexpy/timeseries/_gwf_io.py:219` に現存確認) | 修正方針確定済み |
| #466 (P2×2) | `fit_series` sigma crop の bin 境界不一致 ValueError + `run_mcmc` の `n_walkers<2*ndim` crash | 修正方針確定済み |
| #465 (P3×2) | `student_t_indicator` STFT: stride>fftlength crash、out_times の GPS epoch 消失、DC/Nyquist バイアス、gwpy 既定との差異 | 未着手 |
| #464 (P3×2) | Rayleigh/GauCh Monte-Carlo の unseeded RNG・thread-unsafe cache・metadata 未記録 | 未着手 |
| #451 | `TimeSeries.rms()` の gwpy 互換破壊(breaking) | open, v0.2.0 target / breaking-change gate |
| #444 | FrequencySeries collection registry fallback 不統一 | 意思決定未確定, v0.2.x |
| SP3/SP9 | `spectral/estimation.py:136` object-dtype TypeError / `:481-490` mean_avg>0 短絡が dB データで無効 | **hardening note**(#461 では confirmed silent-failure としては refuted 済み。再検証で動的再現できた場合のみ起票) |

### B. トラッカー衛生

- #460 はコード修正済み(#473, v0.1.7)だが閉じ忘れ
- #461 チェックリストが全項目未チェックのまま stale(実際はテーマ A〜G 消化済み、残りは H/I/J = #464/465/466)

### C. 新規疑義(パターンスキャン発見、コード実在確認済み・動的検証は未)

| # | 箇所 | 内容 |
|---|---|---|
| C1 | `gwexpy/histogram/io/_hdf5.py:52, 110-120` | Channel→str の型退化 round-trip(sample_rate/unit 消失)。過去のバグクラス「round-trip メタ損失」と同型 |
| C2 | `gwexpy/fields/scalar.py:1593-1595` | `zscore` が std==0 縮退を無警告 0.0 化。**併発: baseline に NaN 1点で全出力が無警告 0.0 化**(mean/std が nanmean 系でないため) |
| C3 | `gwexpy/analysis/coupling.py:324-329` | 単位変換失敗を logger.debug のみで dimensionless へ silent フォールバック |
| C4 | `gwexpy/histogram/_core.py:134` | Histogram の copy/crop/rebin 系生成パスで `channel` が参照渡しされる疑い(`kwargs["channel"] = getattr(self, "channel", None)`)。同パターンが `_core.py:222`(crop)と `_rebin.py:176` にも(3箇所セット) |
| C5 | `gwexpy/analysis/bruco.py:374` | 外部 coherence の NaN を無警告 0 化(セマンティクスとして擁護可能、warning/文書化が論点) |

### D. Dangerous Defaults(eps ハードコード問題)の移行状況実測

計画書 `phase1_dangerous_defaults.md`(2026-02-03)の7ファイルを実測した結果:

- **移行済み**: `signal/preprocessing/whitening.py`, `timeseries/preprocess.py`, `timeseries/decomposition.py`(+ `noise/magnetic.py` は該当なし)
- **未移行3件**:
  - `timeseries/pipeline.py:395` — `WhitenTransform eps=1e-12`
  - `timeseries/matrix_analysis.py:794,820` — `partial_correlation_matrix eps=1e-8` が `cov+eps*I` に直加算。strain スケールでは cov 対角 ~1e-42 を eps が完全支配し **pcorr≈0 に潰れる**(本計画中で最も危険)
  - `types/time_plane_transform.py:371` — `normalize_per_sigma eps=1e-30`

### E. 検証不足領域(P4-0 実測で確定済み)

事前の暫定評価(テスト密度分析)と P4-0 実測(2026-07-04)の照合結果:

- **frequencyseries が最下位で確定**(実測68.4%、専用テストのみでは62.9%)。主因は `bifrequencymap.py`(**36.9%**)と `io/dttxml.py`(**28.4%**)。`collections.py` は77.9%で暫定評価より健闘しているが下位圏。テスト2本が gwpy 上流の `import *` スタブ、contract テスト1本のみという構造問題は変わらず
- `timeseries/collections.py`: 実測74.6%。バッチ操作(resample/filter/whiten)の全要素伝播が未検証という懸念は残る
- **実測で新たに特定した低カバレッジファイル**: `timeseries/io/zarr_.py`(29.0%)・`io/win.py`(34.0%)・`arima.py`(**46.3%**、暫定分析では未検出)・`histogram/collections.py`(63.9%)
- analysis/`bruco.py`: 実測70.7%(639 stmts)— sweep 第1波対象として実測でも裏付け。fields は88.1%で相対的に健全(`scalar.py` は最下位25圏外)
- **histogram は間接カバー ±0**: 他スイートから一切叩かれておらず、専用スイートの質に完全依存
- 数値ロバスト性レンズでの未 sweep 領域: fields, analysis, noise, histogram, table, spectrogram, timeseries 本体
- regression 命名テストは4本のみ(analysis/docs/fitting)
- 旧 `test_coverage_gap_report.md`(2026-01-27)の「`hurst.py` 0%」は誤りと実測でも確認(最下位25圏外 = 83%以上)

## Phase 0: トラッカー衛生 + Dangerous Defaults 記録 — **完了(2026-07-04)**

- ✅ #460 クローズ(修正PR #473/v0.1.7 参照コメント付き)
- ✅ #461 チェックリストを実態に更新(A〜G を `[x]` 化・修正PR/収録版を追記、Status update セクション追加。残 H/I/J = #464/#465/#466)
- ✅ Dangerous Defaults 未移行3件を **#482** として起票(`bug`, `needs-triage`)

## Phase 1: 確定バグ修正 Wave 1(v0.1.9 前半・2PR 並行可)

- **W1-a #481**: `_gwf_io.py` の gap 既定 0.0→np.nan。RED: gap 入り GWF フィクスチャで `read(gap="pad")` → gap 部が NaN を期待する regression test + `pad=0.0` 明示時の後方互換テスト。挙動変更につき CHANGELOG 明記
- **W1-b #466**: `fitting/core.py`。RED-1: bin 境界一致 x_range + sigma で ValueError しない。RED-2: 多パラメータモデルで `n_walkers` 自動昇格 or 明示エラー
- ゲート: 両PRマージ + full testsuite green → v0.1.9 の主内容

## Phase 2: P3 確定バグ + C トリアージ(v0.1.9 後半〜v0.1.10)

- **W2-a #464**: `seed`/`rng` 引数追加(np.random.default_rng)、cache の thread-safe 化、metadata に seed/n_monte_carlo 記録。検証: 同一 seed 2回実行の完全一致テスト
- **W2-b #465**(最大工数): stride>fftlength の明示 ValueError、GPS epoch 復元(`out_times += ts.t0`)、DC/Nyquist 除外、gwpy 差異文書化。t0 非ゼロフィクスチャで検証
- **W2-c SP3/SP9 hardening note 再検証**: #461 では confirmed silent-failure finding としては refuted 済み(P3 hardening note として保持)。まず動的再現を試み、再現できた場合のみ issue 起票 → object-dtype 入力テスト + 負値(dB)スペクトルでの非定常検出テスト。再現しない場合は hardening note / doc-only として扱う
- **W2-d C トリアージ**: 各30分〜1hの最小再現スクリプト(scratchpad)→真偽判定→真なら再現コード添付で issue 起票。C2 は「warning 追加(v0.1.x)」と「NaN 化(v0.2.0)」の2段階。C4 は3箇所セットで deepcopy 化。C5 は wont-fix(docstring 明記)もあり得る。結果は真偽問わず監査レポートとして tech_notes に保存

## Phase 3: Dangerous Defaults 残修正 + v0.2.0 準備

- **W3-a**: 未移行3ファイルを whitening.py の確立パターン(`eps='auto'` + `get_safe_epsilon`/`SAFE_FLOOR_STRAIN`)に統一。`matrix_analysis.py` は cov スケール相対 eps(`trace(cov)/n * rel_eps`)。RED: strain スケール(~1e-21)データで pcorr≈0 に潰れることを示してから GREEN(スケール不変性テスト、`tests/numerics/test_scale_invariance.py` パターン踏襲)
- **前倒し注記**: `matrix_analysis.py` の `eps=1e-8` 直加算は本計画中で最も影響が大きい可能性があり、**RED テストの作成だけは W2-d(C トリアージ)より前に実施する**。推奨実施順: W1-a/W1-b → W2-a → W3-a RED テスト(特に `partial_correlation_matrix`)→ W2-b → C トリアージ
- **W3-b**: #451(rms, DeprecationWarning 付き互換シム→v0.2.0 切替)、#444(registry 意思決定→実装)、C2 の NaN 化。#413 migration guide への記載と #401 contract audit wave との整合

## Phase 4: 検証不足領域の対応

> **Phase 4 の優先度は P4-0 実測(2026-07-04)で確定済み。** 事前の暫定順序は概ね実測と一致したが、対象ファイルを実測値で精緻化した。

- **P4-0**: 完了(本計画書冒頭の実測節を参照)
- **P4-1 frequencyseries テスト増強**(実測68.4%で最下位確定): `bifrequencymap.py`(36.9%)と `io/dttxml.py`(28.4%)を最優先、`collections.py`(77.9%)は contract 拡充。gwpy `import *` スタブ2本の実テスト化も含む
- **P4-2 timeseries/collections バッチ伝播**(74.6%): resample/filter/whiten の全要素伝播パラメトリックテスト(コスト低)。実測で `arima.py`(46.3%)を同枠に追加
- **P4-3 未 sweep 領域への数値ロバスト性 sweep**(**コスト大・別途ユーザー承認**): 第1波 = `analysis/bruco.py`(70.7%・639 stmts)+ `histogram/collections.py`(63.9%・間接カバー±0)。fields は実測88.1%と健全なため `scalar.py` は第2波へ後退(LOC は大きいがカバレッジ良好)。**注**: この後退はカバレッジ優先度のみの話であり、C2(`scalar.py` zscore 縮退)は挙動変更候補として v0.2.0 ゲート(W2-d トリアージ → W3-b NaN 化)に残る
- **P4-4 マイナー I/O**: 実測で裏付け — `zarr_.py`(29.0%)・`win.py`(34.0%)・tdms/ats。synthetic fixture での round-trip 契約テスト

## リリース割り当て(週次ペース前提)

| リリース | 内容 |
|---|---|
| v0.1.9 | Phase 0 + W1-a/W1-b(+間に合えば W2-a) |
| v0.1.10 | W2-b/W2-c/C トリアージ確定修正 + W3-a |
| v0.1.11 | 溢れ + P4-1/P4-2(P4-0 実測で優先度確定済み。GO の場合のみ) |
| v0.2.0 | #451, #444, C2 NaN 化等の挙動変更(#413/#401 ゲートと合流) |

## リスク

- W1-a は挙動変更(gap="pad" かつ pad 未指定時のみ)。**merge gate = `pad=0.0` 明示時の後方互換テスト + CHANGELOG 記載**の両方
- W3-a の matrix_analysis.py 修正は数値結果を変える(現状が strain スケールで無意味な値のため改善方向だが CHANGELOG 明記)
- C2 は2段階(warning→NaN 化)で破壊的変更を回避
