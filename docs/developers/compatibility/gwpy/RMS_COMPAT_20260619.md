# TimeSeries.rms gwpy 互換性監査 (2026-06-19)

GitHub issue [#451] を起点に、`TimeSeries.rms` の `gwpy` 後方互換性破壊を修正し、
あわせて同名メソッド群の互換性を `gwpy` 実体（`/tmp/venv`）と照合した記録。

判定は [`API_UX_POLICY_20260303.md`](./API_UX_POLICY_20260303.md) の Rule 1–4 に従う
（`gwpy` 同名メソッドは引数順・位置引数許容を維持、拡張は非破壊で追加、互換は両流の
回帰テストで固定、`gwpy` 非互換の独自 API は明示）。

---

## ✅ 1. `TimeSeries.rms(stride)` の互換性破壊 — **修正済 (2026-06-19)**

- **問題**: `gwpy.timeseries.TimeSeries.rms(stride=1)` は第1位置引数が `stride`（秒）で、
  `stride` 秒ごとに 1 個の RMS 値を持つ新しい `TimeSeries`（`dt = stride`）を返す。
  一方 `gwexpy` は共有 `StatisticalMethodsMixin.rms(axis=None, ...)` を継承しており、
  `data.rms(10)` が `10` を numpy の **axis** と解釈して破綻していた。
  - `data.rms(10)` → `AxisError: axis 10 is out of bounds`
  - `data.rms(10*u.s)` → `TypeError`（Quantity を axis に使用）
  - 引数なしでも **スカラー** を返し、`gwpy` の RMS トレンド `TimeSeries` にならない。
- **原因**: `gwexpy` の `TimeSeries` MRO では汎用 mixin（`gwexpy/types/_stats.py`）が
  `gwpy` 基底より前に来るため、`gwpy` 専用の `TimeSeries.rms(stride)` を shadow していた。
  この mixin は N 次元 `SeriesMatrix`/`Array` および `FrequencySeries`（`gwpy` に `rms`
  なし＝純粋拡張）では正しく、**`TimeSeries` だけ** が `gwpy` 準拠の override を必要とする。
- **修正内容**: `TimeSeries` 専用の `StatisticsMixin`
  （`gwexpy/timeseries/_statistics.py`）に `gwpy` 準拠の `rms` と補助
  `_stride_to_seconds` を追加。汎用 mixin は他型のためにそのまま温存。
  - 数値的に `gwpy` と一致（同じ `int()` 切り捨ての `stridesamp`、末尾の不完全窓を破棄、
    窓ごとに `sqrt(mean(|x|**2))`）。ベクトル化 reshape は `gwpy` の Python ループと等価。
    実 `gwpy`（`/tmp/venv`）に対し `np.allclose` で一致確認済み。
  - 複素データは `np.abs(trimmed)**2`（= `gwpy` の `np.abs(...)**2`）で実数値を返し、
    入力が `float32` または `complex64` でも結果 dtype は `float64` とする。これは
    `gwpy` 4.0.1 が `np.zeros(nsteps)` へ各窓の値を書き込む dtype 契約に合わせたもの。
- **`gwpy` に対する意図的・文書化済みの改善**（非破壊）:
  1. 入力の物理単位を結果に **保持**（`gwpy` は無次元化する）。
  2. 時間 `Quantity` stride（例 `10*u.s`）を受理（`gwpy` は `TypeError`）。
  3. サブサンプル/ゼロ/負の stride は明示的な `ValueError`（`gwpy` は不透明な
     `ZeroDivisionError`）。不規則サンプリング系列も `ValueError`。
  4. 数値 stride の結果名は `gwpy` と同じ書式（例: `"None 1-second RMS"`）を使う。
     Quantity stride は gwexpy 拡張のため、秒へ正規化した値を書式化する。
- **テスト**: `tests/timeseries/test_rms_compat.py`（`gwpy` 参照一致・位置 int 回帰・
  単位保持・Quantity stride・各エッジケース）。`tests/types/test_stats_mixin.py::
  test_rms_with_unit` は実 `TimeSeries` を使う唯一のケースだったため、汎用 mixin を保持する
  `Series` に付け替え（`TimeSeries` の挙動は `test_rms_compat.py` が担当）。

### 呼び出し箇所の移行（スカラー RMS 依存）

`rms()` がトレンド `TimeSeries` を返すようになったため、戻り値を **スカラー** として
使っていた箇所を明示的な numpy 式（`np.sqrt(np.mean(x**2))` 等）に移行。`git` 履歴で
原本が確かに `.rms()` をスカラー文脈で使用していたことを確認済み（本番コードに未移行の
`.rms()` は残っていない）:

- `scripts/dev_tools/make_calibration_tutorial.py`（`ts_raw.rms().value:.2f` ×2）
- `docs/web/en/user_guide/tutorials/case_calibration_pipeline.ipynb`（同上）
- `docs/web/en/user_guide/tutorials/intro_table.ipynb`（行ごとの `row["noise"].rms().value`）
- `docs_redesign/how-to/case-studies/case_calibration_pipeline.ipynb`（同上）
- `docs_redesign/how-to/segments/intro_table.ipynb`（同上）

`docs/web` 版と `docs_redesign` 版はいずれも numpy を使わない `(x**2).mean()**0.5` に統一した。
`docs_redesign` の notebook は numpy を import していないため、`np.*` を使う書き方にすると
`nb_execution_raise_on_error = True` の docs ビルドが失敗する。

次の 2 件は意図的に差分へ戻していない:

- maintainer-local harness（このリポジトリには存在しない）配下の skill ファイルは scope 外。
- `docs_internal/archive/plans/PEMinjection-with-SegmentTable.md` は historical archive の
  exact restore 対象のため変更していない。

---

## 関連メソッドの監査結果

実 `gwpy`（`/tmp/venv`）に対し `inspect.signature` と実測で照合した結果。

| メソッド | 型 | 種別 | 所見 | 対応 |
|---|---|---|---|---|
| `rms(stride)` | `TimeSeries` | **署名破壊** | 位置 `stride` を axis 誤認 | ✅ 本変更で修正 |
| `mean/std/var` の `where` | 1D series | 署名差（寛容） | `gwpy` は `*, where=True`（keyword-only）、`gwexpy` は positional 許容。`gwpy` 互換コードは壊れない（`gwexpy` が上位互換）。`gwexpy` 固有の positional-`where` 呼び出しのみ `gwpy` に移植不可 | ⏳ 別フォローアップ（`*` 追加） |
| `mean/std/var/min/max/median/rms` の `ignore_nan` 既定 | 1D series | 挙動差（署名破壊ではない） | `gwpy`/numpy は NaN を伝播、`gwexpy` 1D は `ignore_nan=True` 既定で **黙って無視**。`ignore_nan` は keyword なので位置/署名破壊ではない | ⏳ 据え置き（下記参照） |
| 同上の 1D と matrix の既定不一致 | 1D vs matrix | 意味的不整合 | 1D=`True` / matrix（`matrix_analysis.py`）=`False`。同一データで結果が割れる | ⏳ `ignore_nan` 統一時に同時解消 |
| `median/min/max` | 1D series | 良性拡張 | `out/overwrite_input/keepdims/initial/where` は numpy 由来の追加。互換性損失なし | 対応不要 |
| `resample`, `crop` | `TimeSeries` | 良性 | 署名は異なるが位置・キーワード両対応で `gwpy` 呼び出しを受理 | 対応不要 |
| `skewness`, `kurtosis` | 1D series | 独自拡張 | `gwpy` に同名なし（Rule 4: 互換対象外） | 対応不要 |

---

## ⏳ 据え置き項目（別フォローアップ・physics-review 対象）

本変更は **rms 修正にスコープを限定**する（コミットを rms に集中させ、stats mixin の
署名変更リスクを混ぜない）ため、以下は本コミットに **含めない**:

1. **`ignore_nan` 既定の見直し**: 1D mixin の `ignore_nan=True` を `gwpy`/numpy 互換の
   `False` に倒すか、あるいは matrix を `True` に揃えるか。NaN を無視すべきか伝播すべきかは
   天体物理データの扱いに関わるため physics-review を要する。位置/署名破壊ではないため
   後続変更で安全に対応可能。
2. **`where` の keyword-only 化**: `_stats.py` の `mean/std/var` に `*` を追加して
   `gwpy` と署名を完全一致させる。機械的だが mixin 署名に触れるため独立変更＋テストで実施。
3. **1D / matrix の `ignore_nan` 既定統一**: 上記 (1) と同時に解消する。

---

## 検証

```bash
conda run -n gwexpy python -m pytest \
    tests/timeseries/test_rms_compat.py \
    tests/types/test_stats_mixin.py -q
conda run -n gwexpy ruff check \
    gwexpy/timeseries/_statistics.py tests/timeseries/test_rms_compat.py
```

`gwpy` 参照一致は `test_rms_matches_gwpy_reference`（`pytest.importorskip`）で固定。

## 2026-08-02 再監査 (GWpy 4.0.1)

現行の `conda` 環境で `inspect.getsource(gwpy.timeseries.TimeSeries.rms)` を再確認した。
実装は `int(stride * self.sample_rate.value)` で窓長を切り捨て、完全な窓だけを
`sqrt(mean(abs(window)**2))` で集約し、`self.__class__`、`t0`、`channel`、生成名、
`sample_rate=1 / stride` を使って結果を構築する。`np.zeros(nsteps)` により GWpy の
結果 dtype は常に `float64` であり、gwexpy も入力が `float32`/`complex64` の場合を
含めて明示的に `float64` を返す。gwexpy の実装はこの変換・窓・型・サンプリング・
既知メタデータの挙動を維持し、数値 stride の生成名も現行 GWpy に一致させている。

GWpy は結果構築時に `unit` を渡さず無次元単位にするが、gwexpy では物理量の RMS として
入力単位を保持することを既存の公開契約とする。時間 Quantity stride と不正 stride の
明示的な `ValueError` も同じく gwexpy の意図的な拡張である。

[#451]: https://github.com/tatsuki-washimi/gwexpy/issues/451
