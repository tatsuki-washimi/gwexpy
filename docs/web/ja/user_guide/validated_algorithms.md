# 検証済みアルゴリズム

:::{note}
**このページを読むべき方**:
- 解析手法の数学的・物理学的な妥当性を確認したい研究者
- 外部ライブラリ（SciPy, LALSuite 等）と GWexpy の数値的な計算誤差を知りたい開発者
- 使用するアルゴリズムが前提としているデータ特性（定常性、ガウス性など）を確認したい方
:::

`gwexpy` で使用されている主要な数値数値アルゴリズムは、科学的な正確性を保証するために厳密な検証プロセスを経て実装されています。

## 検証の基準と計算精度 (Validation & Precision)

「検証済み」と表記されるアルゴリズムは、以下の基準に基づき、指定の精度（Tolerance）を達成しています。

- **一致率 (Tolerance)**: $10^{-12}$ (相対誤差) を原則的なパス基準としています。これは倍精度浮動小数点演算において、丸め誤差以外の有意なロジック差異がないことを示します。
- **物理的不変性 (Invariance)**: スケール変換（データの 1000 倍など）をしても結果が整合する「Scale Invariance」を全アルゴリズムで検証しています。

(validated-ja-objective-evidence)=
## 客観的証拠 (Objective Evidence)

このページの要約は、直接参照できるリポジトリ内の検証資料に基づいています。

| 証拠の種類 | 資料 | 何を示すか |
| :--- | :--- | :--- |
| 監査スコープ | [ALGORITHM_CONTEXT](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/ALGORITHM_CONTEXT.md) | どのアルゴリズムを、どの観点で点検したか |
| 統合結果 | [merged_validation_report](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/merged_validation_report.md) | 独立監査の統合所見、重大度、合意度 |
| 修正履歴 | [algorithm_fix_report_20260201](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/algorithm_fix_report_20260201.md) | 指摘後に何を修正したか |
| Field 系の物理レビュー | [scalarfield_physics_review_20260120](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs_internal/tech_notes/scalarfield_physics_review_20260120.md) | `ScalarField` の FFT と軸整合性に関するレビュー |

## 検証済みアルゴリズム要約表

| アルゴリズム | 主対象 API | API ページ | 証拠 | 関連チュートリアル |
| :--- | :--- | :--- | :--- | :--- |
| **k-space 計算** | `ScalarField.fft_space()` | [Fields API](../reference/api/fields.rst) / [ScalarField](../reference/ScalarField.md) | {ref}`客観的証拠 <validated-ja-objective-evidence>` | [Field 入門](tutorials/field_scalar_intro.ipynb) |
| **Transient FFT** | `TimeSeries.fft(mode="transient")` / `TimeSeries._fft_transient` | [Time Series API](../reference/api/timeseries.rst) / [TimeSeries](../reference/TimeSeries.md) | {ref}`客観的証拠 <validated-ja-objective-evidence>` | — |
| **VIF 補正** | `calculate_correlation_factor()` | [Spectral API](../reference/api/spectral.rst) / [Spectral](../reference/Spectral.md) | {ref}`客観的証拠 <validated-ja-objective-evidence>` | [Bootstrap 用](tutorials/case_bootstrap_gls_fitting.ipynb) |
| **予測時刻計算** | `ArimaResult.forecast()` | [Time Series API](../reference/api/timeseries.rst) / [TimeSeries](../reference/TimeSeries.md) | {ref}`客観的証拠 <validated-ja-objective-evidence>` | — |
| **MCMC / GLS 尤度** | `fit_series()` / `GeneralizedLeastSquares` | [Fitting API](../reference/api/fitting.rst) / [gwexpy.fitting](../reference/fitting.md) | {ref}`客観的証拠 <validated-ja-objective-evidence>` | [Bootstrap 用](tutorials/case_bootstrap_gls_fitting.ipynb) |
| **適応ホワイトニング** | `whiten()` / `WhiteningModel` | [Preprocessing API](../reference/api/preprocessing.rst) | {ref}`客観的証拠 <validated-ja-objective-evidence>` | [数値安定性](numerical_stability.md) |

---

## 読み方ガイド

- 共通の前提条件、時刻系、FFT 規約を先に整理したい場合は [前提条件と規約](prerequisites_and_conventions.md) を入口にしてください。
- 実装側の API を先に確認したい場合は [Fields API](../reference/api/fields.rst), [Time Series API](../reference/api/timeseries.rst), [Spectral API](../reference/api/spectral.rst), [Fitting API](../reference/api/fitting.rst), [Preprocessing API](../reference/api/preprocessing.rst) を参照してください。
- 検証の背景や監査メモまで追いたい場合は、後述の {ref}`客観的証拠 <validated-ja-objective-evidence>` と Audit Trail を参照してください。

## 各アルゴリズムの詳細と仮定

(validated-ja-k-space)=
### 1. k-space 計算
**対象**: {meth}`gwexpy.fields.ScalarField.fft_space`

角波数の計算は、物理学の標準定義 $k = 2\pi / \lambda$ に従っています。

**前提条件・仮定**:
- 空間座標 ($x, y$) が**等間隔（Uniform Grid）**であること。
- 非等間隔グリッドの場合は、事前に補間を行わない限り正しい波数軸は得られません。

**根拠**: Press et al., *Numerical Recipes* (3rd ed., 2007), §12.3.2

**関連 API ページ**
- [Fields API](../reference/api/fields.rst)
- [ScalarField](../reference/ScalarField.md)

---

(validated-ja-transient-fft)=
### 2. 振幅スペクトル（トランジェントFFT）
**対象**: `TimeSeries._fft_transient`

密度スペクトル（PSD）ではなく、時間領域の**ピーク振幅**を直接読み取れる振幅規約を採用しています。

**前提条件・仮定**:
- 入力信号に窓関数が適用されていないこと（矩形窓を前提）。
- 窓関数適用済みのデータに対しては、別途コヒーレント・ゲインによる補正が必要です。

**関連 API ページ**
- [Time Series API](../reference/api/timeseries.rst)
- [TimeSeries](../reference/TimeSeries.md)

---

(validated-ja-vif)=
### 3. VIF (分散膨張係数)
**対象**: {func}`gwexpy.spectral.estimation.calculate_correlation_factor`

Welch 法等におけるセグメント間のオーバーラップによる有効サンプル数の減少を補正します。

ここでの VIF は回帰分析の Variance Inflation Factor をそのまま指すのではなく、**オーバーラップ窓による分散膨張補正をコード上で `calculate_correlation_factor()` として実装したもの**です。理論名と関数名が異なるのは、API では「相関補正係数を計算する」役割を明示するためです。

**前提条件・仮定**:
- データが**弱定常 (Weakly Stationary)** であること。
- 非定常なグリッチやステップ応答が含まれる場合、VIF は分散を過小または過大評価する可能性があります。

**根拠**: Percival, D.B. & Walden, A.T., *Spectral Analysis for Physical Applications* (1993), Eq.(56)

**関連 API ページ**
- [Spectral API](../reference/api/spectral.rst)
- [Spectral](../reference/Spectral.md)

---

(validated-ja-arima-forecast)=
### 4. 予測タイムスタンプ (ARIMA)
**対象**: {meth}`gwexpy.timeseries.arima.ArimaResult.forecast`

GPS 時刻系における連続性を保証するため、LIGO 規約に従った歩進計算を行います。

代表的な更新式は `forecast_t0 = t0 + n_obs * dt` です。ここで、

- `t0`: 元データ系列の開始 GPS 時刻
- `n_obs`: 学習に使った観測点数
- `dt`: サンプル間隔（秒）

を表します。

**前提条件・仮定**:
- 時刻系は **GPS 時刻（閏秒なし）** であること。
- UTC 等の閏秒が存在する時刻系でそのまま使用すると、将来予測時に 1 秒のズレが生じます。

**関連 API ページ**
- [Time Series API](../reference/api/timeseries.rst)
- [TimeSeries](../reference/TimeSeries.md)

(validated-ja-mcmc-gls)=
### 5. MCMC / GLS 尤度
**対象**: `run_mcmc`, `GLS`

MCMC の対数尤度計算では、複素残差に対してもエルミート形式 `r.conj() @ cov_inv @ r` の実部を用いる実装を前提にしています。つまり、**複素データを API にそのまま渡した場合でも、内部では複素共役を考慮した二次形式で評価**されます。

**前提条件・仮定**:
- 共分散行列 `cov_inv` がエルミート正定値に近いこと。
- 複素残差は circular complex Gaussian に準じた扱いを想定しており、単純に実部だけへ切り捨てる挙動ではありません。

**関連 API ページ**
- [Time Series API](../reference/api/timeseries.rst)
- [Fitting API](../reference/api/fitting.rst)
- [gwexpy.fitting](../reference/fitting.md)

---

(validated-ja-adaptive-whitening)=
### 6. 適応ホワイトニング
**対象**: `whiten()`, `WhiteningModel`, `.whiten(eps="auto")`

適応ホワイトニングでは、自動的に決まる安定化パラメータを用いて、非常に小さい PSD ビンや局所的な数値アンダーフローがあっても正規化が破綻しにくいようにしています。

**前提条件・仮定**:
- ホワイトニングに用いる FFT セグメントの範囲では、データが局所的に準定常とみなせること。
- 適応 `eps` は小さい分母を安定化するためのものであり、明らかに非定常なバーストや不適切な窓長設定まで補償するものではありません。

**関連 API ページ**
- [Preprocessing API](../reference/api/preprocessing.rst)
- [数値安定性](numerical_stability.md)

---

## 監査証跡 (Audit Trail)

以下のリンクは、このページの要約を支える監査スコープ・統合所見・修正履歴を指しています。

- [ALGORITHM_CONTEXT](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/ALGORITHM_CONTEXT.md) - 検証対象アルゴリズムの全体像
- [merged_validation_report](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/merged_validation_report.md) - 独立監査を統合した重大度・合意度の一覧
- [check_claude_antigravity](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/check_claude_antigravity.md) - VIF / GLS / `_fft_transient` の監査メモ
- [algorithm_fix_report_20260201](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/algorithm_fix_report_20260201.md) - ARIMA や MCMC 周辺の修正履歴
- [scalarfield_physics_review_20260120](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs_internal/tech_notes/scalarfield_physics_review_20260120.md) - `ScalarField` の FFT と軸整合性に関する物理レビュー
- [report_scalarfield_physics_verification_20260122_222000](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs_internal/archive/reports/report_scalarfield_physics_verification_20260122_222000.md) - Field FFT 挙動の追検証メモ

## 関連ドキュメント

- [数値安定性](numerical_stability.md) - 数値安定性（精度管理）
- [用語集](glossary.rst) - アルゴリズム用語集
- [Fields API](../reference/api/fields.rst)
- [Time Series API](../reference/api/timeseries.rst)
- [Spectral API](../reference/api/spectral.rst)
- [Fitting API](../reference/api/fitting.rst)
- [Preprocessing API](../reference/api/preprocessing.rst)
