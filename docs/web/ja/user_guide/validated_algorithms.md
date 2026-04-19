---
myst:
  html_meta:
    description: "GWexpy の検証済みアルゴリズムについて、数値許容誤差、前提条件、監査根拠、理論・API・チュートリアルへの導線を整理します。"
---

# 検証済みアルゴリズム

**ページ種別:** 高度ガイド / 理論補助
**バッジ:** `Advanced` `Theory companion`

:::{note}
**このページを読むべき方**:
- 解析手法の数学的・物理学的な妥当性を確認したい研究者
- 外部ライブラリ（SciPy, LALSuite 等）と GWexpy の数値的な計算誤差を知りたい開発者
- 使用するアルゴリズムが前提としているデータ特性（定常性、ガウス性など）を確認したい方
:::

:::{important}
**高度な検証・理論ページです**

このページは機能紹介の文書から辿りやすいよう user guide 配下に残していますが、入門ページではありません。API リファレンス、理論ノート、個別チュートリアルを監査目線でつなぐ高度・理論向けの入口として扱ってください。初見の方は [Getting Started](getting_started.md)、[前提条件と規約](prerequisites_and_conventions.md)、または各手法に対応する個別チュートリアルから読むのを推奨します。
:::

`gwexpy` で使用されている主要な数値アルゴリズムは、科学的な正確性を保証するために厳密な検証プロセスを経て実装されています。

**検索のヒント:** `validated algorithms`, `tolerance`, `FFT conventions`, `ホワイトニング`, `GLS`, `MCMC`, `VIF`

## このページの近道

- [検証の基準と計算精度](#検証の基準と計算精度-validation--precision)
- [客観的証拠](#客観的証拠-objective-evidence)
- [参照元一覧](#参照元一覧-source-references)
- [検証済みアルゴリズム要約表](#検証済みアルゴリズム要約表)
- [読み方ガイド](#読み方ガイド)
- [各アルゴリズムの詳細と仮定](#各アルゴリズムの詳細と仮定)
- [監査証跡](#監査証跡-audit-trail)
- [関連ドキュメント](#validated-algorithms-related-documents-ja)

(validated-algorithms-validation-criteria-ja)=
## 検証の基準と計算精度 (Validation & Precision)

「検証済み」と表記されるアルゴリズムは、以下の基準に基づき、指定の精度（Tolerance）を達成しています。

- **一致率 (Tolerance)**: $10^{-12}$ (相対誤差) を原則的なパス基準としています。これは倍精度浮動小数点演算において、丸め誤差以外の有意なロジック差異がないことを示します。
- **物理的不変性 (Invariance)**: スケール変換（データの 1000 倍など）をしても結果が整合する「Scale Invariance」を全アルゴリズムで検証しています。

(validated-ja-objective-evidence)=
## 客観的証拠 (Objective Evidence)

このページの要約は、直接参照できるリポジトリ内の検証資料に基づいています。

この証拠表は監査導線を 1 ページにまとめるため、列数を多めに保っています。モバイルでは共有 CSS に合わせた横スクロール前提で読むのが自然です。

| 証拠の種類 | 資料 | 何を示すか |
| :--- | :--- | :--- |
| 監査スコープ | [ALGORITHM_CONTEXT](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/ALGORITHM_CONTEXT.md) | どのアルゴリズムを、どの観点で点検したか |
| 統合結果 | [merged_validation_report](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/merged_validation_report.md) | 独立監査の統合所見、重大度、合意度 |
| 修正履歴 | [algorithm_fix_report_20260201](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/algorithm_fix_report_20260201.md) | 指摘後に何を修正したか |
| Field 系の物理レビュー | [scalarfield_physics_review_20260120](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs_internal/tech_notes/scalarfield_physics_review_20260120.md) | `ScalarField` の FFT と軸整合性に関するレビュー |

(validated-ja-source-references)=
## 参照元一覧 (Source References)

各アルゴリズム節では、文献・内部資料の参照をこの一覧のキーに寄せて、出典管理をこのページで一元化します。

| キー | 資料 | 用途 |
| :--- | :--- | :--- |
| `S1` | Press et al., *Numerical Recipes* (3rd ed., 2007), §12.3.2 | k-space 計算と FFT 軸解釈 |
| `S2` | Percival, D.B. & Walden, A.T., *Spectral Analysis for Physical Applications* (1993), Eq.(56) | Welch オーバーラップ / VIF 補正 |
| `S3` | [前提条件と規約](prerequisites_and_conventions.md) | 時刻系と FFT 規約に関する共通前提 |
| `S4` | [数値安定性](numerical_stability.md) | ホワイトニングの安定化指針 |
| `S5` | {ref}`客観的証拠 <validated-ja-objective-evidence>` と後述の Audit Trail | transient FFT と複素 GLS / MCMC 挙動の実装根拠 |

(validated-algorithms-summary-table-ja)=
## 検証済みアルゴリズム要約表

| アルゴリズム | 主対象 API | API ページ | 証拠 | 関連チュートリアル |
| :--- | :--- | :--- | :--- | :--- |
| **k-space 計算** | `ScalarField.fft_space()` | [Fields API](../reference/api/fields.rst) / [ScalarField](../reference/ScalarField.md) | {ref}`客観的証拠 <validated-ja-objective-evidence>` | [Field 入門](tutorials/field_scalar_intro.ipynb) |
| **Transient FFT** | `TimeSeries.fft(mode="transient")` / `TimeSeries._fft_transient` | [Time Series API](../reference/api/timeseries.rst) / [TimeSeries](../reference/TimeSeries.md) | {ref}`客観的証拠 <validated-ja-objective-evidence>` | [Signal Extraction](tutorials/case_signal_extraction.ipynb) |
| **VIF 補正** | `calculate_correlation_factor()` | [Spectral API](../reference/api/spectral.rst) / [Spectral](../reference/Spectral.md) | {ref}`客観的証拠 <validated-ja-objective-evidence>` | [Bootstrap 用](tutorials/case_bootstrap_gls_fitting.ipynb) |
| **予測時刻計算** | `ArimaResult.forecast()` | [Time Series API](../reference/api/timeseries.rst) / [TimeSeries](../reference/TimeSeries.md) | {ref}`客観的証拠 <validated-ja-objective-evidence>` | [Advanced ARIMA](tutorials/advanced_arima.ipynb) |
| **MCMC / GLS 尤度** | `fit_series()` / `GeneralizedLeastSquares` | [Fitting API](../reference/api/fitting.rst) / [gwexpy.fitting](../reference/fitting.md) | {ref}`客観的証拠 <validated-ja-objective-evidence>` | [Bootstrap 用](tutorials/case_bootstrap_gls_fitting.ipynb) |
| **適応ホワイトニング** | `whiten()` / `WhiteningModel` | [Preprocessing API](../reference/api/preprocessing.rst) | {ref}`客観的証拠 <validated-ja-objective-evidence>` | [ML Preprocessing 事例](tutorials/case_ml_preprocessing.ipynb) |

---

(validated-algorithms-how-to-read-ja)=
## 読み方ガイド

- 共通の前提条件、時刻系、FFT 規約を先に整理したい場合は [前提条件と規約](prerequisites_and_conventions.md) を入口にしてください。
- 監査メモより先に手を動かしたい場合は、要約表にある個別チュートリアルから該当手法だけ読むのが最短です。
- 実装側の API を先に確認したい場合は [Fields API](../reference/api/fields.rst), [Time Series API](../reference/api/timeseries.rst), [Spectral API](../reference/api/spectral.rst), [Fitting API](../reference/api/fitting.rst), [Preprocessing API](../reference/api/preprocessing.rst) を参照してください。
- 検証の背景や監査メモまで追いたい場合は、後述の {ref}`客観的証拠 <validated-ja-objective-evidence>`、{ref}`参照元一覧 <validated-ja-source-references>`、Audit Trail を参照してください。

(validated-algorithms-detail-sections-ja)=
## 各アルゴリズムの詳細と仮定

(validated-ja-k-space)=
### 1. k-space 計算
**対象**: {meth}`gwexpy.fields.ScalarField.fft_space`

角波数の計算は、物理学の標準定義 $k = 2\pi / \lambda$ に従っています。

**前提条件・仮定**:
- 空間座標 ($x, y$) が**等間隔（Uniform Grid）**であること。
- 非等間隔グリッドの場合は、事前に補間を行わない限り正しい波数軸は得られません。

**参照元**: {ref}`S1 <validated-ja-source-references>`

**関連チュートリアル**
- [Field 入門](tutorials/field_scalar_intro.ipynb)

**関連 API**
- [Fields API](../reference/api/fields.rst)
- [ScalarField](../reference/ScalarField.md)

**関連理論**
- [前提条件と規約](prerequisites_and_conventions.md)
- [アーキテクチャとデータフロー](architecture.md)

---

(validated-ja-transient-fft)=
### 2. 振幅スペクトル（トランジェントFFT）
**対象**: `TimeSeries._fft_transient`

密度スペクトル（PSD）ではなく、時間領域の**ピーク振幅**を直接読み取れる振幅規約を採用しています。

**前提条件・仮定**:
- 入力信号に窓関数が適用されていないこと（矩形窓を前提）。
- 窓関数適用済みのデータに対しては、別途コヒーレント・ゲインによる補正が必要です。

**参照元**: {ref}`S5 <validated-ja-source-references>`

**関連チュートリアル**
- [Signal Extraction チュートリアル](tutorials/case_signal_extraction.ipynb)

**関連 API**
- [Time Series API](../reference/api/timeseries.rst)
- [TimeSeries](../reference/TimeSeries.md)

**関連理論**
- [前提条件と規約](prerequisites_and_conventions.md)
- [数値安定性](numerical_stability.md)

---

(validated-ja-vif)=
### 3. VIF (分散膨張係数)
**対象**: {func}`gwexpy.spectral.estimation.calculate_correlation_factor`

Welch 法等におけるセグメント間のオーバーラップによる有効サンプル数の減少を補正します。

ここでの VIF は回帰分析の Variance Inflation Factor をそのまま指すのではなく、**オーバーラップ窓による分散膨張補正をコード上で `calculate_correlation_factor()` として実装したもの**です。理論名と関数名が異なるのは、API では「相関補正係数を計算する」役割を明示するためです。

**前提条件・仮定**:
- データが**弱定常 (Weakly Stationary)** であること。
- 非定常なグリッチやステップ応答が含まれる場合、VIF は分散を過小または過大評価する可能性があります。

**参照元**: {ref}`S2 <validated-ja-source-references>`

**関連チュートリアル**
- [Bootstrap GLS fitting 事例](tutorials/case_bootstrap_gls_fitting.ipynb)

**関連 API**
- [Spectral API](../reference/api/spectral.rst)
- [Spectral](../reference/Spectral.md)

**関連理論**
- [前提条件と規約](prerequisites_and_conventions.md)

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

**参照元**: {ref}`S3 <validated-ja-source-references>`

**関連チュートリアル**
- [Advanced ARIMA チュートリアル](tutorials/advanced_arima.ipynb)

**関連 API**
- [Time Series API](../reference/api/timeseries.rst)
- [TimeSeries](../reference/TimeSeries.md)

**関連理論**
- [前提条件と規約](prerequisites_and_conventions.md)

(validated-ja-mcmc-gls)=
### 5. MCMC / GLS 尤度
**対象**: `run_mcmc`, `GLS`

MCMC の対数尤度計算では、複素残差に対してもエルミート形式 `r.conj() @ cov_inv @ r` の実部を用いる実装を前提にしています。つまり、**複素データを API にそのまま渡した場合でも、内部では複素共役を考慮した二次形式で評価**されます。

**前提条件・仮定**:
- 共分散行列 `cov_inv` がエルミート正定値に近いこと。
- 複素残差は circular complex Gaussian に準じた扱いを想定しており、単純に実部だけへ切り捨てる挙動ではありません。

**参照元**: {ref}`S5 <validated-ja-source-references>`

**関連チュートリアル**
- [Bootstrap GLS fitting 事例](tutorials/case_bootstrap_gls_fitting.ipynb)

**関連 API**
- [Time Series API](../reference/api/timeseries.rst)
- [Fitting API](../reference/api/fitting.rst)
- [gwexpy.fitting](../reference/fitting.md)

**関連理論**
- [前提条件と規約](prerequisites_and_conventions.md)
- [数値安定性](numerical_stability.md)

---

(validated-ja-adaptive-whitening)=
### 6. 適応ホワイトニング
**対象**: `whiten()`, `WhiteningModel`, `.whiten(eps="auto")`

適応ホワイトニングでは、自動的に決まる安定化パラメータを用いて、非常に小さい PSD ビンや局所的な数値アンダーフローがあっても正規化が破綻しにくいようにしています。

**前提条件・仮定**:
- ホワイトニングに用いる FFT セグメントの範囲では、データが局所的に準定常とみなせること。
- 適応 `eps` は小さい分母を安定化するためのものであり、明らかに非定常なバーストや不適切な窓長設定まで補償するものではありません。

**参照元**: {ref}`S4 <validated-ja-source-references>`

**関連チュートリアル**
- [ML Preprocessing 事例](tutorials/case_ml_preprocessing.ipynb)

**関連 API**
- [Preprocessing API](../reference/api/preprocessing.rst)

**関連理論**
- [数値安定性](numerical_stability.md)
- [前提条件と規約](prerequisites_and_conventions.md)

---

(validated-algorithms-audit-trail-ja)=
## 監査証跡 (Audit Trail)

以下のリンクは、このページの要約を支える監査スコープ・統合所見・修正履歴を指しています。

- [ALGORITHM_CONTEXT](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/ALGORITHM_CONTEXT.md) - 検証対象アルゴリズムの全体像
- [merged_validation_report](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/merged_validation_report.md) - 独立監査を統合した重大度・合意度の一覧
- [check_claude_antigravity](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/check_claude_antigravity.md) - VIF / GLS / `_fft_transient` の監査メモ
- [algorithm_fix_report_20260201](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/algorithm_fix_report_20260201.md) - ARIMA や MCMC 周辺の修正履歴
- [scalarfield_physics_review_20260120](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs_internal/tech_notes/scalarfield_physics_review_20260120.md) - `ScalarField` の FFT と軸整合性に関する物理レビュー
- [report_scalarfield_physics_verification_20260122_222000](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs_internal/archive/reports/report_scalarfield_physics_verification_20260122_222000.md) - Field FFT 挙動の追検証メモ

(validated-algorithms-related-documents-ja)=
## 検証向け関連ドキュメント

- [数値安定性](numerical_stability.md) - 数値安定性（精度管理）
- [前提条件と規約](prerequisites_and_conventions.md) - 時刻系と FFT 規約の共通前提
- [用語集](glossary.rst) - アルゴリズム用語集
- [Fields API](../reference/api/fields.rst)
- [Time Series API](../reference/api/timeseries.rst)
- [Spectral API](../reference/api/spectral.rst)
- [Fitting API](../reference/api/fitting.rst)
- [Preprocessing API](../reference/api/preprocessing.rst)

(validated-algorithms-next-to-read-ja)=
## 次に読む

- [前提条件と規約](prerequisites_and_conventions.md) - このページの前提になる時刻系と FFT 規約を整理する
- [数値安定性](numerical_stability.md) - 適応ホワイトニングなど安定化の挙動を補う
- [アーキテクチャとデータフロー](architecture.md) - 検証対象 API がどの設計の上にあるかを確認する
- [チュートリアル一覧](tutorials/index.rst) - 実際の利用例から該当手法を追う
