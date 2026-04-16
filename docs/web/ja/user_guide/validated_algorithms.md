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

## 検証済みアルゴリズム要約表

| アルゴリズム | 対応 API | 外部比較元 | 数値精度 (Tol) | 物理的仮定 | 関連チュートリアル |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **k-space 計算** | `.fft_space()` | SciPy / FFTW | $10^{-15}$ | 等間隔グリッド | [Field 入門](tutorials/field_scalar_intro.ipynb) |
| **Transient FFT** | `._fft_transient()` | NumPy | $10^{-15}$ | 窓関数なし(Rect) | — |
| **VIF 補正** | `calculate_correlation_factor()` | 文献 (Percival) | — | 定常過程 | [Bootstrap 用](tutorials/case_bootstrap_gls_fitting.ipynb) |
| **予測時刻計算** | `ArimaResult.forecast()` | LIGO 規約 | $10^{-9}$ (s) | 閏秒なし(GPS) | — |
| **適応ホワイトニング** | `.whiten(eps="auto")` | 数値実験 | $10^{-12}$ | 局所的準定常 | [数値安定性](numerical_stability.md) |

---

## 読み方ガイド

- 実装側の API シグネチャを先に確認したい場合は [信号処理 API](../reference/api/signal.rst) と [フィッティング API](../reference/api/fitting.rst) を参照してください。
- 検証の背景や監査メモまで追いたい場合は、後述の Audit Trail から開発者向け資料へ進んでください。

## 各アルゴリズムの詳細と仮定

### 1. k-space 計算
**対象**: {meth}`gwexpy.fields.ScalarField.fft_space`

角波数の計算は、物理学の標準定義 $k = 2\pi / \lambda$ に従っています。

**前提条件・仮定**:
- 空間座標 ($x, y$) が**等間隔（Uniform Grid）**であること。
- 非等間隔グリッドの場合は、事前に補間を行わない限り正しい波数軸は得られません。

**根拠**: Press et al., *Numerical Recipes* (3rd ed., 2007), §12.3.2

---

### 2. 振幅スペクトル（トランジェントFFT）
**対象**: `TimeSeries._fft_transient`

密度スペクトル（PSD）ではなく、時間領域の**ピーク振幅**を直接読み取れる振幅規約を採用しています。

**前提条件・仮定**:
- 入力信号に窓関数が適用されていないこと（矩形窓を前提）。
- 窓関数適用済みのデータに対しては、別途コヒーレント・ゲインによる補正が必要です。

---

### 3. VIF (分散膨張係数)
**対象**: {func}`gwexpy.spectral.estimation.calculate_correlation_factor`

Welch 法等におけるセグメント間のオーバーラップによる有効サンプル数の減少を補正します。

ここでの VIF は回帰分析の Variance Inflation Factor をそのまま指すのではなく、**オーバーラップ窓による分散膨張補正をコード上で `calculate_correlation_factor()` として実装したもの**です。理論名と関数名が異なるのは、API では「相関補正係数を計算する」役割を明示するためです。

**前提条件・仮定**:
- データが**弱定常 (Weakly Stationary)** であること。
- 非定常なグリッチやステップ応答が含まれる場合、VIF は分散を過小または過大評価する可能性があります。

**根拠**: Percival, D.B. & Walden, A.T., *Spectral Analysis for Physical Applications* (1993), Eq.(56)

---

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

### 5. MCMC / GLS 尤度
**対象**: `run_mcmc`, `GLS`

MCMC の対数尤度計算では、複素残差に対してもエルミート形式 `r.conj() @ cov_inv @ r` の実部を用いる実装を前提にしています。つまり、**複素データを API にそのまま渡した場合でも、内部では複素共役を考慮した二次形式で評価**されます。

**前提条件・仮定**:
- 共分散行列 `cov_inv` がエルミート正定値に近いこと。
- 複素残差は circular complex Gaussian に準じた扱いを想定しており、単純に実部だけへ切り捨てる挙動ではありません。

---

## 監査証跡 (Audit Trail)

詳細なユニットテスト結果、文献との比較スクリプト、および過去のレビュー記録は `docs_internal/verification/` ディレクトリに保管されています。
すべての変更は CI 環境での `verify-physics` ゲートをパスした後にマージされます。

- [ALGORITHM_CONTEXT](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/ALGORITHM_CONTEXT.md) - 検証対象アルゴリズムの全体像
- [check_claude_antigravity](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/check_claude_antigravity.md) - VIF / GLS / `_fft_transient` の監査メモ
- [algorithm_fix_report_20260201](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/algorithm_fix_report_20260201.md) - ARIMA や MCMC 周辺の修正履歴
- `docs_internal/tech_notes/scalarfield_physics_review_20260120.md`
- `docs_internal/archive/reports/report_scalarfield_physics_verification_20260122_222000.md`

## 関連ドキュメント

- [数値安定性](numerical_stability.md) - 数値安定性（精度管理）
- [用語集](glossary.rst) - アルゴリズム用語集
- [信号処理リファレンス](../reference/api/signal.rst)
