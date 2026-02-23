# gwexpy.fitting

`gwexpy.fitting` モジュールは、iminuit を使用した高度なフィッティング機能を提供します。

## 概要

- **最小二乗フィッティング**: 実数・複素数データに対応
- **GLS (一般化最小二乗法)**: 共分散行列を考慮したフィッティング
- **MCMC**: emcee を使ったベイズ推定
- **統合パイプライン**: ブートストラップ → GLS → MCMC のワンライナー API

---

## クラスと関数

### メイン関数

| 名前                       | 説明                           |
| -------------------------- | ------------------------------ |
| `fit_series()`             | Series オブジェクトをフィット  |
| `fit_bootstrap_spectrum()` | 統合スペクトル解析パイプライン |

### クラス

| 名前                      | 説明                         |
| ------------------------- | ---------------------------- |
| `FitResult`               | フィット結果を格納するクラス |
| `GeneralizedLeastSquares` | GLS コスト関数クラス         |
| `RealLeastSquares`        | 実数データ用コスト関数       |
| `ComplexLeastSquares`     | 複素数データ用コスト関数     |

---

## fit_series

```python
fit_series(
    series,
    model,
    x_range=None,
    sigma=None,
    cov=None,
    cost_function=None,
    p0=None,
    limits=None,
    fixed=None,
    **kwargs
)
```

### パラメータ

| 名前            | 型                       | 説明                                                |
| --------------- | ------------------------ | --------------------------------------------------- |
| `series`        | Series                   | フィットするデータ                                  |
| `model`         | callable / str           | モデル関数または名前 ("gaussian", "power_law" など) |
| `x_range`       | tuple                    | フィット範囲 (xmin, xmax)                           |
| `sigma`         | array / scalar           | 誤差（`cov` が指定されると無視）                    |
| `cov`           | BifrequencyMap / ndarray | 共分散行列（GLS 用）                                |
| `cost_function` | callable                 | カスタムコスト関数（最優先）                        |
| `p0`            | dict / list              | 初期パラメータ値                                    |
| `limits`        | dict                     | パラメータ制限 {"A": (0, 100)}                      |
| `fixed`         | list                     | 固定するパラメータ名                                |

### 戻り値

`FitResult` オブジェクト

### 使用例

```python
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.fitting import fit_series

# データ準備
fs = FrequencySeries(y, frequencies=frequencies)

# 基本的なフィット
result = fit_series(fs, "power_law", p0={"A": 10, "alpha": -1.5})

# GLS フィット（共分散行列を使用）
result = fit_series(fs, "power_law", cov=covariance_matrix, p0={"A": 10, "alpha": -1.5})

# 結果確認
print(result.params)
print(result.errors)
result.plot()
```

---

## fit_bootstrap_spectrum

```python
fit_bootstrap_spectrum(
    data_or_spectrogram,
    model_fn,
    freq_range=None,
    method="median",
    rebin_width=None,
    block_size=None,
    ci=0.68,
    window="hann",
    fftlength=None,
    overlap=None,
    n_boot=1000,
    initial_params=None,
    bounds=None,
    fixed=None,
    run_mcmc=False,
    mcmc_walkers=32,
    mcmc_steps=5000,
    mcmc_burn_in=500,
    plot=True,
    progress=True,
)
```

### パラメータ

| 名前                  | 型                        | 説明                                                      |
| --------------------- | ------------------------- | --------------------------------------------------------- |
| `data_or_spectrogram` | TimeSeries / Spectrogram  | 入力データ                                                |
| `model_fn`            | callable                  | モデル関数 `model(f, *params)`                            |
| `freq_range`          | tuple                     | フィット周波数範囲                                        |
| `method`              | str                       | ブートストラップ平均化方法 ("median" / "mean")            |
| `rebin_width`         | float                     | 周波数リビン幅 (Hz)                                       |
| `block_size`          | float / Quantity / 'auto' | ブロックブートストラップのブロック時間長（秒）            |
| `ci`                  | float                     | 信頼区間 (デフォルト: 0.68)                               |
| `window`              | str                       | FFT ウィンドウ関数 (デフォルト: "hann")                   |
| `fftlength`           | float or Quantity         | FFT セグメント長（秒） (デフォルト: None、自動計算)       |
| `overlap`             | float or Quantity         | オーバーラップ長（秒） (デフォルト: None、ウィンドウ依存) |
| `n_boot`              | int                       | ブートストラップ回数                                      |
| `initial_params`      | dict                      | 初期パラメータ                                            |
| `bounds`              | dict                      | パラメータ制限                                            |
| `run_mcmc`            | bool                      | MCMC を実行するか                                         |
| `mcmc_steps`          | int                       | MCMC ステップ数                                           |
| `plot`                | bool                      | 結果をプロットするか                                      |

### 戻り値

`FitResult` オブジェクト（追加属性付き）:

- `psd`: ブートストラップ PSD
- `cov`: 共分散 BifrequencyMap
- `bootstrap_method`: 使用した平均化方法

### 使用例

```python
from gwexpy.fitting import fit_bootstrap_spectrum

def power_law(f, A, alpha):
    return A * f**alpha

result = fit_bootstrap_spectrum(
    spectrogram,
    model_fn=power_law,
    freq_range=(5, 50),
    rebin_width=0.5,
    block_size=2.0,  # 2秒
    initial_params={"A": 10, "alpha": -1.5},
    run_mcmc=True,
    mcmc_steps=3000,
)

# 結果
print(result.params)
print(result.parameter_intervals)  # MCMC 信頼区間
result.plot_corner()
```

---

## FitResult

フィット結果を格納するクラス。

### プロパティ

| 名前                  | 型      | 説明                             |
| --------------------- | ------- | -------------------------------- |
| `params`              | dict    | ベストフィットパラメータ         |
| `errors`              | dict    | パラメータ誤差                   |
| `chi2`                | float   | χ² 値                            |
| `ndof`                | int     | 自由度                           |
| `reduced_chi2`        | float   | 換算 χ²                          |
| `cov_inv`             | ndarray | GLS 共分散逆行列                 |
| `parameter_intervals` | dict    | MCMC パーセンタイル (16, 50, 84) |
| `mcmc_chain`          | ndarray | フル MCMC チェーン               |
| `samples`             | ndarray | MCMC サンプル（バーンイン後）    |

### メソッド

| 名前              | 説明                           |
| ----------------- | ------------------------------ |
| `plot()`          | データとフィット曲線をプロット |
| `bode_plot()`     | ボード線図（複素数データ用）   |
| `run_mcmc()`      | MCMC を実行                    |
| `plot_corner()`   | コーナープロット               |
| `plot_fit_band()` | 信頼帯付きフィットプロット     |

### 使用例

```python
result = fit_series(fs, "power_law", p0={"A": 10, "alpha": -1.5})

# 基本情報
print(result)  # Minuit の出力
print(f"χ²/dof = {result.reduced_chi2:.2f}")

# プロット
result.plot()

# MCMC
result.run_mcmc(n_steps=5000, burn_in=500)
print(result.parameter_intervals)
result.plot_corner()
result.plot_fit_band()
```

---

## GeneralizedLeastSquares

共分散逆行列を使った GLS コスト関数。

```python
class GeneralizedLeastSquares:
    errordef = 1.0  # Minuit.LEAST_SQUARES

    def __init__(self, x, y, cov_inv, model):
        ...

    def __call__(self, *args) -> float:
        # χ² = r.T @ cov_inv @ r
        ...

    @property
    def ndata(self) -> int:
        ...
```

### 使用例

```python
from gwexpy.fitting import GeneralizedLeastSquares, fit_series
from iminuit import Minuit

# 直接使用
def linear(x, a, b):
    return a * x + b

gls = GeneralizedLeastSquares(x, y, cov_inv, linear)
m = Minuit(gls, a=1, b=0)
m.migrad()

# fit_series 経由
result = fit_series(ts, linear, cost_function=gls, p0={"a": 1, "b": 0})
```

---

## 組み込みモデル

`gwexpy.fitting.models` で利用可能なモデル：

| 名前                 | 数式                          | パラメータ     |
| -------------------- | ----------------------------- | -------------- |
| `gaussian` / `gaus`  | A \* exp(-(x-μ)²/(2σ²))       | A, mu, sigma   |
| `power_law`          | A \* x^α                      | A, alpha       |
| `damped_oscillation` | A _ exp(-t/τ) _ cos(2πft + φ) | A, tau, f, phi |
| `pol0` ~ `pol9`      | c₀ + c₁x + c₂x² + ...         | p0, p1, ...    |
| `lorentzian`         | A / ((x-x₀)² + γ²)            | A, x0, gamma   |
| `exponential`        | A \* exp(-x/τ)                | A, tau         |

---

## 依存関係

- `iminuit`: 必須
- `emcee`: MCMC 機能に必要
- `corner`: コーナープロットに必要

---

## 単位とモデルのセマンティクス

`gwexpy.fitting` はデータとモデル間の単位の整合性を保証します：

- **単位の伝播**: `result.model(x)` でモデルを評価すると、入力データの単位が自動的に適用されます。
  - 入力 `x` が Hz 単位の場合、モデルは周波数単位で評価されます。
  - 出力 `y` はフィットしたデータの単位を保持します（例：m, V²/Hz）。
- **パラメータアクセス**: `result.params['name']` は `.value` と `.error` 属性を持つオブジェクトを返し、数値と統計的不確かさを分離します。
