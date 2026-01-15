# Signal Processing API Reference

このドキュメントは `gwexpy.timeseries.TimeSeries` の信号処理メソッドの API リファレンスです。

## Hilbert 変換関連

### 概要

Hilbert 変換は実信号から解析信号（analytic signal）を生成するための手法です。
解析信号を用いることで、瞬時位相や瞬時周波数を計算できます。

### 数学的定義

実信号 $x(t)$ に対して、解析信号 $z(t)$ は以下で定義されます：

$$
z(t) = x(t) + i \cdot \mathcal{H}[x(t)]
$$

ここで $\mathcal{H}[x]$ は $x$ の Hilbert 変換であり、$1/(\pi t)$ との畳み込みで定義されます。

瞬時位相と瞬時周波数は以下で定義されます：

$$
\phi(t) = \arg(z(t))
$$

$$
f(t) = \frac{1}{2\pi} \frac{d\phi}{dt}
$$

---

## `hilbert`

```python
TimeSeries.hilbert(
    pad: int | Quantity = 0,
    pad_mode: str = "reflect",
    pad_value: float = 0.0,
    nan_policy: Literal["raise", "propagate"] = "raise",
    copy: bool = True
) -> TimeSeries
```

### 説明

Hilbert 変換を用いて解析信号を計算します。

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `pad` | int または Quantity | 0 | 各端にパディングするサンプル数（または時間長）|
| `pad_mode` | str | "reflect" | パディングモード（'reflect', 'constant', 'edge' など）|
| `pad_value` | float | 0.0 | 'constant' モード時の値 |
| `nan_policy` | str | "raise" | NaN/Inf の処理方法。'raise' で例外、'propagate' で伝播 |
| `copy` | bool | True | 入力が複素数の場合にコピーを返すか |

### 戻り値

複素数の解析信号を含む `TimeSeries`。入力と同じ長さ。

### 例外

- `ValueError`: 入力に NaN または無限大が含まれる場合（`nan_policy='raise'` 時）
- `ValueError`: 不規則サンプリングの場合

### 注意事項

⚠️ **前処理はユーザー責務**: demean、detrend、フィルタリング、窓関数などは自動適用されません。必要に応じてユーザーが事前に適用してください。

⚠️ **端点アーティファクト**: Hilbert 変換はスペクトル漏れにより端点でアーティファクトを生じる可能性があります。`pad` パラメータを使用するか、適切な窓関数を適用してください。

---

## `instantaneous_phase`

```python
TimeSeries.instantaneous_phase(
    deg: bool = False,
    unwrap: bool = False,
    **kwargs
) -> TimeSeries
```

### 説明

Hilbert 変換を用いて瞬時位相を計算します。

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `deg` | bool | False | True の場合、度単位で返す。False の場合、ラジアン |
| `unwrap` | bool | False | True の場合、位相のアンラップ（不連続の除去）を行う |
| `**kwargs` | - | - | `hilbert()` に渡されるオプション |

### 戻り値

瞬時位相を含む `TimeSeries`。単位は 'rad' または 'deg'。

### 定義

```python
analytic = hilbert(x)
phase = np.angle(analytic)  # ラジアン
if unwrap:
    phase = np.unwrap(phase, period=2*np.pi)  # 度の場合は period=360
```

### 注意事項

- 端点は自動でトリミングされません
- 前処理（demean、detrend など）は自動適用されません

---

## `instantaneous_frequency`

```python
TimeSeries.instantaneous_frequency(
    unwrap: bool = True,
    smooth: int | Quantity | None = None,
    **kwargs
) -> TimeSeries
```

### 説明

Hilbert 変換を用いて瞬時周波数を計算します。

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `unwrap` | bool | True | 微分前に位相をアンラップするか（推奨: True）|
| `smooth` | int, Quantity, None | None | 平滑化窓。None で平滑化なし |
| `**kwargs` | - | - | `hilbert()` に渡されるオプション |

### 戻り値

瞬時周波数を含む `TimeSeries`。単位は 'Hz'。

### 定義

```python
phase = instantaneous_phase(unwrap=True, deg=False)  # ラジアン
dphi_dt = np.gradient(phase, dt)  # 時間微分
f_inst = dphi_dt / (2 * np.pi)  # Hz に変換
```

### 注意事項

- 端点は自動でトリミングされません
- 端点付近は数値微分とHilbert変換のアーティファクトにより精度が低下する可能性があります
- 精度評価時は中央領域（例：10%-90%）のみを使用することを推奨

---

## 使用例

### 基本的な使用法

```python
import numpy as np
from gwexpy.timeseries import TimeSeries

# テスト信号の生成
t = np.linspace(0, 10, 10000)
f0 = 5.0  # Hz
signal = np.cos(2 * np.pi * f0 * t)
ts = TimeSeries(signal, dt=0.001, unit='V')

# 前処理（ユーザー責務）
ts_processed = ts.detrend().taper()

# Hilbert 変換
analytic = ts_processed.hilbert()
envelope = np.abs(analytic.value)

# 瞬時位相
phase = ts_processed.instantaneous_phase(unwrap=True)

# 瞬時周波数
f_inst = ts_processed.instantaneous_frequency()

# 中央領域で周波数を評価
n = len(f_inst.value)
central = f_inst.value[int(n*0.1):int(n*0.9)]
print(f"Median frequency: {np.median(central):.2f} Hz")  # ≈ 5.0 Hz
```

### 端点アーティファクトの軽減

```python
# パディングを使用
analytic = ts.hilbert(pad=100)

# または窓関数を適用
ts_windowed = ts.taper(side='both')
analytic = ts_windowed.hilbert()
```

### チャープ信号の解析

```python
# 周波数が変化するチャープ信号
f_start, f_end = 10.0, 50.0
t = np.linspace(0, 5, 50000)
chirp_phase = 2 * np.pi * (f_start * t + (f_end - f_start) / (2 * 5) * t**2)
signal = np.cos(chirp_phase)
ts = TimeSeries(signal, dt=0.0001, unit='V')

# 瞬時周波数で周波数変化を追跡
f_inst = ts.instantaneous_frequency()
```

---

## 関連メソッド

- `envelope()`: Hilbert 変換を用いた包絡線（振幅）の計算
- `radian()`: 複素信号の位相角（Hilbert なし）
- `degree()`: 複素信号の位相角（度単位、Hilbert なし）
- `unwrap_phase()`: `instantaneous_phase(unwrap=True)` のエイリアス

---

## Baseband 復調

### 概要

`baseband` メソッドは、キャリア周波数をベースバンド（DC）にシフトし、オプションでローパスフィルタとリサンプリングを適用します。

処理チェーン：

```
mix_down(f0) → [lowpass(cutoff)] → [resample(output_rate)]
```

### 2つの実行モード

**モードA（解析帯域明示）**:

- `baseband(f0=fc, lowpass=cutoff, output_rate=None|...)`
- ミキシング後にローパスフィルタを適用して解析帯域を定義
- オプションでリサンプルしてデータレートを削減

**モードB（ダウンサンプル優先）**:

- `baseband(f0=fc, lowpass=None, output_rate=rate)`
- 明示的なローパスをスキップし、リサンプルのアンチエイリアスに依存
- 二重フィルタを避けたい場合に有用

---

## `baseband`

```python
TimeSeries.baseband(
    *,
    phase: array_like | None = None,
    f0: float | Quantity | None = None,
    fdot: float | Quantity = 0.0,
    fddot: float | Quantity = 0.0,
    phase_epoch: float | None = None,
    phase0: float = 0.0,
    lowpass: float | Quantity | None = None,
    lowpass_kwargs: dict | None = None,
    output_rate: float | Quantity | None = None,
    resample_kwargs: dict | None = None,
    singlesided: bool = False
) -> TimeSeries
```

### 説明

TimeSeries を周波数シフト（ヘテロダイン）してベースバンドに復調し、オプションでローパスフィルタとリサンプリングを適用します。

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `phase` | array_like または None | None | ミキシング用の明示的な位相配列（ラジアン）|
| `f0` | float または Quantity | None | ミキシングの中心周波数（Hz）。0 < f0 < Nyquist である必要あり |
| `fdot` | float または Quantity | 0.0 | 周波数微分（Hz/s）|
| `fddot` | float または Quantity | 0.0 | 周波数2階微分（Hz/s²）|
| `phase_epoch` | float または None | None | 位相モデルの基準エポック |
| `phase0` | float | 0.0 | 初期位相オフセット（ラジアン）|
| `lowpass` | float または Quantity または None | None | ローパスフィルタのコーナー周波数（Hz）|
| `lowpass_kwargs` | dict または None | None | `lowpass()` に渡す追加引数 |
| `output_rate` | float または Quantity または None | None | 出力サンプルレート（Hz）|
| `resample_kwargs` | dict または None | None | `resample()` に渡す追加引数 |
| `singlesided` | bool | False | True の場合、振幅を2倍（実信号用）|

### 戻り値

複素数のベースバンド信号を含む `TimeSeries`。

### 例外条件

| 条件 | 例外 |
|------|------|
| `f0 <= 0` | `ValueError` |
| `f0 >= Nyquist`（regular series の場合）| `ValueError` |
| `lowpass <= 0` | `ValueError` |
| `lowpass >= Nyquist` | `ValueError` |
| `output_rate <= 0` | `ValueError` |
| `lowpass` と `output_rate` の両方が None | `ValueError` |
| `lowpass >= output_rate/2`（新 Nyquist 超過）| `ValueError` |

### 注意事項

⚠️ **前処理はユーザー責務**: demean、detrend、フィルタリングは自動適用されません。DC オフセットやトレンドがある場合、ベースバンド結果に影響します。

⚠️ **lowpass と f0 の関係**: 一般に `lowpass < f0` が推奨されますが、強制はされません。キャリア周辺の変調のみを捉える場合は、lowpass をキャリア周波数より小さく設定してください。

⚠️ **GWpy 互換**: lowpass と resample の内部処理は GWpy のメソッドに委譲されます。カスタマイズは `lowpass_kwargs` と `resample_kwargs` で可能です。

---

## Baseband 使用例

### モードA: ローパス指定

```python
import numpy as np
from gwexpy.timeseries import TimeSeries

# 100 Hz のキャリア信号
t = np.arange(0, 10, 0.001)  # 1000 Hz サンプリング
signal = np.cos(2 * np.pi * 100 * t)
ts = TimeSeries(signal, dt=0.001, unit='V')

# 前処理（推奨）
ts = ts.detrend()

# ベースバンドに復調（10 Hz 解析帯域）
z = ts.baseband(f0=100, lowpass=10)

# DC 成分が支配的になる
print(f"DC magnitude: {np.abs(np.mean(z.value)):.3f}")
```

### モードB: リサンプルのみ

```python
# リサンプルのアンチエイリアスに依存
z = ts.baseband(f0=100, lowpass=None, output_rate=50)

# 出力サンプルレートが 50 Hz になる
print(f"Output rate: {z.sample_rate}")
```

### 両方を指定

```python
# ローパスとリサンプルの両方
z = ts.baseband(f0=100, lowpass=10, output_rate=50)

# lowpass < output_rate/2 (= 25 Hz) である必要あり
```

### GWpy kwargs の透過

```python
# ローパスフィルタのカスタマイズ
z = ts.baseband(
    f0=100,
    lowpass=10,
    lowpass_kwargs={"filtfilt": True}  # GWpy オプション
)

# リサンプルのカスタマイズ
z = ts.baseband(
    f0=100,
    lowpass=None,
    output_rate=50,
    resample_kwargs={"window": "hamming"}  # GWpy オプション
)
```

---

## `heterodyne` （GWpy 互換）

```python
TimeSeries.heterodyne(
    phase: array_like,
    stride: float | Quantity = 1.0,
    singlesided: bool = True
) -> TimeSeries
```

### 説明

GWpy の `TimeSeries.heterodyne()` と**完全に同一のアルゴリズム**を実装しています。
入力 TimeSeries を位相系列でヘテロダインし、固定ストライドで平均化します。

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `phase` | array_like | - | ミキシング用位相配列（ラジアン）。`len(phase) == len(self)` が必須 |
| `stride` | float または Quantity | 1.0 | 平均化の時間ステップ（秒）。サンプル数は `int(stride * sample_rate)` で切り捨て |
| `singlesided` | bool | True | True の場合、振幅を2倍（実信号用）。gwexpy のデフォルトは True（GWpy は False）|

### 戻り値

複素 `TimeSeries`。`dt = stride`、値は各ストライドでの平均振幅と位相を `mag * exp(1j * phase)` として表現。

### 例外

| 条件 | 例外 |
|------|------|
| `phase` が array_like でない（`len()` が失敗）| `TypeError` |
| `len(phase) != len(self)` | `ValueError` |

### アルゴリズム（GWpy と同一）

```python
stridesamp = int(stride * sample_rate)  # floor 切り捨て
nsteps = int(N // stridesamp)           # 余りのサンプルは破棄

for step in range(nsteps):
    istart = stridesamp * step
    iend = istart + stridesamp          # exclusive end
    mixed = exp(-1j * phase[istart:iend]) * data[istart:iend]
    out[step] = 2 * mixed.mean() if singlesided else mixed.mean()

output.sample_rate = 1 / stride
```

### 使用例

```python
import numpy as np
from gwexpy.timeseries import TimeSeries

# 正弦波を生成
A, f0, phi0 = 2.5, 30.0, np.pi/4
sample_rate = 1024.0
duration = 10.0
n = int(duration * sample_rate)
t = np.arange(n) / sample_rate

data = A * np.cos(2 * np.pi * f0 * t + phi0)
ts = TimeSeries(data, dt=1/sample_rate, unit='V')

# 位相配列を作成
phase = 2 * np.pi * f0 * t

# ヘテロダイン（singlesided=True がデフォルト）
het = ts.heterodyne(phase, stride=1.0)

# 期待値: A * exp(1j * phi0)
print(f"Amplitude: {np.mean(np.abs(het.value)):.3f}")  # ≈ 2.5
print(f"Phase: {np.mean(np.angle(het.value)):.3f}")    # ≈ 0.785 (π/4)
```

---

## `lock_in` （ロックイン増幅器）

```python
TimeSeries.lock_in(
    f0: float | Quantity | None = None,
    *,
    phase: array_like | None = None,
    fdot: float | Quantity = 0.0,
    fddot: float | Quantity = 0.0,
    phase_epoch: float | None = None,
    phase0: float = 0.0,
    stride: float | Quantity | None = None,
    bandwidth: float | Quantity | None = None,
    singlesided: bool = True,
    output: str = "amp_phase",
    deg: bool = True,
    **kwargs
) -> TimeSeries | tuple
```

### 説明

ロックイン増幅（復調 + 平均化またはフィルタリング）を実行します。
2つの動作モードがあり、`bandwidth` パラメータで選択されます。

### 動作モード

**LPF モード（`bandwidth` 指定時）**:

- `baseband(lowpass=bandwidth, ...)` を使用して復調とローパスフィルタリング
- `stride` は指定**禁止**（ValueError）

**ストライド平均モード（`bandwidth` 未指定時）**:

- `heterodyne(phase, stride, ...)` を使用して復調と固定ストライド平均化
- `stride` は**必須**

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `f0` | float または Quantity | None | 中心周波数（Hz）。`phase` と**排他** |
| `phase` | array_like | None | 明示的位相配列（rad）。`f0` 系パラメータと**排他** |
| `fdot` | float または Quantity | 0.0 | 周波数微分（Hz/s）|
| `fddot` | float または Quantity | 0.0 | 周波数2階微分（Hz/s²）|
| `phase_epoch` | float | None | 位相モデル基準エポック |
| `phase0` | float | 0.0 | 初期位相オフセット（rad）|
| `stride` | float または Quantity | None | 平均化時間ステップ（秒）。`bandwidth` と**排他** |
| `bandwidth` | float または Quantity | None | LPF 帯域幅（Hz）。`stride` と**排他** |
| `singlesided` | bool | True | True で振幅2倍 |
| `output` | str | "amp_phase" | 出力形式: `'complex'`, `'amp_phase'`, `'iq'` |
| `deg` | bool | True | `'amp_phase'` 時に位相を度で返すか |
| `**kwargs` | - | - | LPF モード時に `baseband()` に渡す引数 |

### 戻り値

| `output` | 戻り値 |
|----------|--------|
| `'complex'` | 複素 TimeSeries |
| `'amp_phase'` | `(amplitude, phase)` タプル |
| `'iq'` | `(I, Q)` タプル（real/imag 成分）|

### 例外条件

| 条件 | 例外 |
|------|------|
| `phase` と `f0`/`fdot`/`fddot`/`phase_epoch`/`phase0` の同時指定 | `ValueError` |
| `phase` も `f0` も未指定 | `ValueError` |
| `bandwidth` と `stride` の同時指定 | `ValueError` |
| `bandwidth` も `stride` も未指定 | `ValueError` |
| `output` が無効な値 | `ValueError` |

### 位相指定優先規則

`phase` パラメータは**最優先**されます。`phase` を指定する場合、
`f0`/`fdot`/`fddot`/`phase_epoch`/`phase0` パラメータは指定できません
（デフォルト値を除く）。これにより、曖昧な設定を防止します。

### 使用例

**ストライド平均モード（f0 から位相生成）:**

```python
# 固定周波数での復調
amp, phase = ts.lock_in(f0=100.0, stride=1.0, output='amp_phase')
```

**ストライド平均モード（明示的位相）:**

```python
# カスタム位相配列で復調
phase_arr = 2 * np.pi * 100.0 * ts.times.value
result = ts.lock_in(phase=phase_arr, stride=1.0, output='complex')
```

**LPF モード:**

```python
# ローパスフィルタを使用した復調
amp, phase = ts.lock_in(f0=100.0, bandwidth=10.0, output='amp_phase')
```

**チャープ信号の追跡:**

```python
# 周波数変化する信号の復調
result = ts.lock_in(f0=100.0, fdot=0.1, stride=1.0, output='complex')
```

---

## 関連項目

- `heterodyne()`: 位相ヘテロダインとストライド平均
- `baseband()`: ベースバンド復調（LPF + リサンプル）
- `mix_down()`: 複素オシレータとのミキシング（低レベル）
- `_build_phase_series()`: 内部ヘルパー（f0 系から位相生成）
