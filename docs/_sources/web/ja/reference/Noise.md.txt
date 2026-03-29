# Noise

検出器および環境ノイズモデル用のユーティリティ。ASDヘルパーと時間領域の合成機能を含みます。
`gwexpy.noise` モジュールは以下の2つのサブモジュールに分かれています：

- `gwexpy.noise.asd`: 振幅スペクトル密度 (`FrequencySeries`) を返す関数
- `gwexpy.noise.wave`: 時系列波形 (`TimeSeries`) を返す関数

## gwexpy.noise.asd

振幅スペクトル密度を生成する関数。すべての関数は `FrequencySeries` を返します。

### 検出器ノイズモデル

| 関数 | 説明 |
|------|------|
| `from_pygwinc(ifo, *, quantity='strain', fmin=10, fmax=8192, df=1.0)` | pyGWINC 検出器ノイズモデル (aLIGO, AdV など) から ASD を生成 |
| `from_obspy(model, *, quantity='displacement', fmin=0.01, fmax=100)` | ObsPy 地震ノイズモデル (NLNM, NHNM) から ASD を生成 |

### カラードノイズ ASD

| 関数 | 説明 |
|------|------|
| `power_law(exponent, amplitude=1.0, f_ref=1.0, frequencies=...)` | べき乗則 ASD (f^-exponent) を生成 |
| `white_noise(amplitude=1.0, frequencies=...)` | ホワイトノイズ ASD (フラットスペクトル) |
| `pink_noise(amplitude=1.0, frequencies=...)` | ピンクノイズ ASD (1/f^0.5) |
| `red_noise(amplitude=1.0, frequencies=...)` | レッド/ブラウニアンノイズ ASD (1/f) |

### 地磁気ノイズモデル

| 関数 | 説明 |
|------|------|
| `schumann_resonance(harmonics=8, frequencies=...)` | シューマン共振モデル (~7.83 Hz とその高調波) |
| `geomagnetic_background(frequencies=...)` | 背景地磁気ノイズモデル |

### スペクトル線形状

| 関数 | 説明 |
|------|------|
| `lorentzian_line(center, width, amplitude=1.0, frequencies=...)` | ローレンツ線形状 |
| `gaussian_line(center, sigma, amplitude=1.0, frequencies=...)` | ガウス線形状 |
| `voigt_line(center, sigma, gamma, amplitude=1.0, frequencies=...)` | フォークトプロファイル (ガウスとローレンツの畳み込み) |

## gwexpy.noise.wave

時系列波形を生成する関数。すべての関数は `TimeSeries` を返します。

### ノイズジェネレータ

| 関数 | 説明 |
|------|------|
| `gaussian(duration, sample_rate, std=1.0, mean=0.0, ...)` | ガウス (正規) ホワイトノイズ |
| `uniform(duration, sample_rate, low=-1.0, high=1.0, ...)` | 一様ホワイトノイズ |
| `colored(duration, sample_rate, exponent, amplitude=1.0, ...)` | べき乗則カラードノイズ |
| `white_noise(duration, sample_rate, amplitude=1.0, ...)` | ホワイトノイズ (exponent=0) |
| `pink_noise(duration, sample_rate, amplitude=1.0, ...)` | ピンクノイズ (1/f^0.5 スペクトル) |
| `red_noise(duration, sample_rate, amplitude=1.0, ...)` | レッド/ブラウニアンノイズ (1/f スペクトル) |
| `from_asd(asd, duration, sample_rate, ...)` | ASD からカラードノイズを生成 |

### 周期波形

| 関数 | 説明 |
|------|------|
| `sine(duration, sample_rate, frequency, ...)` | サイン波 |
| `square(duration, sample_rate, frequency, duty=0.5, ...)` | 矩形波 |
| `sawtooth(duration, sample_rate, frequency, width=1.0, ...)` | のこぎり波 |
| `triangle(duration, sample_rate, frequency, ...)` | 三角波 |
| `chirp(duration, sample_rate, f0, f1, method='linear', ...)` | 周波数掃引コサイン (チャープ) |

### 過渡信号

| 関数 | 説明 |
|------|------|
| `step(duration, sample_rate, t_step=0.0, amplitude=1.0, ...)` | ステップ (ヘビサイド) 関数 |
| `impulse(duration, sample_rate, t_impulse=0.0, amplitude=1.0, ...)` | インパルス信号 |
| `exponential(duration, sample_rate, tau, decay=True, ...)` | 指数関数 (減衰/成長) |

## 使用例

```python
from gwexpy.noise.wave import sine, gaussian, chirp, from_asd
from gwexpy.noise.asd import from_pygwinc, schumann_resonance

# サイン波
wave = sine(duration=1.0, sample_rate=1024, frequency=10.0)

# ガウスノイズ
noise = gaussian(duration=1.0, sample_rate=1024, std=0.1)

# チャープ (周波数掃引サイン波)
sweep = chirp(duration=1.0, sample_rate=1024, f0=10, f1=100)

# pyGWINC からの検出器歪み ASD
asd = from_pygwinc('aLIGO', quantity='strain', fmin=4.0, fmax=1024.0, df=0.01)
noise = from_asd(asd, duration=128, sample_rate=2048, t0=0)

# シューマン共振モデル
sch_asd = schumann_resonance(harmonics=5)
```
