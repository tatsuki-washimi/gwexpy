# Noise

検出器・環境ノイズのモデル化とASDからの波形生成のユーティリティです。
`gwexpy.noise` は以下の2つのサブモジュールで構成されます：

- `gwexpy.noise.asd`: 振幅スペクトル密度(ASD)を生成する関数 (`FrequencySeries` を返す)
- `gwexpy.noise.wave`: 時系列波形を生成する関数 (`TimeSeries` を返す)

## gwexpy.noise.wave

時系列波形を生成する関数群です。すべての関数は `TimeSeries` を返します。

### ノイズ生成

| 関数 | 説明 |
|------|------|
| `gaussian(duration, sample_rate, std=1.0, mean=0.0, ...)` | ガウス（正規）分布のホワイトノイズ |
| `uniform(duration, sample_rate, low=-1.0, high=1.0, ...)` | 一様分布のホワイトノイズ |
| `colored(duration, sample_rate, exponent, amplitude=1.0, ...)` | べき乗則に従う色付きノイズ |
| `white_noise(duration, sample_rate, amplitude=1.0, ...)` | ホワイトノイズ（exponent=0） |
| `pink_noise(duration, sample_rate, amplitude=1.0, ...)` | ピンクノイズ（1/f^0.5 スペクトル） |
| `red_noise(duration, sample_rate, amplitude=1.0, ...)` | レッド/ブラウニアンノイズ（1/f スペクトル） |
| `from_asd(asd, duration, sample_rate, ...)` | ASD から色付きノイズを生成 |

### 周期波形

| 関数 | 説明 |
|------|------|
| `sine(duration, sample_rate, frequency, ...)` | サイン波 |
| `square(duration, sample_rate, frequency, duty=0.5, ...)` | 矩形波 |
| `sawtooth(duration, sample_rate, frequency, width=1.0, ...)` | のこぎり波 |
| `triangle(duration, sample_rate, frequency, ...)` | 三角波 |
| `chirp(duration, sample_rate, f0, f1, method='linear', ...)` | スウェプトサイン（周波数掃引信号） |

### 過渡信号

| 関数 | 説明 |
|------|------|
| `step(duration, sample_rate, t_step=0.0, amplitude=1.0, ...)` | ステップ（ヘビサイド）関数 |
| `impulse(duration, sample_rate, t_impulse=0.0, amplitude=1.0, ...)` | インパルス信号 |
| `exponential(duration, sample_rate, tau, decay=True, ...)` | 指数関数（減衰/増幅） |

### 使用例

```python
from gwexpy.noise.wave import sine, gaussian, chirp, from_asd
from gwexpy.noise.asd import from_pygwinc

# サイン波
wave = sine(duration=1.0, sample_rate=1024, frequency=10.0)

# ガウスノイズ
noise = gaussian(duration=1.0, sample_rate=1024, std=0.1)

# スウェプトサイン
sweep = chirp(duration=1.0, sample_rate=1024, f0=10, f1=100)

# ASDからノイズ生成
asd = from_pygwinc('aLIGO', fmin=4.0, fmax=1024.0, df=0.01)
noise = from_asd(asd, duration=128, sample_rate=2048, t0=0)
```
