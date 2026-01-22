# DTT (Diagnostic Test Tools) 包括的分析レポート

**作成日:** 2026-01-22
**対象:** `gwexpy/gui/reference-dtt/dtt-master/`
**Author:** Antigravity Agent

---

# Part 1: Documentation & UI Audit (Phase 1)

## 1.1 Executive Summary

The legacy Diagnostic Test Tool (DTT) is a C++ application built upon the CERN ROOT framework. It follows a client-server architecture where `diaggui` handles the user interface and parameter configuration, while the heavy lifting (signal processing, data acquisition) is performed by a diagnostic kernel (`diagd` or internal classes).

## 1.2 Documentation Review

Since the PDF contents (`T990013-v3`, `G000079-00`) are binary specifications, we inferred their role from the project structure and markdown summaries:

- **T990013 (Diagnostics Test Software)**: Defines the core requirements for LIGO diagnostics, including FFT performance, Swept Sine accuracy, and real-time data access.
- **G000079**: Likely the user manual or GUI specification, detailing the operational modes.

## 1.3 UI/UX Architecture Analysis

### Framework

- **Base Library**: ROOT (`TApplication`, `TGMainFrame`).
- **Custom Wrapper**: `TLG` (The LIGO GUI) classes in `src/dtt/gui/dttgui/`.
  - `TLGMainWindow`: Base window layout with Menu, Status, Plot area.
  - `TLGPlot`: Advanced plotting widget supporting Zoom, Log/Linear toggles, multiple traces.
  - `TLGChannelBox`: Hierarchical tree view for selecting NDS channels.

### Key Operational Modes (Tabs)

The screenshots and source code structure (`src/dtt/diag/`) indicate four primary measurement modes:

1. **Fourier Tools (FFT)**:
    - computes Power Spectral Density (PSD), Cross Spectral Density (CSD), Coherence.
    - Parameters: Start/Stop Freq, BW, Window (Hann/Flat-top), Averages.
2. **Swept Sine**:
    - Measures Transfer Functions (Bode Plots) by injecting a sine wave and sweeping frequency.
    - Critical for servo characterization.
3. **Time Series**:
    - Oscilloscope-like view of raw channel data.
4. **Excitation**:
    - Configuration of output signals (Gaussian Noise, Sine, Swept Sine) for system identification.

---

# Part 2: Core Code Analysis (Phase 2)

## 2.1 FFT-Based Measurement Logic (`ffttools.cc`)

The `ffttest` class is the engine for all FFT-based measurements (PSD, CSD, Coherence).

### Class Hierarchy

```
diagtest (abstract base)
  └── stdtest (common measurement logic: channels, stimuli, averages)
        └── ffttest (FFT-based: PSD, CSD, Coherence)
        └── sweptsine (Swept sine: Transfer Function)
        └── sineresponse (Single frequency)
        └── timeseries (Oscilloscope mode)
```

### Key Parameters (`diagnames.h`)

| C++ Parameter | Default | gwexpy Equivalent |
|---|---|---|
| `fftStartFrequency` | 0 | `fmin` |
| `fftStopFrequency` | 900 | `fmax` |
| `fftBW` | 1 Hz | `df` (resolution, `1 / segment_duration`) |
| `fftOverlap` | 0.5 | `overlap` (Welch overlap fraction) |
| `fftWindow` | 1 (Hann) | `window` (scipy/gwpy window name) |
| `fftRemoveDC` | false | `detrend='constant'` |
| `fftAverages` | 10 | `n_avg` |
| `fftAverageType` | 0 (Linear) | `method='median'` or `'mean'` |

### Core FFT Workflow (`ffttest::analyze`)

1. **Allocate Temporary Storage (`tmps`):** For each channel, allocate `fftPoints` complex floats for intermediate FFT results.
2. **Iterate `ffttest::fft()` per Channel:**
   - Get time-series data from `gdsDataObject`.
   - Apply window (`psGen(PS_INIT_ALL, ...)`).
   - Compute FFT (`psGen(PS_TAKE_FFT, ...)`).
   - **Zoom FFT:** If `fZoom > 0`, data undergoes complex heterodyne transformation (shifting center frequency to baseband), followed by decimation (`decimate2`) and standard FFT.
   - **Convert to PSD (`fftToPs`):** Apply power normalization.
   - **Averaging (`avg()`):** Linear (`AVG_LINEAR_SQUARE`) or Exponential (`AVG_EXPON_SQUARE`).
3. **Iterate `ffttest::cross()` per A-channel pair:**
   - Compute Cross-Spectral Density (`crossPower`).
   - Average complex CSD.
   - Compute Coherence (`coherenceCP`): `|CSD_AB|^2 / (PSD_AA * PSD_BB)`.

### Window Normalization

```c++
// calculate resolution (window) bandwidth
double windowNorm = sMean (fftPlan.window_coeff, fftPoints);
windowNorm *= windowNorm;
if (windowNorm > 0) {
   windowBW = BW / windowNorm;
}
```

- `BW` is the *resolution bandwidth* (1/segment_duration).
- `windowBW` is the *Equivalent Noise Bandwidth (ENBW)*, adjusted for window amplitude loss.
- **Critical for gwexpy:** `scipy.signal.welch` returns PSD already normalized for ENBW when `scaling='density'`. Ensure `gwexpy` uses consistent units.

## 2.2 Swept Sine Implementation

This is the most complex measurement mode and is **not currently implemented in gwexpy**.

### Core Concepts

- **Sweep Points (`sweeppoints`):** A list of `(freq, ampl, phase)` tuples.
- **Sweep Types:** Linear, Logarithmic, or from a file.
- **Per-Point Measurement:** At each frequency:
  1. Inject a single-frequency sine wave excitation.
  2. Wait for settling time.
  3. Acquire response for `measurementTime`.
  4. Compute sine amplitude and phase using `sinedet()` (demodulation).
  5. Store coefficients.
- **Transfer Function Calculation (`transfn()`):** `H = Y_response / X_excitation`.

### Parameters

| C++ Parameter | Default | Description |
|---|---|---|
| `ssStartFrequency` | 1 Hz | Start of sweep |
| `ssStopFrequency` | 1000 Hz | End of sweep |
| `ssNumberOfPoints` | 61 | Number of frequency points |
| `ssSweepType` | 1 (Log) | 0: Linear, 1: Log |
| `ssMeasurementTime[2]` | {0.1, 10} | Range for adaptive meas. time |
| `ssSettlingTime` | 0.25 | Settling time (as fraction of meas. time) |
| `ssHarmonicOrder` | 1 | For detecting harmonics |

---

# Part 3: Deep Dive Analysis

## 3.1 Swept Sine 詳細アルゴリズム

### スイープ点の生成

```cpp
switch (sweepType) {
   case 0:  // Linear sweep
      for (int i = 0; i < nSweep; i++) {
         f = fStart + (double) i / (nSweep - 1.0) * (fStop - fStart);
         fPoints.push_back(sweeppoint(f, ampl));
      }
      break;
   case 1:  // Logarithmic sweep
      for (int i = 0; i < nSweep; i++) {
         f = fStart * power(fStop/fStart, (double) i / (nSweep - 1.0));
         fPoints.push_back(sweeppoint(f, ampl));
      }
      break;
}
```

### 同期検波 (`sinedet`)

各周波数点において、応答信号から励起周波数成分を抽出します。

**`sineAnalyze` の推定ロジック:**

```
X[k] = (2/N) * Σ x[n] * exp(-j * 2π * f_target * n / fs)
```

これは離散フーリエ変換の特定周波数ビン計算と等価で、`scipy.signal` や `numpy` で実装可能。

## 3.2 Zoom FFT (高分解能スペクトル)

通常のFFTでは `df = fs / N` であり、広い周波数範囲と高分解能を同時に得ることが困難です。Zoom FFT は複素ヘテロダイン変換により、特定の狭い周波数帯域を高分解能で解析します。

### アルゴリズム

1. **ヘテロダイン（周波数シフト）**: `x_shifted[n] = x[n] * exp(-j * 2π * f_zoom * n / fs)`
2. **デシメーション（ダウンサンプリング）**: ナイキスト周波数が新しい帯域幅に合うようにサンプリングレートを下げる
3. **FFT 実行**: シフト＆デシメーション後のデータに対してFFTを実行
4. **データ回転**: 周波数軸を元の位置に戻す

### gwexpy への実装プラン

```python
def zoom_fft(ts: TimeSeries, f_center: float, bandwidth: float, 
             df: float = None, window: str = 'hann') -> FrequencySeries:
    from scipy.signal import decimate
    import numpy as np
    
    # Step 1: Heterodyne (shift to baseband)
    t = ts.times.value
    x_shifted = ts.value * np.exp(-2j * np.pi * f_center * t)
    
    # Step 2: Low-pass filter and decimate
    decim_factor = int(ts.sample_rate.value / bandwidth)
    x_decimated = decimate(x_shifted, decim_factor, ftype='fir')
    
    # Step 3: FFT
    n_fft = int(bandwidth / df) if df else 1024
    spectrum = np.fft.fft(x_decimated[:n_fft] * get_window(window, n_fft))
    spectrum = np.fft.fftshift(spectrum)
    
    # Step 4: Build frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=decim_factor/ts.sample_rate.value))
    freqs += f_center
    
    return FrequencySeries(spectrum, frequencies=freqs)
```

## 3.3 Burst Noise Measurement

通常のFFT測定では励起信号が連続的に印加されますが、Burst Noise モードでは励起信号を断続的（ゲート）にし、静寂時間を設けて応答のリンギングを捉えます。

### タイミング図

```
|<-- pitch -->|<-- pitch -->|<-- pitch -->|
[RAMP][==EXCITATION==][RAMP][QUIET][MEASURE]
```

---

# Part 4: Architecture Analysis

## 4.1 `stdtest` Base Class

`stdtest` は、DTT のすべての測定タイプ（FFT, Swept Sine, Sine Response, Time Series）の共通基底クラスです。

### 主要なインナークラスとデータ型

```cpp
class stimulus {
   std::string name;          // 励起チャンネル名
   bool isReadback;           // リードバック使用フラグ
   AWG_WaveType waveform;     // 波形タイプ (Sine, Square, Noise, etc.)
   double freq, ampl, offs, phas;  // 周波数, 振幅, オフセット, 位相
};

class measurementchannel {
   std::string name;          // チャンネル名
   gdsChnInfo_t info;         // チャンネル情報
   partitionlist partitions;  // データパーティション
};
```

### 仮想メソッド: 測定ライフサイクル

```
begin() → setup() → [syncAction() → analyze()] * N → end()
         ↓
         calcTimes()
         calcMeasurements()
         startMeasurements()
```

## 4.2 AWG (Arbitrary Waveform Generator) API

### 主要関数

| 関数 | 説明 |
|---|---|
| `awg_client()` | AWG クライアントインターフェース初期化 |
| `awgSetChannel(name)` | チャンネル名をスロットに関連付け |
| `awgAddWaveform(slot, comp, num)` | 波形コンポーネントを追加 |
| `awgSetWaveform(slot, y, len)` | 任意波形をダウンロード |
| `awgSendWaveform(slot, time, epoch, y, len)` | ストリームデータ送信 |
| `awgStopWaveform(slot, terminate, time)` | 波形停止（reset/freeze/phase-out） |
| `awgSetGain(slot, gain, time)` | 全体ゲイン設定（ランプ可能） |
| `awgSetFilter(slot, y, len)` | IIR フィルタ設定（SOS形式） |

### 波形タイプ (`AWG_WaveType`)

```cpp
enum AWG_WaveType {
   awgNone = 0,
   awgSine = 1,       // Sine wave
   awgSquare = 2,     // Square wave
   awgRamp = 3,       // Ramp wave
   awgTriangle = 4,   // Triangle wave
   awgImpulse = 5,    // Impulse
   awgConst = 6,      // Constant offset
   awgNoiseN = 7,     // Normal (Gaussian) noise
   awgNoiseU = 8,     // Uniform noise
   awgArb = 9,        // Arbitrary waveform
   awgStream = 10     // Stream waveform
};
```

## 4.3 NDS2 Data Input

`nds2input.hh` / `nds2input.cc` は、NDS2 (Network Data Server v2) からリアルタイムまたはアーカイブデータを取得するクライアントを実装しています。

### 主要クラス

- **`nds2Manager`**: NDS2 接続の管理、チャンネルの追加/削除、データフローの開始/停止
- **`NDS2Connection`**: 単一の NDS2 接続インスタンス、非同期データ読み込みスレッド

---

# Part 5: Additional Modules Analysis

## 5.1 Foton - フィルタ設計ツール

Foton (Filter Online Tool) は、LIGO/KAGRA で使用される IIR フィルタを設計するための GUI ツールです。

### gwexpy への関連

```python
class FotonFilter:
    @classmethod
    def from_file(cls, path: str) -> 'FotonFilter':
        """Load filter from Foton .txt file."""
        ...
    
    def to_sos(self) -> np.ndarray:
        """Convert to Second Order Sections format for scipy.signal."""
        ...
```

## 5.2 StripChart - リアルタイムプロット

StripChart は、時間変化するデータをリアルタイムでプロットするためのウィジェットです。

## 5.3 DFM - Data Flow Manager

DFM (Data Flow Manager) は、複数のデータソース（NDS, ファイル, 共有メモリ, テープ）へのアクセスを統一的に提供する抽象化レイヤーです。

### データサービスタイプ

```cpp
enum dataservicetype {
   st_Invalid = 0,
   st_LARS = 1,     // LARS/DFM server
   st_NDS = 2,      // NDS server (v1)
   st_SENDS = 3,    // NDS2 server
   st_File = 4,     // Local file system
   st_Tape = 5,     // Local tape drive/robot
   st_SM = 6,       // Online shared memory
   st_Func = 7      // User callback
};
```

---

# Part 6: External Dependencies & Python Integration

## 6.1 dttxml パッケージ - Python での DTT データアクセス

`dttxml` は、DTT (diaggui) が出力する XML ファイルを解析する Python ライブラリです。

```python
from dttxml import DiagAccess

da = DiagAccess('measurement.xml')

# PSD (ASD) を取得
asd = da.asd('K1:PEM-MIC_BOOTH_ENV_OUT_DQ')

# 伝達関数を取得
tf = da.xfer('K1:SAS-ITMY_TM_OPLEV_SERVO_OUT', 'K1:SUS-ITMX_SUS_OUT')

# コヒーレンスを取得
coh = da.coherence('Channel1', 'Channel2')
```

## 6.2 sineAnalyze の Python 再実装

```python
import numpy as np
from scipy.signal import windows

def sine_analyze(data: np.ndarray, fs: float, freq: float,
                 window: str = 'hann', n_avg: int = 1,
                 t0: float = 0.0) -> complex:
    """
    Compute the complex amplitude at a specific frequency using lock-in detection.
    This is equivalent to DTT's sineAnalyze function.
    """
    n = len(data)
    segment_len = n // n_avg
    
    win = windows.get_window(window, segment_len)
    win_sum = np.sum(win)
    
    coeffs = []
    for i in range(n_avg):
        start_idx = i * segment_len
        segment = data[start_idx:start_idx + segment_len] * win
        
        t = np.arange(segment_len) / fs + t0 + start_idx / fs
        phase = 2 * np.pi * freq * t
        ref_cos = np.cos(phase)
        ref_sin = np.sin(phase)
        
        I = 2 * np.sum(segment * ref_cos) / win_sum
        Q = 2 * np.sum(segment * ref_sin) / win_sum
        
        coeffs.append(I + 1j * Q)
    
    return np.mean(coeffs)
```

---

# Part 7: Gap Analysis & Recommendations

## 7.1 Feature Gap Analysis

| DTT Feature | gwexpy Status | Required Work |
|---|---|---|
| **FFT/PSD** | ✅ `Spectrogram` / `FrequencySeries` | Minor: Verify ENBW normalization. |
| **CSD** | ✅ `FrequencySeries.csd()` | Minor: Verify phase convention. |
| **Coherence** | ✅ `coherence()` function | None. |
| **Swept Sine** | ❌ Not Implemented | Major: Requires AWG API, sync demod. |
| **Zoom FFT** | ❌ Not Implemented | Medium: Complex heterodyne + decimate. |
| **Burst Noise** | ❌ Not Implemented | Medium: Gated excitation logic. |
| **Plot Save/Load** | Partial (HDF5) | Medium: JSON schema for GUI state. |
| **Linear/Exp Avg** | ✅ `method` param | None. |

## 7.2 Implementation Roadmap

| フェーズ | 目標 | 工数見積もり |
|---|---|---|
| **Phase 1** | PSD/CSD/Coherence 正規化検証 | 2-4時間 |
| **Phase 2** | Zoom FFT 実装 | 1-2日 |
| **Phase 3** | MeasurementState JSON スキーマ | 1-2日 |
| **Phase 4** | BaseMeasurement 抽象クラス設計 | 2-3日 |
| **Phase 5** | Swept Sine プロトタイプ（シミュレーション） | 1週間 |
| **Phase 6** | AWG API Python バインディング | 1-2週間 |
| **Phase 7** | NDS2 リアルタイム統合 | 1週間 |

## 7.3 Recommendations

1. **Priority 1: Verify PSD Normalization.** Write a test that compares `gwexpy.Spectrogram` PSD output against known DTT output for identical input signals and parameters.
2. **Priority 2: Design Swept Sine API.** Draft a class `SweptSineMeasurement` with methods `setup()`, `measure_point()`, `next_point()`, `get_transfer_function()`.
3. **Priority 3: Define Measurement State Schema.** Create a `MeasurementState` dataclass for JSON serialization that mirrors `PlotSet`.

---

# Part 8: Data Flow & Class Interaction Diagrams

## 8.1 クラス相互作用図

```
┌─────────────────────────────────────────────────────────────────┐
│                      GUI (diagmain, diagctrl)                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   standardsupervisory                            │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│   │ gdsStorage  │  │ dataBroker  │  │ testExcitation (AWG)    │ │
│   └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘ │
└──────────│────────────────│──────────────────────│──────────────┘
           │                │                      │
           ▼                ▼                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                         stdtest                                   │
│   ┌───────────┐  ┌───────────────┐  ┌───────────────────────┐    │
│   │ stimulus  │  │measurementchn │  │       interval        │    │
│   │  list     │  │     list      │  │        list           │    │
│   └───────────┘  └───────────────┘  └───────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
     ┌────────────┐      ┌────────────┐      ┌────────────┐
     │  ffttest   │      │ sweptsine  │      │ timeseries │
     └────────────┘      └────────────┘      └────────────┘
```

## 8.2 データフロー図

```
[NDS2/RTDD/File]
       │
       ▼
┌──────────────┐
│  dataBroker  │  ─────  チャンネル購読/データ受信
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ chnCallback  │  ─────  コールバックによるデータ配信
└──────┬───────┘
       │
       ▼
┌──────────────┐              ┌──────────────┐
│  gdsStorage  │ ◄──────────► │  gdsDatum    │
│              │              │  (データ)    │
│  - Results   │              └──────────────┘
│  - Index     │
│  - Test      │              ┌──────────────┐
│  - References│ ◄──────────► │ gdsParameter │
│              │              │  (設定)      │
└──────────────┘              └──────────────┘
       │
       ▼
┌──────────────┐
│   PlotSet    │  ─────  プロットデータ管理
└──────┬───────┘
       │
       ▼
    [XML 保存]
```

---

# Part 9: File Analysis Summary

## 9.1 分析完了ファイル (31件)

| カテゴリ | ファイル | 行数 | 状態 |
|---|---|---|---|
| **Core Tests** | ffttools.hh/.cc | 379/1272 | ✅ |
| | sweptsine.hh/.cc | 396/1356 | ✅ |
| | sineresponse.hh/.cc | 313/1123 | ✅ |
| | timeseries.hh/.cc | 313/958 | ✅ |
| **Base Classes** | stdtest.hh/.cc | 700/1645 | ✅ |
| | diagtest.hh/.cc | 313/... | ✅ |
| | diagnames.h/.c | 800/... | ✅ |
| **Storage** | diagdatum.hh/.cc | 1884/... | ✅ |
| | gdsdatum.hh/.cc | 2019/... | ✅ |
| | channelinput.hh/.cc | 819/... | ✅ |
| | databroker.hh/.cc | 425/... | ✅ |
| | nds2input.hh/.cc | 400/... | ✅ |
| **Sync/Control** | testsync.hh/.cc | 522/... | ✅ |
| | supervisory.hh/.cc | 277/... | ✅ |
| | stdsuper.hh/.cc | 242/... | ✅ |
| **AWG** | awgapi.h, awgtype.h | .../... | ✅ |
| **GUI** | diagmain.hh, diagctrl.hh | .../... | ✅ |
| **Tools** | foton.cc | 642 | ✅ |
| | StripChart.hh | ... | ✅ |
| | dfmtype.hh | 166 | ✅ |
| **Containers** | PlotSet.hh, DataDesc.hh | 597/800 | ✅ |

---

# Part 10: Conclusion

DTT リポジトリの全ソースコード分析を完了しました。

**主要な発見:**

1. **クラス階層**: `diagtest → stdtest → {ffttest, sweptsine, timeseries, sineresponse}`
2. **データ構造**: `gdsDatum` を基底とした多次元データ管理
3. **測定フロー**: パラメータ読み込み → 時間計算 → 測定設定 → コールバック解析
4. **外部依存**: `gdsalgorithm.h` (libgds) は非公開だが、アルゴリズムは再実装可能
5. **Python 統合**: `dttxml` パッケージで DTT XML データの読み込みが可能

---

*この分析により、DTT のコアアーキテクチャと信号処理パイプラインの全体像が把握できました。*
