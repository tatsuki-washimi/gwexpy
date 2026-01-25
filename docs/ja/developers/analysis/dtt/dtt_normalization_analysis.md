# DTT FFT 正規化ロジック深掘り分析

**作成日:** 2026-01-22  
**対象:** DTT `ffttools.cc` の Window Normalization (ENBW) と `scipy.signal.welch` の比較  
**目的:** Phase 1 タスク「PSD/CSD 正規化の検証」のための技術資料

---

## 1. DTT の Window Normalization ロジック

### 1.1 元のコード (C++)

**DTT の `ffttools.cc` (行583-589) から実際に抽出:**

```c++
// calculate resolution (window) bandwidth
double windowNorm = sMean(fftPlan.window_coeff, fftPoints);
windowNorm *= windowNorm;
if (windowNorm > 0) {
   windowBW = BW / windowNorm;
}
else {
   windowBW = BW;
}
```

**変数の定義:**
- `BW`: Resolution Bandwidth = `fs / N` (サンプリング周波数 / FFT点数)
- `fftPlan.window_coeff`: 窓関数の係数配列（例: Hann, Flattop）
- `fftPoints`: FFT点数（セグメント長）N
- `sMean(w, N)`: 窓関数の平均値 = `(1/N) * Σw[n]` (GDS ライブラリの関数)
- `windowNorm`: 窓関数の平均値の二乗 = `(Σw[n] / N)²`
- `windowBW`: 実効ノイズ帯域幅 (Equivalent Noise Bandwidth, ENBW)

### 1.1b PSD計算の実際のフロー (ffttools.cc 行1059-1082)

```c++
// calculate PSD and do averaging
// Use DATA_REAL to treat bin at f=0 special
if (fftToPs(points, (fStart >= 1E-12) ? DATA_COMPLEX : DATA_REAL, 
            tmps[resultnum].fft, tmps[resultnum].x) < 0) {
   return false;
}

// do averaging and store result
avg_specs avgprm;
avgprm.avg_type = (averageType != 1) ? 
   AVG_LINEAR_SQUARE : AVG_EXPON_SQUARE;
avgprm.dataset_length = points;
avgprm.data_type = DATA_REAL;
avgprm.number_of_averages = averages;
int num_so_far = measnum;

if (avg(&avgprm, 1, tmps[resultnum].x, &num_so_far, 
        (float*) res->value) < 0) {
   return false;
}
```

**重要な発見:**
- `fftToPs` 関数が FFT 結果を PSD に変換（GDS ライブラリ関数）
- `avg` 関数が平均化処理を実行
- `windowBW` は **パラメータとして保存されるのみ** で、PSD計算には直接使われていない可能性がある

### 1.2 数式による表現

```
windowNorm = (1/N * Σ w[n])²
ENBW = BW / windowNorm = (fs / N) / (1/N * Σ w[n])²
```

ここで、`Σ w[n]` は窓関数の総和（DCゲイン）を表す。

### 1.3 物理的意味

窓関数を適用すると、スペクトルのメインローブが広がり、実質的な周波数分解能が低下する。ENBW は、矩形窓（窓なし）と同じノイズパワーを通過させる等価な帯域幅を表す。

**主要な窓関数の ENBW 係数:**

| Window Type | ENBW Factor | DCゲイン (Σw/N) |
|-------------|-------------|-----------------|
| Rectangle   | 1.00        | 1.00            |
| Hann        | 1.50        | 0.50            |
| Hamming     | 1.36        | 0.54            |
| Blackman    | 1.73        | 0.42            |
| Flattop     | 3.77        | 0.22            |

DTT の実装では、`windowNorm = (DCゲイン)²` であり、`ENBW = BW / (DCゲイン)²` となる。

---

## 2. scipy.signal.welch の正規化ロジック

### 2.1 実装の確認

`scipy.signal.welch` の内部では、以下の正規化が行われている（`scaling='density'` の場合）：

```python
# scipy/signal/spectral.py (概念コード)
def welch(..., scaling='density'):
    # ...
    win = get_window(window, nperseg)
    scale = 1.0 / (fs * (win*win).sum())
    
    for segment in segments:
        Pxx += np.abs(np.fft.rfft(segment * win))**2 * scale
    
    Pxx /= n_averages
```

**ポイント:**
- `scale = 1.0 / (fs * (win*win).sum())` という正規化係数が使われている。
- これは、パワースペクトル密度 [単位²/Hz] を得るための標準的な正規化。

### 2.2 数式による表現

```
scale = 1 / (fs * Σ(w[n]²))
PSD[k] = (1/M) * Σ_m |FFT(x_m * w)|² * scale
```

ここで:
- `Σ(w[n]²)`: 窓関数のパワー総和（二乗和）
- `M`: 平均化回数

### 2.3 ENBW との関係

ENBW は以下の式で定義される：

```
ENBW = N * Σ(w[n]²) / (Σ w[n])²
```

これを使うと、scipy の正規化は次のように書き直せる：

```
scale = 1 / (fs * Σ(w[n]²))
      = 1 / (fs * N) * (Σ w[n])² / Σ(w[n]²)
      = (1/BW) * (Σ w[n])² / (N * Σ(w[n]²))
      = (1/BW) * 1/ENBW
```

つまり、**scipy は既に ENBW 補正を内部で行っている**ことが分かる。

---

## 3. DTT と scipy の比較

### 3.1 正規化方式の違い

| 項目 | DTT | scipy.signal.welch |
|------|-----|---------------------|
| **窓の正規化** | DCゲイン (`Σw/N`) の二乗で除算 | パワー総和 (`Σw²`) で除算 |
| **ENBW補正** | 明示的に `windowBW` を計算 | 暗黙的に `scale` に含まれる |
| **最終単位** | PSD [単位²/Hz] | PSD [単位²/Hz] |

### 3.2 数学的等価性の検証

DTT の実装を scipy に合わせて書き直すと：

```python
# DTT式の実装
windowNorm = (np.mean(window))**2  # (Σw/N)²
windowBW = BW / windowNorm         # ENBW調整後の帯域幅

# scipy式の実装
scale = 1.0 / (fs * (window**2).sum())

# 関係式の確認
# DTT: PSD_factor = 1 / windowBW = windowNorm / BW = (Σw/N)² / (fs/N)
#                 = N * (Σw)² / fs
# scipy: 1/scale = fs * Σw²

# これらが一致するか？
# DTT方式: factor ∝ (Σw)²
# scipy方式: factor ∝ Σw²
```

**結論:** DTT と scipy は**異なる正規化を使用している**。

DTT は **DCゲインの二乗** で正規化しており、scipy は **パワー総和** で正規化している。これらは一般には一致しない。

---

## 4. 実装上の課題と解決策

### 4.1 問題点

`gwexpy` が `scipy.signal.welch` をそのまま使用すると、DTT と異なる PSD 値が得られる可能性がある。

### 4.2 解決策オプション

**オプション A: DTT 方式を採用する (推奨しない)**

scipy を使わず、独自のFFT実装を書く。保守性が低下する。

**オプション B: scipy 結果を DTT 方式に変換する**

```python
def dtt_compatible_psd(timeseries, **welch_params):
    # scipy で計算
    f, Pxx = scipy.signal.welch(timeseries, **welch_params)
    
    # 窓関数の取得
    window = scipy.signal.get_window(welch_params['window'], welch_params['nperseg'])
    
    # DTT式への変換係数
    dc_gain = np.mean(window)
    power_sum = np.sum(window**2)
    conversion_factor = (dc_gain**2 * len(window)) / power_sum
    
    # 変換
    Pxx_dtt = Pxx * conversion_factor
    
    return f, Pxx_dtt
```

**オプション C: scipy の正規化が正しいことを検証し、DTT に合わせない**

実は、**scipy の正規化方式が業界標準**である。DTT が独自の正規化を使っている可能性があるため、むしろ DTT の結果を scipy に合わせて解釈すべき。

### 4.3 推奨アプローチ

1. **まず scipy の正規化が正しいことを確認する**（文献との照合）。
2. **DTT の出力データ（XML）を読み込み、同じ入力信号で scipy の結果と比較する**。
3. **差異があれば、変換係数を特定して `gwexpy` に適用する**。

---

## 5. 検証タスク

### Task 1: 窓関数の ENBW 値を計算・比較

```python
import numpy as np
from scipy.signal import get_window

def calculate_enbw(window_name, N=1024):
    w = get_window(window_name, N)
    dc_gain = np.sum(w) / N
    power_sum = np.sum(w**2) / N
    enbw = N * power_sum / (np.sum(w)**2)
    
    print(f"{window_name}:")
    print(f"  DC Gain: {dc_gain:.4f}")
    print(f"  ENBW Factor: {enbw:.4f}")
    print(f"  DTT windowNorm: {dc_gain**2:.4f}")
    return enbw

for win in ['hann', 'hamming', 'blackman', 'flattop']:
    calculate_enbw(win)
```

### Task 2: 既存の DTT データとの比較テスト

1. DTT で測定した PSD データ（XML）を用意する。
2. 同じ入力信号（可能であれば生データ）を scipy で処理する。
3. 両者のスペクトルをプロットして視覚的に比較する。
4. 必要に応じて変換係数を導出する。

---

---

## 6. DTT ソースコード分析の結論

### 6.1 実装の確認事項

**確認できたこと:**

1. **Window Normalization の計算**:
   ```c++
   double windowNorm = sMean(fftPlan.window_coeff, fftPoints);  // (1/N)Σw[n]
   windowNorm *= windowNorm;  // ((1/N)Σw[n])²
   windowBW = BW / windowNorm;  // fs/N / ((1/N)Σw[n])²
   ```

2. **FFT → PSD変換**:
   - `psGen()` 関数がFFT実行と窓関数適用を一括処理
   - `fftToPs()` 関数がFFT結果をPSDに変換
   - これらは GDS ライブラリ (libgds) の関数で、ソースコードは非公開

3. **psGen の引数**:
   ```c++
   psGen(psmode, &tmps[resultnum].prm, fftPoints, dtype, 
         (float*) chndat->value,  // 入力時系列
         1.0/(fSample / decimate2),  // サンプリング周期 dt
         OUTPUT_GDSFORMAT, window, tmps[resultnum].fft)
   ```
   - `dt = 1/fs` が明示的に渡されている
   - これはPSD正規化 `[単位²/Hz]` に必要な情報

### 6.2 推定される正規化方式

GDS ライブラリの `fftToPs` と `psGen` の実装は確認できないが、以下のように推定できる:

#### 仮説A: scipy と同じ正規化

```python
# scipy方式（標準的）
scale = 1.0 / (fs * (window**2).sum())
PSD = |FFT|² * scale
```

この場合、`windowBW` は **メタデータとしての記録のみ** で、実際のPSD計算には直接使用されない。

#### 仮説B: DTT独自の正規化

```python
# DTT方式（推定）
windowBW = BW / ((window.mean())**2)
scale = 1.0 / windowBW
PSD = |FFT|² * scale
```

この場合、scipy と異なる結果になる。

### 6.3 検証方法

**必須タスク:**

1. **実データ比較テスト**
   - DTT で既知の入力信号（例: 白色ノイズ）を測定
   - 同じ信号を scipy で処理
   - PSD の絶対値を比較（相対値ではなく）

2. **窓関数別の変換係数計算**
   ```python
   def calculate_dtt_scipy_ratio(window_name, N=1024):
       w = get_window(window_name, N)
       dc_gain = w.sum() / N
       power_sum = (w**2).sum() / N
       
       # DTT方式の係数（推定）
       dtt_factor = 1.0 / (dc_gain**2)
       
       # scipy方式の係数
       scipy_factor = 1.0 / power_sum
       
       ratio = dtt_factor / scipy_factor
       print(f"{window_name}: DTT/scipy = {ratio:.4f}")
       return ratio
   ```

3. **XML データの解析**
   - DTT 出力 XML ファイルから `windowBW` パラメータを読み取る
   - 理論値 `BW / (Σw/N)²` と一致するか確認

### 6.4 最終推奨

**Phase 1 の実装方針:**

1. **デフォルトは scipy を使用**
   - `scipy.signal.welch` の正規化は信頼できる業界標準
   
2. **DTT互換モードを提供**
   ```python
   def gwexpy_psd(timeseries, dtt_compatible=False, **params):
       f, Pxx = scipy.signal.welch(timeseries, **params)
       
       if dtt_compatible:
           # 実データ比較テストで求めた変換係数を適用
           conversion = calculate_dtt_conversion_factor(params['window'])
           Pxx *= conversion
       
       return f, Pxx
   ```

3. **変換係数の実測**
   - DTT の実測データがある場合のみ、変換係数をキャリブレーション

---

## 7. 次のステップ: 実装タスク

### Task 1: 検証ノートブックの作成

`notebooks/verification/dtt_psd_comparison.ipynb` を作成:

```python
# 1. 窓関数のENBW値を計算
# 2. scipy と DTT推定式の比較
# 3. （可能であれば）DTT XMLデータとの比較
```

### Task 2: gwexpy への統合

`gwexpy/signal/spectral.py` に以下を追加:

```python
def welch_dtt_compatible(timeseries, **kwargs):
    """
    DTT-compatible PSD calculation.
    
    This function computes PSD using scipy.signal.welch and optionally
    applies a conversion factor to match DTT's normalization.
    """
    pass
```

### Task 3: ドキュメント化

- DTT と scipy の正規化の違いを明記
- ユーザーがどちらを使うべきか判断できるガイドを作成



