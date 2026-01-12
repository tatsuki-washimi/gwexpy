# ASD API Reference

`gwexpy.noise.asd` モジュールは Amplitude Spectral Density (ASD) を生成する関数を提供します。
すべての関数は `FrequencySeries` オブジェクトを返します。

---

## from_pygwinc

pyGWINC 検出器ノイズモデルから ASD を取得します。

### シグネチャ

```python
from_pygwinc(
    model: str,
    frequencies: np.ndarray | None = None,
    quantity: Literal["strain", "darm", "displacement"] = "strain",
    **kwargs
) -> FrequencySeries
```

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|------------|------|
| `model` | str | 必須 | pyGWINC モデル名 (例: "aLIGO", "Aplus", "Voyager") |
| `frequencies` | array | None | 周波数配列 [Hz]。None の場合は fmin/fmax/df から生成 |
| `quantity` | str | "strain" | 物理量 (下記参照) |
| `fmin` | float | 10.0 | 最小周波数 [Hz] |
| `fmax` | float | 4000.0 | 最大周波数 [Hz] |
| `df` | float | 1.0 | 周波数ステップ [Hz] |

### 許可される quantity

| quantity | 単位 | 説明 |
|----------|------|------|
| `"strain"` | 1/√Hz | 歪み ASD |
| `"darm"` | m/√Hz | Differential Arm Length ASD |
| `"displacement"` | m/√Hz | `"darm"` の非推奨エイリアス |

### 例外

- `ValueError`: quantity が許可値以外
- `ValueError`: `quantity="darm"` で arm length 取得不可
- `ValueError`: `fmin >= fmax`
- `ImportError`: pygwinc 未インストール

### 変換規則

`darm = strain × L` (L は IFO の arm length `ifo.Infrastructure.Length`)

### 使用例

```python
from gwexpy.noise.asd import from_pygwinc

# Strain ASD
strain_asd = from_pygwinc("aLIGO", quantity="strain")
# unit: 1 / sqrt(Hz)

# DARM ASD
darm_asd = from_pygwinc("aLIGO", quantity="darm")
# unit: m / sqrt(Hz)
```

---

## from_obspy

ObsPy の地震・超低周波音ノイズモデルから ASD を取得します。

### シグネチャ

```python
from_obspy(
    model: str,
    frequencies: np.ndarray | None = None,
    quantity: Literal["displacement", "velocity", "acceleration"] = "acceleration",
    **kwargs
) -> FrequencySeries
```

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|------------|------|
| `model` | str | 必須 | モデル名: "NHNM", "NLNM", "IDCH", "IDCL" |
| `frequencies` | array | None | 周波数配列 [Hz]。None の場合はモデルの元の周波数を使用 |
| `quantity` | str | "acceleration" | 物理量 (下記参照) |

### 許可される quantity (地震モデルのみ)

| quantity | 単位 | 説明 |
|----------|------|------|
| `"acceleration"` | m/(s²·√Hz) | 加速度 ASD |
| `"velocity"` | m/(s·√Hz) | 速度 ASD |
| `"displacement"` | m/√Hz | 変位 ASD |

> **⚠️ 重要**: `quantity="strain"` は**サポートされていません**。地震ノイズモデルには歪みの概念がありません。歪み ASD には `from_pygwinc` を使用してください。

### 例外

- `ValueError`: `quantity="strain"` (非サポート)
- `ValueError`: quantity が許可値以外
- `ValueError`: model が未知
- `ImportError`: obspy 未インストール

### 変換規則 (加速度から)

- `velocity = acceleration / (2πf)`
- `displacement = acceleration / (2πf)²`
- **f=0 では NaN** (無限大ではない)

### 使用例

```python
from gwexpy.noise.asd import from_obspy

# 加速度 ASD (デフォルト)
acc_asd = from_obspy("NLNM")
# unit: m / (s² · sqrt(Hz))

# 変位 ASD
disp_asd = from_obspy("NLNM", quantity="displacement")
# unit: m / sqrt(Hz)

# strain は ValueError
# from_obspy("NLNM", quantity="strain")  # NG!
```

---

## quantity 対応表 (まとめ)

### pyGWINC (`from_pygwinc`)

| quantity | 単位 | 備考 |
|----------|------|------|
| strain | 1/√Hz | デフォルト |
| darm | m/√Hz | = strain × L |
| displacement | m/√Hz | darm のエイリアス (非推奨) |
| velocity | - | **ValueError** |
| acceleration | - | **ValueError** |

### ObsPy (`from_obspy`)

| quantity | 単位 | 備考 |
|----------|------|------|
| acceleration | m/(s²·√Hz) | デフォルト |
| velocity | m/(s·√Hz) | = acc / (2πf) |
| displacement | m/√Hz | = acc / (2πf)² |
| strain | - | **ValueError** |
