# ScalarField Phase 3 Signal Processing Extensions - 実装完了レポート

**日時**: 2026-01-21T18:40  
**担当**: Claude Opus 4.5

---

## 成果物（変更ファイル）

### 1. 新規作成ファイル

| ファイル | 概要 |
|---------|------|
| `gwexpy/fields/demo.py` | デモデータ生成関数（make_demo_scalar_field 等） |
| `gwexpy/fields/signal.py` | 信号処理コア関数（PSD, 相互相関, コヒーレンス等） |

### 2. 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `gwexpy/fields/__init__.py` | 新規関数のエクスポート追加 |
| `gwexpy/fields/scalar.py` | ScalarField クラスにメソッドラッパー追加（compute_psd, freq_space_map など） |

---

## 追加・変更した Public API 一覧

### デモデータ生成 (`gwexpy.fields.demo`)

| 関数 | 説明 |
|------|------|
| `make_demo_scalar_field(pattern, ...)` | 決定論的デモScalarField生成（gaussian/sine/standing/noise） |
| `make_propagating_gaussian(**kw)` | 伝搬ガウシアンパルス（エイリアス） |
| `make_sinusoidal_wave(**kw)` | 正弦波（エイリアス） |
| `make_standing_wave(**kw)` | 定在波（エイリアス） |

### 信号処理関数 (`gwexpy.fields.signal`)

| 関数 | 説明 |
|------|------|
| `compute_psd(field, point_or_region, **kwargs)` | Welch法によるPSD計算 |
| `freq_space_map(field, axis, at, **kwargs)` | 周波数-空間マップ生成 |
| `compute_freq_space(...)` | freq_space_mapの別名 |
| `compute_xcorr(field, point_a, point_b, **kwargs)` | 2点間の相互相関 |
| `time_delay_map(field, ref_point, plane, at, **kwargs)` | 時間遅延マップ |
| `coherence_map(field, ref_point, *, plane, at, band, **kwargs)` | コヒーレンスマップ |

### ScalarField メソッドラッパー

| メソッド | 説明 |
|---------|------|
| `field.compute_psd(point_or_region, **kwargs)` | compute_psdのラッパー |
| `field.freq_space_map(axis, at, **kwargs)` | freq_space_mapのラッパー |
| `field.compute_xcorr(point_a, point_b, **kwargs)` | compute_xcorrのラッパー |
| `field.time_delay_map(ref_point, plane, at, **kwargs)` | time_delay_mapのラッパー |
| `field.coherence_map(ref_point, plane, at, **kwargs)` | coherence_mapのラッパー |

---

## 主要関数の docstring（抜粋）

### compute_psd

```python
def compute_psd(
    field: ScalarField,
    point_or_region: tuple | list | dict,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
    scaling: Literal["density", "spectrum"] = "density",
    average: Literal["mean", "median"] = "mean",
    return_onesided: bool = True,
) -> FrequencySeries | FrequencySeriesList:
    """Compute power spectral density using Welch's method.

    Returns
    -------
    FrequencySeries or FrequencySeriesList
        - 単一点: FrequencySeries（Unit: field.unit² / Hz）
        - 複数点: FrequencySeriesList

    Raises
    ------
    ValueError
        時間軸が不規則、または axis0_domain != 'time' の場合。
    """
```

### freq_space_map

```python
def freq_space_map(
    field: ScalarField,
    axis: str,
    at: dict | None = None,
    *,
    method: Literal["welch", "fft"] = "welch",
    ...
) -> ScalarField:
    """Compute frequency-space map along a spatial axis.

    Returns
    -------
    ScalarField
        Shape: (n_freq, n_x, 1, 1) など。axis0_domain='frequency'。
    """
```

### compute_xcorr

```python
def compute_xcorr(
    field: ScalarField,
    point_a: tuple,
    point_b: tuple,
    *,
    max_lag: int | Quantity | None = None,
    mode: Literal["full", "same", "valid"] = "full",
    normalize: bool = True,
    detrend: bool = True,
    window: str | None = None,
) -> TimeSeries:
    """Compute cross-correlation between two spatial points.

    Returns
    -------
    TimeSeries
        ラグ軸を持つ相互相関関数。正のラグは point_b が point_a に先行することを示す。
    """
```

### time_delay_map

```python
def time_delay_map(
    field: ScalarField,
    ref_point: tuple,
    plane: str = "xy",
    at: dict | None = None,
    *,
    max_lag: int | Quantity | None = None,
    stride: int = 1,
    roi: dict | None = None,
    normalize: bool = True,
    detrend: bool = True,
) -> ScalarField:
    """Compute time delay map from a reference point.

    Returns
    -------
    ScalarField
        遅延値を保持するスライス。Unit は時間単位（例: s）。
    """
```

### coherence_map

```python
def coherence_map(
    field: ScalarField,
    ref_point: tuple,
    *,  # 注意: plane, atはキーワード専用
    plane: str = "xy",
    at: dict | None = None,
    band: tuple | None = None,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    stride: int = 1,
) -> ScalarField | FieldDict:
    """Compute magnitude-squared coherence map from a reference point.

    Returns
    -------
    ScalarField or FieldDict
        - band指定時: スカラーコヒーレンス値の ScalarField
        - band=None: 周波数軸を持つ ScalarField
    """
```

---

## テスト担当へ渡す「テスト観点」メモ

### 1. デモデータ生成 (`demo.py`)

#### 機能テスト

- [ ] `make_demo_scalar_field` の全パターン（'gaussian', 'sine', 'standing', 'noise'）が正常に生成される
- [ ] seed を固定した場合、同じデータが再生成される（決定論性）
- [ ] 軸メタデータ（times, x, y, z）が正しい単位を持つ
- [ ] `axis0_domain='time'`、`space_domain='real'` が設定される

#### エッジケース

- [ ] nt=1, nx=1 など最小サイズでエラーにならない
- [ ] 非常に大きなサイズ（nt=1000, nx=64）でメモリエラーにならない

### 2. compute_psd

#### 正常系

- [ ] 単一点指定で FrequencySeries が返る
- [ ] 複数点指定で FrequencySeriesList が返る
- [ ] region（dict）指定で平均化された PSD が返る
- [ ] 周波数軸の単位が正しい（例: 1/s = Hz）
- [ ] PSD 単位が正しい（density: unit²/Hz, spectrum: unit²）

#### 合成信号検証

- [ ] 既知周波数の正弦波で、PSD ピークがその周波数に現れる
- [ ] ホワイトノイズで PSD が概ねフラット

#### エラー系

- [ ] `axis0_domain='frequency'` でエラー
- [ ] 不規則時間軸（irregular）でエラー
- [ ] 時間軸長 < 2 でエラー

### 3. freq_space_map

#### 正常系

- [ ] 返り値が ScalarField で axis0_domain='frequency'
- [ ] shape が (n_freq, n_x, 1, 1) など期待どおり
- [ ] at 指定なしでも length=1 軸は自動使用

#### エラー系

- [ ] 軸 length > 1 で at 未指定時にエラー

### 4. compute_xcorr

#### 正常系

- [ ] 返り値が TimeSeries
- [ ] normalize=True で値が [-1, 1] 範囲
- [ ] max_lag による切り詰めが動作する
- [ ] mode='full'/'same'/'valid' で出力長が変わる

#### 合成信号検証

- [ ] 既知のシフトを持つ信号で、ピークラグがシフト値と一致

### 5. time_delay_map

#### 正常系

- [ ] 返り値が ScalarField
- [ ] 遅延値の単位が時間単位（s など）
- [ ] stride によりサブサンプリングされた shape

#### 合成信号検証

- [ ] 伝搬波形で、遅延マップが距離に比例して変化

#### パフォーマンス

- [ ] stride, roi 指定で計算量が減少

### 6. coherence_map

#### 正常系

- [ ] band 指定時: スカラー ScalarField（shape に time 軸が 1）
- [ ] band=None 時: 周波数軸付き ScalarField（axis0_domain='frequency'）
- [ ] コヒーレンス値が [0, 1] 範囲

#### 合成信号検証

- [ ] 同一信号の自己コヒーレンスが 1.0
- [ ] 無相関なノイズ間のコヒーレンスが ≈ 0

#### エラー系

- [ ] band が周波数範囲外で ValueError

### 7. 共通

#### 単位整合性

- [ ] 入力 ScalarField の単位が出力に正しく反映される
- [ ] 軸座標の単位変換（m→km など）が正常

#### 4D shape 保持

- [ ] 返り値 ScalarField が常に 4 次元

---

## 今後の拡張ポイント（参考）

1. **描画ラッパー**: `plot_psd`, `plot_freq_space`, `plot_xcorr`, `plot_delay_map`, `plot_coherence_map`
2. **GPU/並列化**: `time_delay_map`, `coherence_map` の numpy 演算を vectorize
3. **非等間隔時間軸対応**: Lomb-Scargle 法への切り替え

---

*End of Report*
