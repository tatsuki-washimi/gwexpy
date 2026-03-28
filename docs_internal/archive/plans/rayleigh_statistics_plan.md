# Rayleigh統計・非ガウス性解析ツールキット 実装計画

山村隼聖氏の修士論文（2024年）および山本尚弘氏の研究（2015-2016年）に基づく、
重力波データの非ガウス雑音検出・評価のための包括的ツールキットを GWexpy に実装する計画です。

**参考文献**: `docs_internal/references/data-analysis/non-Gaussian/` 配下の論文群

---

## 実装項目一覧

| # | 機能 | 優先度 | 難易度 | 参照 |
|---|------|--------|--------|------|
| 1 | `rayleigh_spectrogram` ラッパー | 高 | 低 | GWpy既存機能 |
| 2 | GauCh（修正KS検定） | 高 | 中〜高 | 論文 5.2節 |
| 3 | Rayleigh statistic の検定化 | 高 | 中 | 論文 5.1節 |
| 4 | Student-t 非ガウス性指標 | 中 | 中 | 山本氏 2015-2016 |
| 5 | 自動 DataQualityFlag 生成 | 中 | 低〜中 | 論文 7章応用 |
| 6 | 非ガウス雑音シミュレーター | 中 | 低〜中 | 論文 6.1節 |
| 7 | ROC 曲線による検出性能評価 | 低〜中 | 中 | 論文 6.2-6.3節 |
| 8 | ラインノイズ自動除外マスク | 低〜中 | 低 | 論文 5.2.1節 |
| 9 | 複合ダッシュボードプロット | 低 | 中 | 論文 5.2.2節 図5.2.6 |
| 10 | チュートリアルノートブック＆ドキュメント | 高 | 中 | — |

---

## 1. `rayleigh_spectrogram` の GWexpy ラッパー実装

**目的**: 継承元の `gwpy` メソッドを直接呼び出す現状を改善し、`gwexpy.spectrogram.Spectrogram` を返すようにする。

### 現状
- `TimeSeries.rayleigh_spectrogram()` は `gwpy` から継承されて利用可能
- 戻り値が `gwpy.spectrogram.Spectrogram` であり、GWexpy 独自機能（PlotMixin, InteropMixin等）が使えない

### 変更内容
#### [MODIFY] `gwexpy/timeseries/_spectral_fourier.py`
- `TimeSeriesSpectralFourierMixin` に `rayleigh_spectrogram` メソッドを追加
- 内部で `gwpy` の実装を呼び出し、結果を `gwexpy.spectrogram.Spectrogram` に変換して返す
- 他のスペクトログラムメソッド（`spectrogram`, `cwt` 等）と同じパターンに合わせる

---

## 2. GauCh（修正 Kolmogorov-Smirnov 検定）

**目的**: ASD分布がレイリー分布に従うかどうかを修正KS検定で評価し、非ガウス雑音を自動判定する。

### アルゴリズム概要（論文 5.2.1節より）
1. 長さ $T$ のデータを FFT時間 $T_{FFT}$ で分割し、$n$ 個の ASD を計算
2. ASD値 $\{ASD_1, \ldots, ASD_n\}$ がレイリー分布に従うと仮定し、パラメータ $\hat{\sigma}^2$ を最尤推定:
   $$\hat{\sigma}^2 = \frac{1}{2n} \sum_{i=1}^{n} ASD_i^2$$
3. 経験分布 $F_n(x)$ と理論的レイリー分布 $F(x; \hat{\sigma}^2)$ のKS統計量 $D_n$ を計算:
   $$D_n = \sup_x |F_n(x) - F(x; \hat{\sigma}^2)|$$
4. **修正**: パラメータをデータから推定しているため通常のKS検定公式(4.2.3)は使えない → モンテカルロ法でバックグラウンド分布を推定し、片側検定で $p$ 値を計算

### 変更内容
#### [NEW] `gwexpy/statistics/gauch.py`
- `GauChResult` クラス: p値マップ、統計量マップ、メタデータを保持
- `compute_gauch()` 関数: コアアルゴリズム
- バックグラウンド分布テーブル: 代表的なサンプルサイズ $n$ に対する事前計算済み $D_n$ 分布を内蔵（補間で任意の $n$ に対応）
- ラインノイズマスク対応: 指定された周波数帯域を除外可能

#### [MODIFY] `gwexpy/timeseries/_statistics.py` または新規 `_gaussianity.py`
- `ts.gauch(T, fftlength, ...)` メソッドを追加
- 戻り値: `Spectrogram` 形式の p値マップ（時間×周波数）

### 設計上の決定事項
- **出力形式**: p値を直接返す。可視化メソッドで `-log10(p)` 変換を選択可能にする
- **バックグラウンド分布**: ハードコードされたテーブル＋補間を基本とし、オプションでon-the-flyモンテカルロも可能にする

---

## 3. Rayleigh statistic の検定化（論文 5.1節）

**目的**: 既存の `rayleigh_spectrogram` の出力（指標値 $R$）を、統計的仮説検定に落とし込む。

### アルゴリズム概要
1. 白色ガウス雑音から Rayleigh statistic $R$ のバックグラウンド分布をモンテカルロ法で事前推定
2. 実データの $R$ とバックグラウンド分布を比較し、**両側検定**で $p$ 値を計算（$R$ は1付近がガウス的、離れると非ガウス的）
3. $p < \alpha$ で非ガウス雑音と判定

### 変更内容
#### [NEW] `gwexpy/statistics/rayleigh_test.py`
- バックグラウンド分布テーブル（サンプルサイズ $n$ 依存）
- `rayleigh_pvalue()` 関数

#### [MODIFY] TimeSeries メソッド
- `ts.rayleigh_test(stride, fftlength, alpha=0.05)` を追加
- 戻り値: p値の `Spectrogram` オブジェクト

---

## 4. Student-t 分布を用いた非ガウス性指標

**目的**: 山本氏の研究に基づき、周波数成分の分布を Student-t 分布でフィットし、自由度 $\nu$ を非ガウス性指標として使用する。

### 概要
- ガウス分布は $\nu \to \infty$ の Student-t 分布に対応
- $\nu$ が小さいほど裾が重い（ヘヴィーテール）= 非ガウス性が強い
- Rayleigh/GauCh とは独立な切り口の指標

### 変更内容
#### [NEW] `gwexpy/statistics/student_t_indicator.py`
- 周波数成分（FFTの実部/虚部）に対する Student-t 分布フィット
- 自由度 $\nu$ の推定（最尤推定）

#### [MODIFY] TimeSeries メソッド
- `ts.student_t_spectrogram(stride, fftlength)` を追加
- 戻り値: 自由度 $\nu$ の `Spectrogram` オブジェクト

---

## 5. 自動 DataQualityFlag (セグメント) 生成

**目的**: GauCh / Rayleigh 検定 / Student-t 指標の結果から、非ガウス雑音が存在する時間帯を自動抽出し、Veto セグメントとして出力する。

### 変更内容
#### [NEW] `gwexpy/statistics/dq_flag.py`
- `to_segments(p_value_map, alpha=0.05, min_duration=1.0)` 関数
- p値マップ（Spectrogram形式）を入力し、全周波数で $p < \alpha$ となる時間帯を `SegmentList` として出力
- 周波数バンドごとの判定も可能（例: 10-100 Hz のみで判定）
- GWpy の `DataQualityFlag` オブジェクトとして直接出力可能

---

## 6. 非ガウス雑音シミュレーター（論文 6.1節）

**目的**: 論文で使用された非ガウス雑音モデルを簡単に生成するユーティリティ。

### Model I: 一時的ガウス雑音の重ね合わせ（非定常雑音）
$$x(t) = n_0(t) + A_1 \cdot B(t) \cdot n_1(t)$$
- $n_0, n_1$: KAGRAデザイン感度PSDを持つガウス雑音
- $B(t)$: Tukey窓（$\alpha=0.5$）、ランダムな開始時刻、継続時間 $T/6$
- $A_1$: 非ガウス性の強度パラメータ

### Model II: 散乱光雑音（定常雑音）
$$x(t) = n_0(t) + G \cdot \sin\left(\frac{4\pi}{\lambda}(x_0 + \delta x_{sc}(t))\right)$$
- $\delta x_{sc} = A_2(1 + 0.25\sin(2\pi f_{amp} t)) \cdot \cos(2\pi f_{sc} t)$
- パラメータ: $G$, $\lambda$, $f_{sc}$, $A_2$ 等

### 変更内容
#### [NEW] `gwexpy/noise/` モジュール
- `transient_gaussian_noise(duration, sample_rate, A1, psd=None)` → TimeSeries
- `scatter_light_noise(duration, sample_rate, A2, f_sc=0.2, G=3e-22)` → TimeSeries
- `inject_noise(clean_ts, noise_ts)` → TimeSeries (重ね合わせユーティリティ)

---

## 7. ROC 曲線による検出性能評価ツール（論文 6.2-6.3節）

**目的**: 非ガウス雑音モデルと検出手法のパラメータを指定し、ROC曲線・AUCを自動計算。

### 変更内容
#### [NEW] `gwexpy/statistics/roc.py`
- `evaluate_roc(method, model, params, n_trials=1000, freq_band=None)` 関数
- ROC曲線プロットメソッド
- AUC算出
- サンプルサイズ $n$ vs TPR の関係も出力可能（論文 図6.3.1の再現）

---

## 8. ラインノイズ自動除外マスク

**目的**: GauCh 出力から既知のラインノイズを除外するためのマスク生成。

### 変更内容
#### [NEW] `gwexpy/statistics/line_mask.py`（または `gwexpy/noise/line_mask.py`）
- KAGRAの既知ラインリスト（電源60Hz系列、calibration line等）を内蔵データとして保持
- `create_line_mask(frequencies, detector='KAGRA', custom_lines=None)` → boolean配列
- `apply_line_mask(spectrogram, mask)` → マスク適用済み Spectrogram

---

## 9. 複合ダッシュボードプロット（論文 5.2.2節 図5.2.6）

**目的**: GauCh の運用で使用されるマルチパネルプロットを一発で出力。

### パネル構成
1. GauCh p値マップ（赤=非ガウス / 水色=ガウス / 白=除外）
2. Rayleigh statistic スペクトログラム
3. Whitened time series
4. Inspiral range（オプション）

### 変更内容
#### [NEW] `gwexpy/plot/gauch_dashboard.py`
- `plot_gauch_dashboard(ts, T, fftlength, line_mask=None)` 関数
- matplotlib の `GridSpec` を使用したマルチパネルレイアウト
- 各パネルのカラーマップ・カラーバーを論文と同様のスタイルに統一

---

## 10. チュートリアルノートブック＆ドキュメント

### ノートブック構成案
#### [NEW] `docs_internal/notebooks/rayleigh_gauch_tutorial.ipynb`

1. **理論紹介**: ガウス雑音と非ガウス雑音、Rayleigh statistic の定義
2. **Rayleigh Spectrogram**: `ts.rayleigh_spectrogram()` の基本的な使い方
3. **GauCh**: `ts.gauch()` による自動判定とプロット
4. **シミュレーション**: Model I / Model II の雑音を生成し、検出性能を比較
5. **ROC 評価**: パラメータ最適化のデモ
6. **実用**: DataQualityFlag の生成とVetoセグメントの活用

### Sphinx ドキュメント
- API リファレンスに全新規クラス・関数のドキュメントを追加
- User Guide に「非ガウス雑音解析」セクションを新設

---

## 実装順序（推奨）

```
Phase 1 (基盤):
  1. rayleigh_spectrogram ラッパー
  6. 非ガウス雑音シミュレーター（テスト用データ生成に必要）

Phase 2 (コア機能):
  2. GauCh（修正KS検定）
  3. Rayleigh statistic 検定化
  4. Student-t 指標

Phase 3 (応用機能):
  5. 自動 DataQualityFlag 生成
  8. ラインノイズ自動除外マスク

Phase 4 (評価・可視化):
  7. ROC 曲線評価ツール
  9. 複合ダッシュボードプロット

Phase 5 (ドキュメント):
  10. チュートリアル＆ドキュメント
```

---

## Open Questions

> [!IMPORTANT]
> **バックグラウンド分布の管理方法**:
> GauCh と Rayleigh 検定の両方で、サンプルサイズ $n$ に依存するバックグラウンド分布テーブルが必要です。
> - 方針A: 代表的な $n$ (例: 20, 39, 79, 159, 239) の分布をパッケージ内にハードコード → 高速、オフライン利用可
> - 方針B: 初回実行時にモンテカルロを回してキャッシュ → 柔軟だが初回が遅い
> - **推奨**: 方針A をベースに、カスタム $n$ が指定された場合のみ方針B にフォールバック

> [!IMPORTANT]
> **p値の出力形式**:
> - p値そのまま (0〜1) を返し、可視化時にオプションで `-log10(p)` を選択可能にする方針でよいか？
> - GauCh の binary 出力（赤/水色の2値プロット）も別途メソッドとして用意するか？

> [!IMPORTANT]
> **モジュール配置**:
> 新規モジュールの配置先について：
> - `gwexpy/statistics/` を新設して gauch, rayleigh_test, student_t, roc, dq_flag をまとめるか？
> - `gwexpy/noise/` を新設してシミュレーター・ラインマスクをまとめるか？
> - 既存の `gwexpy/timeseries/_statistics.py` に追加するか？

---

## Verification Plan

### Automated Tests
- `rayleigh_spectrogram` が `gwexpy.spectrogram.Spectrogram` を返すことの確認
- 白色ガウス雑音入力時の GauCh の FPR が有意水準 $\alpha$ に近いことの統計的検証
- Rayleigh 検定の p値分布が一様分布に従うことの $\chi^2$ 検定（論文 5.1.2節の再現）
- Model I / Model II シミュレーターの出力が物理的に妥当であることの検証

### Manual Verification
- チュートリアルノートブックの全セル実行
- 論文の図（5.2.4, 6.2.1〜6.3.1 等）の定性的再現
