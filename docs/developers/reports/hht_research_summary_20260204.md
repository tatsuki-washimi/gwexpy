# Hilbert-Huang Transform (HHT) 文献調査レポート

提供された `docs/developers/references/data-analysis/HHT/` 以下の資料に基づき、HHT 関連の研究内容と `gwexpy` への実装の関連性をまとめました。

## 1. 主要な研究テーマと知見

提供された 33 個の PDF 資料から、以下の主要な研究テーマが特定されました。

### A. 重力波バースト・超新星（CCSNe）の解析

- **主な著者**: Takeda (武田), Sakai (酒井), Hu, Yokozawa (横澤)
- **内容**:
  - コア崩壊型超新星（CCSNe）からの重力波信号に対し、HHT が高い時間・周波数分解能を持つことを利用した解析。
  - Dimmelmeier らの超新星波形モデルを用いたシミュレーション。
  - 信号の「進化トラック（evolutionary track）」を Hilbert spectrum 上で捉え、物理パラメータ（状態方程式 EoS、質量、回転など）を推定する。

### B. 重力波トリガー解析

- **主な著者**: Sakai (酒井)
- **内容**:
  - 振幅ベースの重力波バースト検出アルゴリズム（HHT-based trigger generator）。
  - 非定常・非線形なノイズ環境下でのトリガー性能の評価。

### C. パラメータ推定（BNS 合体）

- **主な著者**: Kaneyama (兼山), Sakai (酒井)
- **内容**:
  - 連星中性子星（BNS）合体時の潮汐変形や合体後の信号から、中性子星の半径や状態方程式（EOS）を推定する手法。

### D. アルゴリズムの改善

- **内容**:
  - **EEMD (Ensemble EMD)**: モードミキシング（mode-mixing）問題を解決するための手法。
  - **Multiresolution HHT (MHHT)**: 異なる時間・周波数スケールでの解析を統合し、より鮮明なスペクトルを得る手法（「stacking」等の表現で言及あり）。

---

## 2. 現状の `gwexpy` 実装との対応

現在のコードベースにおける HHT の実装状況は以下の通りです。

- **`gwexpy.types.hht_spectrogram.HHTSpectrogram`**:
  - HHT 特有のプロット（対数軸、IA/IA^2 重み付け）をサポートするクラス。
- **`gwexpy.timeseries._spectral_special.TimeSeriesSpectralSpecialMixin.hht()`**:
  - `PyEMD` を用いた EMD/EEMD 分解をサポート。
  - 各 IMF に対して `hilbert_analysis` を行い、瞬時振幅（IA）と瞬時周波数（IF）を算出。
  - 最終的にビン詰め（binning）を行い `HHTSpectrogram` を生成する。

---

## 3. 今後の発展・実装の可能性

資料に記載されているが、現時点で `gwexpy` に明示的な API が見当たらない、または強化できる要素：

1. **Multiresolution HHT (MHHT)**:
    - 複数の HHT マップを重ね合わせ（stacking）てパターンを強調する手法の実装。
2. **HHT-based Trigger Generator**:
    - 資料 `HHT_trigger_Sakai.pdf` にあるような、特定の閾値判定に基づいたトリガーリスト（EventTable）生成機能。
3. **SN Waveform Analysis Utilities**:
    - Dimmelmeier らの波形モデルや特定の進化トラックを抽出するためのフィッティング・フィルタリング機能。
4. **Edge Effect Mitigation**:
    - 端点効果（edge effects）をより高度に抑制するための窓関数の適用や補外手法の拡充。

以上の内容により、提供された資料の全体像を把握しました。
これらの知見は、今後の `gwexpy` における HHT 関連機能の高度化において非常に重要なリファレンスとなります。
