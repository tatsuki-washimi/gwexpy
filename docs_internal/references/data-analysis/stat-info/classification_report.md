# 統計情報ドキュメントの分類レポート

`docs_internal/references/data-analysis/stat-info/` 内のドキュメントを内容に基づいて分類した結果です。

## カテゴリ一覧

### 1. 統計的手法 (Statistical Methods: Correlation & Information Theory)
相関分析、順位統計学、相互情報量、最大情報係数（MIC）などの基礎理論。

- **相関指標 (Correlation Measures)**
  - **A weighted Kendall's tau statistic** (`1-s2.0-S0167715298000066-main.pdf`, `S0167715298000066.html`)
  - **The Kendall Rank Correlation Coefficient** (`Abdi-KendallCorrelation2007-pretty.pdf`)
- **最大情報係数 (MIC: Maximal Information Coefficient)**
  - **Theoretical Foundations of Equitability and the MIC** (`1408.4908v3.pdf`)
  - **An Empirical Study of the MIC and Leading Measures of Dependence** (`AOAS-2018.pdf`)
  - **Differentially Private Maximal Information Coefficients** (`lazarsfeld22a.pdf`)
  - **Lecture Notes: Measures of Correlation and Dependence** (`CDT_04_Corrs__b_MIC_FoDS.pdf`)
- **相互情報量の推定・モデル選択 (MI Estimation & Selection)**
  - **fastMI: a fast and consistent copula-based nonparametric estimator of MI** (`2212.10268v4.pdf`)
  - **Mutual information model selection algorithm for time series** (`CJAS_47_1707516.pdf`)
  - **Multiscale part mutual information for direct associations** (`btab182.pdf`)

### 2. GW/KAGRA 研究 (GW/KAGRA Research: Detector Noise & Search)
重力波検出器（KAGRA等）のノイズ分析、監視システム（CAGMon）、および探索アルゴリズム。

- **KAGRA 監視・解析ツール (CAGMon & Monitoring)**
  - **Identifying and diagnosing coherent associations (CAGMon)** (`CAGMonGW-KAGRAv15.pdf`)
  - **Comments for the CAGmon paper** (`washimi_CAGMONpaper.pdf`)
  - **Glitch Catalog for empirical study on CAGMon parameters** (`Glitch Catalog for empirical study on CAGMon parameters.pdf`)
- **自己回帰モデル・探索パイプライン (AR Search & Pipelines)**
  - **Autoregressive Search of Gravitational Waves: I. Denoising** (`ARIMA_DeNoise.pdf`)
  - **Autoregressive Search for Unmodeled transients (BEACON)** (`BEACON.pdf`)
  - **Autoregressive Search of Unmodeled GW (Sparkler)** (`Sparkler_1min.pdf`)
- **検出器・バースト探索 (Noise Reduction & Burst)**
  - **Usefulness of an ARMA model for noise reduction in GW detectors** (`f2f_1220.pdf`)
  - **Toward Model-Agnostic Frameworks for Detecting Burst Transients** (`KAGRAF2F36&KIW13.pdf`)

### 3. 機械学習・応用分野 (ML & Other Applications)
深層学習、継続学習、およびバイオ・医学画像への応用。

- **深層学習・継続学習 (Deep Learning & Continual Learning)**
  - **Online Continual Learning through Mutual Information Maximization (OCM)** (`guo22g.pdf`)
  - **A robust estimator of mutual information for deep learning interpretability** (`Piras_2023_Mach._Learn.__Sci._Technol._4_025006.pdf`)
  - **MIST: Mutual Information Estimation via Supervised Training** (`2511.18945v1.pdf`)
  - **A Benchmark Suite for Evaluating Neural MI Estimators** (`2410.10924v1.pdf`)
- **特定分野への応用 (Bioinformatics & Medical)**
  - **Comparison of co-expression measures: MI vs Correlation** (Gene networks) (`1471-2105-13-328.pdf`)
  - **Self-similarity weighted mutual information: Image registration** (`1-s2.0-S1361841513001746-main.pdf`)
