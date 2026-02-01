# Algorithm Validation Report (LLM-based Cross-Check)

以下は gwexpy の指定領域に対する監査報告です。各所見は "Severity / Finding / Recommendation" の形式で記載します。数式や短いスニペットのみを含み、実装そのものの書き換えは行いません。

---

## 1. Physical Fields & k-space
- Severity: High  
  Finding: numpy.fft.fftfreq は周期数（cycles per unit）を返すため、物理的波数としては 2π をかけてラジアン表現に変換しなければならない。FFT の shift・正規化規約と合わせないと微分演算やスペクトル振幅が誤る。  
  Recommendation: k を明示的に生成し、shift と FFT スケーリング規約をプロジェクト全体で統一する。例:
  k = 2π * fftfreq(N, d=dx)
  k_shifted = fftshift(k)

- Severity: Medium  
  Finding: sampling interval（時間/長さ）の単位が明示されていない。  
  Recommendation: Field オブジェクトが sampling interval を保持し、fft_space がそれを参照して k を生成するようにする。

---

## 2. Transient Response Analysis
- Severity: Medium  
  Finding: 自動ステップ検出がノイズや閾値感度に脆弱で、開始/終了のズレがトランジェント抽出を歪める。  
  Recommendation: 平滑化→高階差分→ロバスト閾値、あるいは CUSUM 等の変化点検出を検討し、検出感度解析を追加する。

- Severity: High  
  Finding: _fft_transient で窓関数を使う際に窓ゲイン補正やエネルギー保存（Parseval）を明示しないと振幅が変化する。  
  Recommendation: 窓ゲインで正規化する（例：窓の平均/二乗和で補正）か、入力-復元でエネルギー保存を検証するテストを追加する。

---

## 3. Robust Statistics（VIF & block bootstrap）
- Severity: High  
  Finding: 時系列自己相関に対して回帰用 VIF を誤用している箇所がある。時系列の「実効サンプル数」は Bartlett 等に基づく式を用いるべき。  
  Recommendation: 自己相関による分散膨張は次のような式で扱う（例）:
  Neff ≈ N / (1 + 2 ∑_{k=1}^{M} ρ_k)
  回帰 VIF（1/(1-R^2)）と混同しないこと。

- Severity: Medium  
  Finding: spectrogram に対する block bootstrap のブロック長が固定だと時間-周波数相関を遮断する可能性あり。  
  Recommendation: データ駆動なブロック長選択（Politis & White 等）、stationary/circular bootstrap の検討とカバレッジ検証を行う。

---

## 4. Bayesian Fitting（GLS）
- Severity: High  
  Finding: GLS 尤度は
  log L = -0.5 * (r^T Σ^{-1} r + ln det Σ + N ln 2π)
  であり、Σ の逆や行列式を直接計算すると数値不安定。Σ がパラメータ依存なら det 項を無視すると尤度が誤る。  
  Recommendation: Cholesky を使う（solve を用い、ln det Σ = 2 * sum(log(diag(L)))）。Σ が半正定なら対角に小さな jitter を加えて安定化する。

---

## 5. Time Series Modeling（ARIMA → GPS-aware TimeSeries）
- Severity: Medium  
  Finding: ARIMA は等間隔データを前提とするため、GPS タイムスタンプが不規則だと不整合が生じる。さらに leap second 等の時刻系扱いが未指定だと長期オフセットが蓄積する恐れ。  
  Recommendation: ARIMA を適用する前に等間隔化（リサンプリング）または状態空間（Kalman）アプローチを検討する。時刻系は GPS epoch 秒等で統一し、メタデータに変換手順を記録する。

- Severity: Low  
  Finding: ARIMA の予測分散が TimeSeries に正しく伝搬されていない可能性あり。  
  Recommendation: 予測分散／予測区間を TimeSeries メタデータとして保持する。

---

## 6. Dimensionality Reduction（PCA/ICA の flatten/reconstruct）
- Severity: Medium  
  Finding: 3D (time × channels × freq) の flatten の軸順が不明確だと再構成で軸不一致が起こる。周波数で複素データを扱う場合の方針（実/虚分離 vs 複素 PCA）も明示する必要がある。  
  Recommendation: flatten の軸順を API とドキュメントで固定し、reconstruct に必要な情報（mean, components, axis order）を保存する。複素データは明示的に扱うか、実/虚の分離方針を採用する。PCA/ICA 前後で総分散が保存されるテストを追加する。

---

## 一般的数値安定性注意点
- 行列の逆は直接計算せず、solve/Cholesky を使う。  
- 正定性が怪しい行列には jitter（例: 1e-8 × trace）を追加して安定化し、感度解析を行う。  
- FFT のスケーリングをプロジェクト全体で統一し、基本的な循環テスト（sin FFT→IFFT 復元）を自動化テストに入れる。

---

## 参考文献（簡潔）
- Bartlett (1946) — 有効サンプルサイズ（自己相関）  
- Politis & Romano / Politis & White — ブロックブートストラップ  
- Matrix Cookbook / Numerical Recipes — 行列演算と数値安定化

---

必要であれば、該当モジュール（ファイル名・関数名）を指定してください。該当箇所に対する具体的な式補正や短いスニペットを提示します。


# Algorithm Validation Report (LLM-based Cross-Check)

以下は gwexpy の指定領域に対する監査報告です。各所見は "Severity / Finding / Recommendation" の形式で記載します。数式や短いスニペットのみを含み、実装そのものの書き換えは行いません。

---

## 1. Physical Fields & k-space
- Severity: High  
  Finding: numpy.fft.fftfreq は周期数（cycles per unit）を返すため、物理的波数としては 2π をかけてラジアン表現に変換しなければならない。FFT の shift・正規化規約と合わせないと微分演算やスペクトル振幅が誤る。  
  Recommendation: k を明示的に生成し、shift と FFT スケーリング規約をプロジェクト全体で統一する。例:
  k = 2π * fftfreq(N, d=dx)
  k_shifted = fftshift(k)

- Severity: Medium  
  Finding: sampling interval（時間/長さ）の単位が明示されていない。  
  Recommendation: Field オブジェクトが sampling interval を保持し、fft_space がそれを参照して k を生成するようにする。

---

## 2. Transient Response Analysis
- Severity: Medium  
  Finding: 自動ステップ検出がノイズや閾値感度に脆弱で、開始/終了のズレがトランジェント抽出を歪める。  
  Recommendation: 平滑化→高階差分→ロバスト閾値、あるいは CUSUM 等の変化点検出を検討し、検出感度解析を追加する。

- Severity: High  
  Finding: _fft_transient で窓関数を使う際に窓ゲイン補正やエネルギー保存（Parseval）を明示しないと振幅が変化する。  
  Recommendation: 窓ゲインで正規化する（例：窓の平均/二乗和で補正）か、入力-復元でエネルギー保存を検証するテストを追加する。

---

## 3. Robust Statistics（VIF & block bootstrap）
- Severity: High  
  Finding: 時系列自己相関に対して回帰用 VIF を誤用している箇所がある。時系列の「実効サンプル数」は Bartlett 等に基づく式を用いるべき。  
  Recommendation: 自己相関による分散膨張は次のような式で扱う（例）:
  Neff ≈ N / (1 + 2 ∑_{k=1}^{M} ρ_k)
  回帰 VIF（1/(1-R^2)）と混同しないこと。

- Severity: Medium  
  Finding: spectrogram に対する block bootstrap のブロック長が固定だと時間-周波数相関を遮断する可能性あり。  
  Recommendation: データ駆動なブロック長選択（Politis & White 等）、stationary/circular bootstrap の検討とカバレッジ検証を行う。

---

## 4. Bayesian Fitting（GLS）
- Severity: High  
  Finding: GLS 尤度は
  log L = -0.5 * (r^T Σ^{-1} r + ln det Σ + N ln 2π)
  であり、Σ の逆や行列式を直接計算すると数値不安定。Σ がパラメータ依存なら det 項を無視すると尤度が誤る。  
  Recommendation: Cholesky を使う（solve を用い、ln det Σ = 2 * sum(log(diag(L)))）。Σ が半正定なら対角に小さな jitter を加えて安定化する。

---

## 5. Time Series Modeling（ARIMA → GPS-aware TimeSeries）
- Severity: Medium  
  Finding: ARIMA は等間隔データを前提とするため、GPS タイムスタンプが不規則だと不整合が生じる。さらに leap second 等の時刻系扱いが未指定だと長期オフセットが蓄積する恐れ。  
  Recommendation: ARIMA を適用する前に等間隔化（リサンプリング）または状態空間（Kalman）アプローチを検討する。時刻系は GPS epoch 秒等で統一し、メタデータに変換手順を記録する。

- Severity: Low  
  Finding: ARIMA の予測分散が TimeSeries に正しく伝搬されていない可能性あり。  
  Recommendation: 予測分散／予測区間を TimeSeries メタデータとして保持する。

---

## 6. Dimensionality Reduction（PCA/ICA の flatten/reconstruct）
- Severity: Medium  
  Finding: 3D (time × channels × freq) の flatten の軸順が不明確だと再構成で軸不一致が起こる。周波数で複素データを扱う場合の方針（実/虚分離 vs 複素 PCA）も明示する必要がある。  
  Recommendation: flatten の軸順を API とドキュメントで固定し、reconstruct に必要な情報（mean, components, axis order）を保存する。複素データは明示的に扱うか、実/虚の分離方針を採用する。PCA/ICA 前後で総分散が保存されるテストを追加する。

---

## 一般的数値安定性注意点
- 行列の逆は直接計算せず、solve/Cholesky を使う。  
- 正定性が怪しい行列には jitter（例: 1e-8 × trace）を追加して安定化し、感度解析を行う。  
- FFT のスケーリングをプロジェクト全体で統一し、基本的な循環テスト（sin FFT→IFFT 復元）を自動化テストに入れる。

---

## 参考文献（簡潔）
- Bartlett (1946) — 有効サンプルサイズ（自己相関）  
- Politis & Romano / Politis & White — ブロックブートストラップ  
- Matrix Cookbook / Numerical Recipes — 行列演算と数値安定化

---

## 追加スキャン結果
- Severity: Low  
  Finding: リポジトリをスキャンした結果、上記で既に対象とした主要な独自アルゴリズム群（k-space/ScalarField, ResponseFunctionAnalysis, bootstrap_spectrogram, run_mcmc, ARIMA→TimeSeries マッピング, TimeSeriesMatrix の flatten/reconstruct）以外に、アルゴリズム上重要な独自実装は見つかりませんでした。  
  Recommendation: 新たなモジュールや重大なアルゴリズム追加が行われた場合は再レビューを実施してください。特に物理単位・FFT スケーリング・共分散行列の取り扱いについては変更があれば早めに検証を行うことを推奨します。

---

必要であれば、該当モジュール（ファイル名・関数名）を指定してください。該当箇所に対する具体的な式補正や短いスニペットを提示します。