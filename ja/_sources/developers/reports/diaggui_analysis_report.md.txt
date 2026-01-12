# DTT (diaggui) ソースコード解析レポート

DTT (Diagnostic Test Tools) の `src/dtt` ディレクトリ内のソースコード（主に `ffttools.cc/hh`）を解析した結果を報告します。

## 1. 全体構造
DTTのコアロジックは C++ で実装されており、以下の主要コンポーネントで構成されています。

*   **`diag::ffttest` クラス**: `FFT` 測定のメインクラス。パラメータの読み込み、測定のタイミング制御、FFT計算の実行、結果の保存を担当。
*   **`diag::stdtest`**: すべての診断テスト（SineResponse, SweptSineなど）の基底クラス。
*   **`diagResult` / `gdsDataObject`**: 測定データや結果を保持するための共通データコンテナ。

## 2. スペクトル計算ロジック (`ffttools.cc`)
`ffttest::fft` および `ffttest::cross` メソッドにて、以下の手順で計算が行われています。

### 2.1 信号処理フロー
1.  **DCオフセット除去 (`removeDC`)**: オプションにより、データから直流成分を除去。
2.  **ウィンドウ関数適用**: `Hann` ウィンドウなどが適用される（`psGen` 関数内）。
3.  **FFTの実行**: 指定された `fftPoints` に基づき高速フーリエ変換を実行。
4.  **ズームFFT (`fZoom > 0`)**: 複素ヘテロダイン変換を用いた高解像度ズーム機能をサポート。
5.  **PSD/CSD計算**: FFT結果からパワースペクトル密度（PSD）やクロススペクトル密度（CSD）を算出。

### 2.2 アベレージング（平均化）
`avg_specs` 構造体を用いて、時間方向の平均化が制御されています。
*   **Linear (線形平均)**: 全測定期間の平均を等しく計算。
*   **Exponential (指数平均)**: 新しいデータに高い重みを置くリアルタイム更新向けモード。

## 3. 主要パラメータ (`diagnames.h`)
GUIで設定可能なパラメータが定数として定義されています。

| パラメータ名 | 説明 |
| :--- | :--- |
| `fftStartFrequency` / `fftStopFrequency` | 測定周波数範囲 |
| `fftBW` | 解像度帯域幅 (RBW) |
| `fftOverlap` | FFTセグメントのオーバーラップ率 (デフォルト 0.5) |
| `fftWindow` | ウィンドウ関数の種類 (1: Hann, etc.) |
| `fftAverages` | 平均化回数 |

## 4. 測定モードの詳細
*   **FFTモード**: 定常的な信号のスペクトル解析。
*   **Timeseriesモード**: 生データの時間波形表示。
*   **SineResponse / Swept Sine**: 励起信号（Excitation）に対するシステムの応答計測。

## 5. Python GUI（gwexpy）への適用に向けた知見
*   **計算の置き換え**: C++ の `psGen` や `fftToPs` のロジックは、Python では `gwpy.timeseries.TimeSeries.asd()` や `csd()` で直接代替可能です。
*   **状態管理**: `ffttest::analyze` に相当するループを Python の `QTimer` 内で実装し、アベレージングの状態を保持する `DttEngine` が必要です。
*   **パラメータ連携**: `diagnames.h` で定義されている各種フラグと GUI の `Range` / `Config` タブの項目を正確に紐付けることで、DTT本来の挙動を再現できます。
