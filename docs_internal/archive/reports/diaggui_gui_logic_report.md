# diaggui GUI機能・ロジック解析レポート

DTT (diaggui) のソースコード（特に `src/dtt/gui/diagctrl.cc`, `hh`）を解析し、GUIコンポーネントと解析・描画パラメータの紐付けを調査した結果を報告します。

## 1. GUIとパラメータの同期メカニズム
diagguiでは、`diag::DiagTabControl` クラスがGUIのタブと各種コントロール（入力フィールド、ボタン等）を管理しています。

*   **メッセージループ**: `ProcessMessage` 関数がボタンのクリックやテキスト更新のイベントをキャッチし、`ProcessButton`, `ProcessTextentry` などの専用関数へ振り分けます。
*   **パラメータへの読み書き**: `ReadMeasParam` 関数が、GUI上の数値（`fFFTStart`, `fFFTAverages` 等）を `MeasParam_t` という内部構造体に同期させます。逆に `WriteMeasParam` は内部データをGUIに表示します。

## 2. タブごとの主要ロジック
再現において重要な機能マッピングは以下の通りです。

### 2.1 Measurement (FFT) タブ
| GUI項目 | 内部パラメータ (`MeasParam_t`) | 備考 |
| :--- | :--- | :--- |
| Start Frequency | `fStart` | FFTの開始周波数。 |
| Stop Frequency | `fStop` | FFTの終了周波数。 |
| Res. Bandwidth | `fResolutionBW` | RBW (Resolution Bandwidth)。 |
| Averages | `fAverages` | 平均化回数。 |
| Avg. Type | `fAverageType` | 0: Linear, 1: Exponential, 2: Peak Hold。 |
| Window | `fWindow` | 0: Uniform, 1: Hanning, etc. |
| Overlap | `fOverlap` | 指定されたパーセントを 1/100 して内部保存。 |

> [!IMPORTANT]
> **用語の誤りに関する注意点**
> オリジナルの diaggui では、数学的には **Amplitude Spectral Density (ASD)** に相当するものが **Power Spectrum** と誤って表記されています。本移植プロジェクトでは、この誤記を踏襲せず、正しく **Amplitude Spectral Density** として実装・表示します。

### 2.2 Traces (グラフ設定) タブ
`diagplot.cc` 内のロジックにより、各トレース(`0-7`)に割り当てられたチャンネルとグラフタイプが管理されます。
*   **Graph Type**: 選択に応じて、`Amplitude Spectral Density`, `Coherence` などの計算モードが切り替わります。
*   **Active チェック**: `fMeasActive[i]` フラグにより、計算対象から外す制御が行われます。

### 2.3 Range / X-axis / Y-axis タブ
描画ライブラリ（ROOTベース）の座標軸設定に直結しています。
*   **Limit**: 手動設定(`rb_x_man`)時は `fStart`, `fStop` がビューポートの制限として機能します。
*   **Log/Linear**: スケール設定が描画エンジンの軸モードを切り替えます。

## 3. 実装上の重要な計算式
ソースコード内で見られた、GUI値から計算用パラメータへの変換式です。
*   **Settling Time**: `GUI値(%) / 100` として秒数に換算されます。
*   **Overlap**: `GUI値(%) / 100` で計算に使用。
*   **RBWとFFTポイント数**: `fResolutionBW` に基づいて FFTの `N` (ポイント数) が決定されます。

## 4. 移植に向けた結論
`dtt_gui.py` における `update_graphs` 関数は、単にランダムデータを生成するのではなく、上記の `MeasParam_t` に相当する Python 辞書やオブジェクトを参照し、`gwexpy` の計算メソッドに正しい引数（`fftlength`, `overlap`, `window` 等）を渡すよう設計する必要があります。
