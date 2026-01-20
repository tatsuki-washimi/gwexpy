# DTT (diaggui) リファレンス解析レポート

このレポートは、`gwexpy/gui/reference-dtt/dtt-master/src/dtt/diag` にあるオリジナルの C++ 版 DTT (Diagnostic Test Tools) 実装を解析したものです。Python による再実装版 (`gwexpy/gui`) の開発および検証のためのリファレンス情報を提供することを目的としています。

## 1. コアアーキテクチャ (C++)

測定の中核ロジックは `src/dtt/diag` に存在します。クラス階層を用いて異なる測定タイプを処理しています。

### クラス階層
*   **`diag::diagtest`**: 抽象基底クラス。
*   **`diag::stdtest`**: 標準的なテスト実装クラス。共通パラメータ（開始/終了周波数、トリガーなど）を管理します。
    *   **`diag::ffttest` (`ffttools.cc`)**: **Broadband** 解析（FFT, PSD, CSD）を実装。"FFT" モードおよび "Coherence" モードで使用されます。
    *   **`diag::sweptsine` (`sweptsine.cc`)**: **Stepped Sine** 解析を実装。"SweptSine" モードで使用されます。
    *   **`diag::sineresponse` (`sineresponse.cc`)**: 固定周波数のサイン応答測定。
*   **`diagnames.h`**: すべての GUI パラメータの定義文字列を定数として保持し、UI と Engine 間の疎結合を保証しています。

## 2. 主要な測定モード

### A. FFT / Broadband (`ffttools.cc`)
このモードは、現在の `gwexpy` における `SpectralAccumulator` の実装に相当します。

*   **ロジック**:
    1.  **設定**: `calcTimes` を通じて典型的なパラメータ（`BW` (帯域幅), `Window` (窓関数), `Overlap`, `Averages` (平均化回数)）を読み込みます。
    2.  **Zoom FFT**: "Zoom" モード (`fZoom != 0`) を明示的に扱います。ズーム中心周波数が設定された場合、システムはデータを **ヘテロダイン**（周波数変換）してから FFT 処理を行うことで、狭帯域での高分解能を実現します。
    3.  **Burst Mode**: `BurstNoiseQuietTime` をサポートします。バースト（断続的）な測定において、システムが静定する時間を確保するための "Quiet" (静寂) 区間と "Excitation" (加振) 区間を計算します。
*   **パラメータマッピング**:
    *   `fftWindow`: 0-6 (Uniform, Hanning, FlatTop など)
    *   `fftOverlap`: 0.0 - 1.0 (比率)
    *   `fftAverageType`: Linear (線形平均) vs Exponential (指数平均)。

### B. Swept Sine (`sweptsine.cc`)
**重要な相違点**: C++ 版の "SweptSine" は **Stepped Sine** (ステップ正弦波) 測定であり、連続的なチャープ (Chirp) ではありません。

*   **ロジック**:
    *   離散的な周波数ポイントの集合 (`fPoints`) を定義します。
    *   **シーケンス**: 各ポイントに対して以下の処理を行います：
        1.  加振周波数と振幅を設定。
        2.  **待機**: `SettlingTime` (過渡応答が収まるまで待機)。
        3.  **測定**: `MeasurementTime` (数サイクル分を積分)。
        4.  その特定周波数のフーリエ係数を計算 (単一ビン DFT)。
        5.  次のポイントへ移動。
    *   **スイープタイプ**: Linear (線形), Logarithmic (対数), User Defined (ユーザー定義リスト)。

## 3. パラメータリファレンス (`diagnames.h`)

このファイルは DTT 設定の「ロゼッタストーン」です。XML/RPC で使用される主要な識別子は以下の通りです：

| カテゴリ | DTT パラメータ名 | 説明 |
| :--- | :--- | :--- |
| **共通** | `Bandwidth` (BW) | 周波数分解能。 `dt = 1/BW`。 |
| | `SettlingTime` | 測定開始前に待機する時間 (秒)。 |
| | `Window` | 窓関数のインデックス。 |
| **加振** | `StimulusAmplitude` | 出力電圧/カウント。 |
| | `RampUp` / `RampDown` | 過渡応答を防ぐためのソフトスタート/ストップ時間。 |
| **SweptSine** | `MeasurementTime` | **各ポイントごと** の積分時間。 |
| | `HarmonicOrder` | 高調波（2f, 3f 応答など）を測定する機能。 |
| **グラフ** | `UnitsX...`, `UnitsY...` | データと共に保存されるプロット単位の設定。 |

## 4. Result Tab 実装分析 (`diagmain.cc` / `TLGPad.cc`)

ユーザー機能について、実装の詳細（メニュー vs ボタン）を調査しました。

### 機能マッピング (C++ メニューコマンド vs Python 実装)

| User Feature | C++ Implementation Source | 説明と Python 実装状況 |
| :--- | :--- | :--- |
| **Reset** | `M_PLOT_RESET` | **実装済**: オートレンジ実行。 |
| **Import** | `M_FILE_IMPORT` | **実装済**: データ読み込み。 |
| **Export** | `M_FILE_EXPORT` | **実装済**: `gwpy.write` を用いた HDF5/GWF/CSV 保存機能を実装しました。 |
| **Options** (Layout) | `TLGMultiPad::OptionDlg` (`kGMPadOptionID`) | **実装済**: Result Tab の "Options" ボタンは **レイアウト編集** (Grid Layout) ダイアログを開き、1x1, 2x1 (縦/横) の切り替えをサポートします。 |
| **Options** (Detail) | `TLGPad::OptionDlg` (`kGOptDialogID`) | **未実装**: 詳細設定は各プロット上のアイコンボタンとして別途検討が必要。 |
| **New** | `TLGMultiPad::NewWindow` (`kGMPadNewID`) | **実装済**: Result Tab の "New" ボタンは、`ResultWindow` クラスを用いて、共有データ (`PlotSet`) を参照する新しいウィンドウを開く機能を実装しました。 |
| **Zoom** | `TLGMultiPad::Zoom` (`kGMPadZoomID`) | **実装済**: Result Tab の "Zoom" ボタンは、アクティブなグラフの **最大化表示** ↔ **レイアウト表示** トグル機能として実装しました。 |
| **Active** | `TLGMultiPad::SetActivePad` (`kGMPadActiveID`) | **実装済**: Result Tab の "Active" ボタンは、フォーカスを次のグラフへ移動させる機能として実装しました。 |
| **Reference** | `TLGMultiPad::ReferenceTracesDlg` (`kGMPadReferenceID`) | **実装済**: "Reference..." ボタンでリファレンストレース管理ダイアログが開く機能を実装しました。（実際のプロット描画ロジックは別途拡充が必要） |
| **Calibration** | `TLGMultiPad::CalibrationEditDlg` (`kGMPadCalibrationID`) | **実装済**: "Calibration..." ボタンでキャリブレーションテーブル編集ダイアログ（チャンネル設定、ZPK編集、ファイル入出力）が開く機能を実装しました。 |

## 5. `gwexpy` 開発への提言

## 5. `gwexpy` 開発への提言（最新状況反映）

1.  **"Options" の実装完了**: Result Tab の "Options" ボタンは「レイアウト設定 (1x1, 2x1等)」として実装を完了しました。プロット詳細設定（対数軸、トレース選択）は別途アイコンまたはメニューでの対応が推奨されます。
2.  **"Reference" / "Calibration" のUI実装完了**:
    *   **Reference**: ダイアログUIは完成しました。今後はリファレンスデータを実際にグラフ上に重ねて描画するエンジンの拡張が必要です。
    *   **Calibration**: 編集・入出力UIは完成しました。今後はこの設定値 (`gain`, `pole/zero`) を測定データにリアルタイムで適用する信号処理パイプラインの統合が必要です。
3.  **Export の実装完了**: HDF5/GWF/CSV への保存機能が実装され、データのポータビリティが確保されました。
4.  **Stepped Sine の検証**: (前述の通り)

---
*解析元: `gwexpy/gui/reference-dtt/dtt-master/src/dtt/diag` & `src/dtt/gui/dttgui`*
