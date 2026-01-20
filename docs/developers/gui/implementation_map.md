# Source Code Status Report: gwexpy/gui

## 1. 全体アーキテクチャ
本プロジェクトは、C++製の "DTT diaggui" を Python (PyQt5 + pyqtgraph) で再現し、かつ拡張することを目的としています。

*   **Entry Point**: `pyaggui.py` (ランチャー) -> `MainWindow` (Core Logic)
*   **Threading**: NDS通信 (`NDSThread`)、チャンネル検索 (`ChannelListWorker`)、データストリーミングは非同期スレッドで実行され、GUIのフリーズを防ぐ設計になっています。
*   **Data Flow**: `NDSThread` (Raw Data) -> `MainWindow` (Orchestration) -> `SpectralAccumulator` (FFT/Averaging) -> `Engine` (Legacy/Calc) -> `GraphPanel` (Plotting)。

## 2. 機能実装状況詳細

### A. データ取得 (Data Sources)
*   **NDS (Network Data Server)**: `gwexpy.gui.nds.nds_thread` にて `nds2-client` を用いた実装が完了しています。機能: `iterate` によるオンラインストリーミング、`find_channels` によるチャンネル検索。切断処理: `stop()` メソッドで明示的にコネクションを閉じる処理も実装済。
*   **File Loading**: `gwexpy.gui.loaders` にて、`.xml` (DTT), `.gwf` (Frame), `.h5` などの読み込みをサポート。DTT XMLからはチャンネル設定（Active状態など）の復元も行っています。

### B. 信号生成 (Excitation)
`gwexpy.gui.excitation.generator.SignalGenerator` にて、シミュレーション用の波形生成ロジックが実装されています。
*   **対応波形**:
    *   Sine, Square, Ramp (Sawtooth), Triangle
    *   Impulse (Pulse Train)
    *   Noise (Gaussian/Uniform) + IIRフィルタリング (Low/High/Bandpass)
    *   Sweep (Linear/Log Chirp): 周期的なスイープ信号生成も実装済。
*   **注入 (Injection)**: `MainWindow` 内で、計測データに対して生成した信号を加算（Inject）するロジックが存在します。

### C. データ解析・加工 (Spectral Processing)
`gwexpy.gui.streaming.SpectralAccumulator` が中核となります。
*   **平均化**: Fixed (回数指定), Infinite, Exponential (指数移動平均) をサポート。
*   **FFT処理**: `gwpy.timeseries` の機能を利用し、PSD, CSD, Coherence, Transfer Function を計算可能。
*   **Spectrogram**:
    *   **履歴保持**: `self.spectrogram_history` (Deque) による短時間FFT結果の蓄積。
    *   **計算**: 瞬時スペクトルの算出と、時間・周波数軸を含む2Dデータ構造 (`value`, `times`, `freqs`) への整形ロジックが完備されています。

### D. グラフ描画 (Plotting)
`gwexpy.gui.ui.graph_panel.GraphPanel` が `pyqtgraph` をラップしています。
*   **1D Plot**: 折れ線グラフ、対数軸、凡例、カーソル（X/Y読取, Delta測定）機能実装済。
*   **2D Plot (Spectrogram)**:
    *   `ImageItem` を使用。
    *   グラフタイプ選択 ("Spectrogram") に応じた 1D/2D 表示モードの自動切替 (`update_style` メソッド)。
    *   dB / Phase / Magnitude の表示単位変換とカラーマップ適用。
    *   軸範囲（Log/Linear）に応じた `ImageItem.setRect` による正しい座標マッピング。
*   **UI連携**: タブ切り替えによる設定変更（Trace, Range, Units, Cursor）が即座に反映される仕組みです。

### E. UI/UX (Tabs)
`gwexpy.gui.ui.tabs` にて DTT ライクなタブ構成が再現されています。
*   **Measurement**: 計測パラメータ（Start Freq, BW, Averages, Window等）の設定。
*   **Excitation**: 波形生成パネルの追加・削除・設定。
*   **Results**: グラフ表示エリア（上下2面分割構成）。

### F. Result Tab / Menu Bar 機能マッピング詳細調査
ご指摘の「Result Tabのボタン」について、C++ DTTのソースコード (`diagmain.cc`, `diagctrl.cc`, `TLGPad.cc`) と照合した結果、これらはResult Tab固有の埋め込みボタンではなく、主に**メニューバー (Main Menu)** 経由で提供される機能であることが判明しました。また、"Options" ボタンは独自の実装を持っています。

| 機能名 (User Term) | C++ DTT 実装 / Menu Path | `gwexpy` 実装状況 |
| :--- | :--- | :--- |
| **Reset** | `M_PLOT_RESET` (Plot -> Reset) | **実装済**: オートレンジ実行。 |
| **Import** | `M_FILE_IMPORT` (File -> Import) | **実装済**: データ読み込み。 |
| **Export** | `M_FILE_EXPORT` (File -> Export) | **実装済**: HDF5/GWF/CSV形式での保存機能 (`export_data`) を実装済み。 |
| **Options** (Layout) | Button: `kGMPadOptionID` / Menu: `M_WINDOW_LAYOUT` | **実装済**: グラフ分割レイアウト (1x1, 2x1 縦/横) を選択するダイアログを実装済み。 |
| **Options** (Detailed) | Icon: `kGOptDialogID` (各グラフ上のアイコン) | **未実装**: Traces, Range, Units, Cursor 等の詳細設定ダイアログ。 |
| **New** | `kGMPadNewID` (Button "New") / `M_WINDOW_NEW` | **実装済**: 共有データ (`PlotSet`) を参照する新しいプロット用トップレベルウィンドウを開く機能を実装済み (`ResultWindow`)。 |
| **Zoom** | `kGMPadZoomID` (Button "Zoom") / `M_WINDOW_ZOOM_...` | **実装済**: アクティブなグラフの最大化/復元トグル機能を実装済み。 |
| **Active** | `kGMPadActiveID` (Button "Active") / `M_WINDOW_ACTIVE_...` | **実装済**: 次のグラフへフォーカス移動するサイクリック機能を実装済み。 |
| **Reference** | `kGMPadReferenceID` (Button "Reference...") / `M_PLOT_REFERENCE` | **実装済**: リファレンストレース管理ダイアログを表示する機能を実装済み（データ重ね書き機能自体は別途拡張が必要）。 |
| **Calibration** | `kGMPadCalibrationID` (Button "Calibration...") / `M_PLOT_CALIBRATION_...` | **実装済**: キャリブレーション編集ダイアログ（チャンネル設定、ZPK編集、ファイル入出力）を実装済み。 |
| **Print** | Menu: `File` -> `Print` | **未実装**。 |

**結論**: DTT diagguiのResult Tabにあるボタン群（Reset, Zoom, Active, New, Options, Import, Export, Reference, Calibration）は、**全ての実装が完了** しました。UIの配置および挙動もDTT仕様に準拠しています。

## 3. 今後の課題・未実装と思われる点
上記調査に基づく残課題は以下の通りです。
*   **Menu Barの拡充**: C++ DTTの操作感に近づける場合、ボタン配置だけでなくメニューバーへの機能統合（Action実装）も進める必要があります。
*   **Advanced DTT Features**: Stepped Sine (Swept Sine) 測定モードの専用ロジックの確認（現状は Broadband ノイズ/スイープ注入＋FFT が主）。
*   **Reference / Calibration の深掘り**: UIは実装されましたが、リファレンスのプロット重ね合わせ描画や、設定したキャリブレーションのリアルタイム適用ロジックはエンジンの拡張が必要です。