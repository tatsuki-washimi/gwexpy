# DTT サブシステム詳細分析レポート

**作成日:** 2026-01-22
**対象:** `gwexpy/gui/reference-dtt/dtt-master/src/dtt/`
**Author:** Antigravity Agent

---

# Section 1: GUI ロジック詳細分析

## 1.1 概要

本セクションでは、DTT (Diagnostic Test Tools) のGUI実装、特にメインウィンドウの構造、イベント処理、データバインディング、および描画ロジックに関する詳細なコード分析の結果をまとめる。

## 1.2 ウィンドウ構造とレイアウト

アプリケーションのメインウィンドウは `DiagMainWindow` クラス（ROOTの `TGMainFrame` を継承）によって定義されている。

* **メニューバー (`TGMenuBar`)**:
  * File, Edit, Measurement, Plot, Window, Help などの標準的な構成。
* **メインコントロールエリア**:
  * `DiagTabControl` を使用して機能をタブで分割。
  * **設定タブ**: Input, Measurement, Excitation などのパラメータ設定用。
  * **表示タブ**: 結果表示用の `TLGMultiPad` を含む。
* **ボタンバー**:
  * テスト実行制御用のボタン群（Start, Pause, Resume, Abort）。
* **ステータスバー (`TGStatusBar`)**:
  * プログラムの状態、ハートビート、進捗状況を表示。

## 1.3 ウィジェットクラス階層

ROOTフレームワークをベースに、LIGO特有の要件を満たすカスタムクラス（`TLG` プレフィックス）が拡張されている。

* **コンテナ**:
  * `TGMainFrame` -> `DiagMainWindow`
  * `TGCompositeFrame` -> `TLGPad`, `TLGMultiPad`, `DiagTabControl`
* **カスタムコントロール** (`src/dtt/gui/dttgui/`):
  * **`TLGTextEntry` / `TLGNumericEntry`**: バリデーション、単位入力、増減ボタンなどを備えた入力フィールド。
  * **`TLGChannelBox`**: LIGOチャンネル選択用の階層型コンボボックス（Site -> IFO -> System）。
* **描画ウィジェット**:
  * **`TLGPad`**: `TRootEmbeddedCanvas` をラップし、グラフ (`TGraph`, `TH1`)、軸、凡例、オプションパネルの描画を管理する単一の描画領域。
  * **`TLGMultiPad`**: 複数の `TLGPad` をグリッド状（例：2x2）に配置・管理するコンテナ。

## 1.4 イベント処理パターン

ROOTのメッセージマップとディスパッチ関数を利用した標準的なイベント駆動モデルを採用している。

* **メッセージ処理**:
  * `ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)` が中央処理ハブとして機能。
  * `kC_COMMAND` (ボタン、メニュー), `kC_STATUS`, `kC_NOTIFY` などのメッセージタイプに応じて、`ProcessButton`, `ProcessMenu` などのハンドラに振り分けられる。
* **タイマー処理**:
  * **Heartbeat (`fHeartbeat`)**: 100msごとのタイマー。バックエンドからの通知キュー (`fNotifyMsgs`) をチェックし、UIの更新（テスト完了通知など）をトリガーする。
  * **Watchdog (`fXExitTimer`)**: X11ディスプレイ接続を監視。
* **バックエンド通信**:
  * GUIは直接計算を行わず、`basic_commandline` インターフェースを通じてコマンド（"run", "save"など）を送信し、状態を取得するクライアント・サーバー（または分離カーネル）モデルに近い構造。

## 1.5 データバインディングとフロー

データフローは、ポーリングとコマンド応答モデルに基づいている。

* **設定**:
  * `TestParam_t` 構造体が全てのテスト設定を保持。
  * `TransferParameters` メソッドにより、GUIウィジェットと構造体の間で値を同期。
* **データ発見プロセス**:
  * `AddDataFromIndex` が `basic_commandline` にインデックスを問い合わせ、利用可能なデータ（時系列、PSD、伝達関数など）を発見する。
  * テキストベースの階層構造（例: `Result.IndexEntry[1] = ...`）をパースして解析。
* **データ記述子**:
  * **`DiagDataDescriptor`**: 実データへのアクセサ。描画が必要になった時点で `cmd->getData` を呼び出し、遅延ロードを行う。
  * **`PlotDescriptor`**: データ (`BasicDataDescriptor`) とメタデータ、キャリブレーション情報を紐付ける。

## 1.6 描画ロジック

描画システムは `PlotSet` コンテナによってメインウィンドウから分離されている。

1. **リポジトリ (`PlotSet`)**:
    * 利用可能な全プロットデータを管理する中央レジストリ。グラフタイプやチャンネル名で整理される。
2. **更新サイクル**:
    * ハートビートタイマーが「新しいテスト結果」を検知すると、`UpdatePlot` が発火。
    * 新規データをスキャンし、`PlotSet` に追加。
3. **レンダリング**:
    * `fPlot->Update()` が実行され、登録された `TLGPad` を反復処理。
    * データ更新がある場合、`TLGPad` は `DiagDataDescriptor` 経由で配列データを取得し、ROOTオブジェクト (`TGraph`) を更新。
    * ユーザーの表示設定（Mag, Phase, dBなど）に基づき、複素数データの変換を行ってから描画する。

## 1.7 補助機能とダイアログ

### メニューシステム (`mainmenu.cc`, `TLGMainMenu`)

* **構成**: `TLGMainMenu` クラスがメニューバーの構築とコールバックの初期処理を担当。
* **機能**:
  * ファイル操作 (`New`, `Open`, `Save`, `Import`, `Export`, `Print`)
  * ウィンドウ操作 (`Zoom`, `Layout`, `Active`)
  * 外部ツール起動（`Launch` メニュー）: `mainmenu.cc` は `launch_client` クラスを使用して、`dataviewer`, `foton` などの外部プロセスを起動するランチャー機能も兼ねている。

### エクスポートと保存 (`TLGExport`, `TLGSave`)

* **形式**: 主に LIGO独自の **XSIL (XML)** 形式と汎用テキスト形式をサポート。
* **ロジック**:
  * `TLGExport::ExportToFileXML`: `PlotSet` 内のデータを走査し、XMLタグ構造 (`<XSIL>`) を生成。
  * **データ変換 (`DoConversion`)**: 生データ（複素数など）をユーザー指定の形式（振幅、dB、位相、実部/虚部）に変換してからファイルに書き出す。

---

# Section 2: Kernel Logic Analysis

## 2.1 Overview

The `src/dtt/diag/` directory contains the core "diagnostic kernel" of the DTT application. This kernel is responsible for executing measurement tests (such as FFT, Swept Sine, Time Series), managing excitation signals, synchronizing with data acquisition (NDS/DAQS), and performing real-time signal processing and analysis.

## 2.2 Architecture & Core Components

### Class Hierarchy

* **`diagtest` (Abstract Base Class)**: Defines the universal interface for any diagnostic test. It encapsulates the test environment (`diagStorage`, `dataBroker`, `excitationManager`) and defines lifecycle methods (`begin`, `setup`, `end`) and the main analysis trigger.
* **`stdtest` (Standard Test)**: Inherits from `diagtest`. Implements the common logic found in most tests:
  * **Channel Management**: Manages lists of Stimulus (`stimulus`) and Measurement (`measurementchannel`) channels.
  * **Measurement Scheduling**: Defines `addMeasurements` and `newMeasPoint` to generate a sequence of `interval`s and `syncpointer`s.
  * **Synchronization**: Implements `syncAction`, which is called when data for a specific interval is available, triggering the `analyze` method.
* **Concrete Signal Processing Tests**: `ffttest`, `sweptsine`, `timeseries`, `sineresponse`.

### Execution Engine: `standardsupervisory`

* **Modes**: Supports Real-Time (`runRT`) and Off-Line (`runOL`) execution.
* **Loop**: It runs a main loop that drives the test.
  * Calls `test->setup()` to initialize the measurement schedule.
  * Enters a wait loop (`syncWait` or `syncRead`).
  * Upon data availability, it triggers the callback mechanism in `stdtest` which performs the analysis.
  * Handles asynchronous events like Pause, Resume, and Abort.

### Synchronization: `testsync`

* **`syncpointer`**: Represents a future point in time or a data condition that the test waits for.
* **`interval`**: Defines a specific time window `[t0, t0 + duration]` for which data is required.

## 2.3 Test Implementations & Signal Processing

### FFT Test (`ffttest`)

* **Setup**: Calculates parameters (`calcTimes`) like bandwidth, window type, overlap, and averaging.
* **Execution**:
  * **`calcMeasurements`**: Sets up excitation (random noise, periodic) and measurement intervals.
  * **`analyze`**: Allocates temporary storage (`tmpresult`) and iterates through channels.
    * **`fft`**: Computes Power Spectral Density (PSD).
    * **`cross`**: Computes Cross-Spectral Density (CSD) and Coherence.

### Swept Sine Test (`sweptsine`)

* **Setup**: Generates a list of sweep points (`sweeppoints`). Supports Linear, Logarithmic, and User-defined sweeps.
* **Signal Processing**:
  * **`sinedet` (Sine Detection)**: Extracts the specific frequency component from the time series data.
  * **`transfn`**: Computes the Transfer Function `H = Out/In` and Coherence.

### Triggered Time Series (`timeseries`)

* **Setup**: Configures pre-trigger and post-trigger durations.
* **Processing**: Supports simple averaging or summation of time traces across multiple triggers.

## 2.4 Data Flow Summary

1. **Configuration**: GUI parameters (`TestParam`) are read by `readParam` in the specific test class.
2. **Scheduling**: `standardsupervisory` calls `test->setup()`.
3. **Acquisition**: `dataBroker` (external) fetches data corresponding to these intervals.
4. **Wait**: `standardsupervisory` blocks on `syncWait`.
5. **Callback**: When data arrives, `stdtest::syncAction` is invoked.
6. **Analysis**: `syncAction` calls the virtual `analyze` method.
7. **Processing**: `analyze` calls helper methods (`fft`, `sinedet`).
8. **Publication**: Results are stored in `diagStorage`.

---

# Section 3: Data Acquisition Logic Analysis

## 3.1 Overview

The "Data Acquisition" logic in DTT is decoupled into two distinct layers:

1. **Channel Metadata Layer (`src/dtt/daq/`)**: Handles channel name resolution, attribute lookup (sample rate, units, calibration), and site-specific prefixes.
2. **Data Transport Layer (`src/dtt/storage/`)**: Defines the abstract `dataBroker` interface for connecting to NDS/DAQS servers, requesting time-series data, and managing data streams.

## 3.2 Channel Metadata Layer (`src/dtt/daq/`)

### Core C API: `gdschannel`

* **File**: `gdschannel.h`, `gdschannel.c`
* **Struct `gdsChnInfo_t`**: The core data structure defining a channel.
  * **Identity**: `chName` (max 60 chars), `chNum`, `dcuId`, `ifoId`.
  * **Properties**: `dataRate` (Hz), `dataType` (int16, float, complex, etc.), `unit`, `chGroup` (Fast/Slow).
  * **Calibration**: `gain`, `slope`, `offset`.
* **Key Functions**:
  * `gdsChannelInfo(name, info)`: Fills the struct for a given channel name.
  * `gdsChannelList(ifo, query, ...)`: Returns a list of channels matching criteria.

### C++ Abstraction: `testchn`

* **Class `channelHandler`**: A utility class to manage site and interferometer prefixes.
  * **Prefix Management**: Stores Default/Mandatory Site (H, L) and IFO (0, 1, 2) identifiers.
  * **Name Expansion**: `channelName()` expands short names (e.g., "ASC-X_TR") to full names (e.g., "H1:ASC-X_TR").

## 3.3 Data Transport Interface (`src/dtt/storage/`)

### Abstract Data Broker: `dataBroker`

* **Key Responsibilities**:
  * **Connection**: `connect()`, `reconnect()`.
  * **Subscription**: `add(channel)` to build a request list.
  * **Request**: `set(start, duration)` (Offline/Archive) or `set(start, active)` (Real-time).
  * **Flow Control**: `clear()`, `reset()`, `stop()`.
* **Error Handling**: Defines specific exceptions for `NoDataError` and `DataOnTapeError`.

## 3.4 Porting Implications for `gwexpy`

* **Legacy C API**: The `gdschannel` C struct and API should be replaced by NDS2-Client Python bindings.
* **Structure Preservation**: The separation of "Name Resolution" from "Data Fetching" is a good pattern.
* **Name Expansion**: The logic for handling "H1:", "L1:" prefixes is essential for UX.

---

# Section 4: Foton (Filter Online Tool) Logic Analysis

## 4.1 Overview

Foton (Filter Online Tool) is the primary logic for designing, visualizing, and exporting digital filters (IIR) for the LIGO real-time systems.

## 4.2 Architecture

### Directory Structure

* **`src/dtt/foton/`**: Contains the `main` entry point (`foton.cc`) and the application wrapper.
* **`src/dtt/filterwiz/`**: Contains the core business logic and GUI implementation.

### Key Classes

* **`TLGFilterWizard` (Logic + GUI)**: The main window controller.
* **`FilterFile` (Data Model)**: Represents the entire filter configuration file.
* **`FilterModule`**: Represents a named collection of filters.
* **`FilterSection`**: Represents a single filter stage (e.g., "Boost" or "Notch").
* **`IIRSos` (Signal Processing)**: Represents a single Biquad (Second Order Section).

## 4.3 Core Logic Flow

### Loading a Filter File

1. **Entry**: `foton.cc` parses arguments.
2. **`TLGFilterWizard::Setup`**: Initializes the window and triggers file loading.
3. **`FilterFile::read`**: Parses the text file line-by-line.

### Designing a Filter

1. **User Input**: The user enters a design string (e.g., `zpk(...)`, `reso(...)`).
2. **Parsing**: The string is parsed into poles and zeros.
3. **Computation**: The z-domain coefficients are computed.
4. **Validation**: Foton checks if the filter is stable.

### Saving

1. **`FilterFile::write`**: Iterates over all `FilterModule`s and writes the `# MODULES` header and SOS coefficients.

## 4.4 Porting Implications for `gwexpy`

* **Parsing Logic**: The parsing of the `.txt` filter file format must be ported to Python.
* **Filter Design**: Python's `scipy.signal` (`zpk2sos`, `bilinear`) provides 90% of the math needed.
* **Stability Checks**: `gwexpy` should implement checks similar to `check_poles`.

---

# Section 5: AWG GUI Logic Analysis

## 5.1 概要

本セクションでは、LIGO診断ツール群における波形生成（Excitation/AWG）に関するGUIの実装を解析した結果をまとめる。解析対象には、独立した波形生成ツールである **AWG GUI** (`awggui.cc`) と、診断テストの一部として波形生成を行う **DTT GUI Excitation Tab** (`diagctrl.cc`) の双方が含まれる。

## 5.2 AWG GUI (`awggui.cc`) の詳細解析

AWG GUIは、リアルタイムで任意の波形信号を制御・出力するためのスタンドアロンアプリケーション（または独立ウィンドウ）である。

### クラス・データ構造

* **メインクラス:** `AwgMainFrame` (`TGMainFrame` を継承)
* **データ構造:** `struct awg` (グローバル変数 `awgCmd` としてインスタンス化)
* **設定保存:** `struct configuration`

### バックエンド通信ロジック

AWG GUIは、`awgapi.h` で定義されたAPI関数を使用してバックエンド（AWGサーバー、`awgtpman` 等）と直接通信を行う。

1. **チャンネル確保 (`ReadChannel`)**:
    * `awgSetChannel(const char* channelName)`: スロット番号を取得。
    * `awgRemoveChannel(int slotNum)`: チャンネルの割り当てを解除。

2. **コマンド送信 (`HandleButtons` - "Set/Run")**:
    * **コマンド例:** `set <slotNum> sine <freq> <amp> <offset> <phase>`
    * `awgcmdline(const char* command)`: コマンドを送信・実行。

### 主要な機能

* **波形タイプ:** Sine, Square, Ramp, Triangle, Offset (DC), Uniform (Noise), Normal (Gaussian Noise), Arbitrary, Sweep。
* **フィルタ:** `foton` ツールを動的にロードしてフィルタ係数を生成・適用可能。
* **制御:** ゲイン調整、ランプ時間設定、停止。

## 5.3 DTT GUI Excitation Tab (`diagctrl.cc`) の詳細解析

DTT GUIのExcitationタブは、伝達関数測定などの診断テストを実行する際に、被測定系に与える刺激（Stimulus）信号を設定するためのインターフェースである。

### バックエンド通信ロジック

DTTでは、「テスト実行時の一括設定」というモデルを採用している。

1. **パラメータ設定 (`DiagMainWindow::WriteParam`)**:
    * `fCmdLine->putVar(...)` メソッドを使用してバックエンドの変数を更新。

2. **実行**:
    * 全てのパラメータがセットされた後、`fCmdLine->parse("run")` が実行される。

## 5.4 比較とまとめ

| 特徴 | AWG GUI (`awggui.cc`) | DTT GUI Excitation (`diagctrl.cc`) |
| :--- | :--- | :--- |
| **主な目的** | 任意のタイミング・波形での手動信号出力 | 診断テスト（FFT/SweptSine等）のための刺激信号設定 |
| **通信タイミング** | ユーザー操作時に即時送信 | テスト開始時に一括送信 |
| **通信API** | `awgcmdline()` (テキストコマンド直接送信) | `fCmdLine->putVar()` (変数設定) -> `run` |
| **チャンネル管理** | `awgSetChannel` で動的にスロット確保 | `basic_commandline` の設定変数として管理 |
| **柔軟性** | 高い（Addによる重ね合わせ、即時停止等） | テストシーケンスに従属 |

**結論:**
AWG GUIは**手続き型・即時実行型**の制御を行っており、DTT GUIは**宣言型・バッチ実行型**のアプローチをとっている。

---

*このレポートにより、DTT の各サブシステムの実装詳細が把握できました。*
