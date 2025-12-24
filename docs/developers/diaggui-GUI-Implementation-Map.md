# diaggui GUI Implementation Map

`diaggui` のGUIは、アプリケーション固有のエントリーポイントと、共通GUIライブラリ（`dttgui` / `TLG`クラス群）の2層構造で構築されています。

## 1. アプリケーション・エントリーポイント

**パス**: `src/dtt/gui/diaggui.cc`

プログラムの起動とメインウィンドウの初期化を担当します。

| 関数 / ブロック | 役割・情報 |
| --- | --- |
| **`main`** | **プログラム開始地点**。<br>

<br>- コマンドライン引数の解析 (`_argGUI`, `_argServer` 等)<br>

<br>- ROOTアプリケーション (`TApplication`) の初期化<br>

<br>- `DiagMainWindow` (メイン画面) のインスタンス作成と表示 (`mainWindow.Print` または `theApp.Run`) |
| `diagcommandline` | CLIとGUIのブリッジ。カーネルからの通知を受け取るコールバックを設定。 |

## 2. メインウィンドウの実装 (Base Class)

**パス**: `src/dtt/gui/dttgui/TLGMainWindow.cc`

LIGO診断ツール共通のメインウィンドウ基底クラス (`TLGMainWindow`) です。ウィンドウの骨組み（メニュー、ボタン、プロットエリア）を構築します。

| 関数名 | 役割・情報 |
| --- | --- |
| **`TLGMainWindow`** (Constructor) | **初期設定**。<br>

<br>- プロットセット (`PlotSet`)、印刷設定、エクスポート設定の初期化<br>

<br>- XMLオブジェクトやメッセージキューの準備 |
| **`SetupWH`** | **GUI部品の配置 (Layout)**。<br>

<br>- `MenuSetup`: メニューバーの構築<br>

<br>- `GetStatusBar`: ステータスバーの配置<br>

<br>- `AddMainWindow`: メインのプロットエリア (`TLGMultiPad`) の追加<br>

<br>- `AddButtons`: 画面下部の操作ボタンの追加<br>

<br>- タイマー (`fHeartbeat`, `fX11Watchdog`) の開始 |
| **`AddStdButtons`** | **標準ボタンの生成**。<br>

<br>- "Start", "Stop", "Pause", "Abort", "Exit" などのボタンを作成し、イベントハンドラを紐付けます。<br>

<br>（`SetupWH` から呼ばれます） |
| **`ShowDefaultPlot`** | **プロット画面の初期表示**。<br>

<br>- 測定タイプ（パワースペクトル、伝達関数、時系列など）に応じて、適切なグラフタイプ (`kPTPowerSpectrum` 等) を選択し表示します。 |
| `ProcessButton` | **ボタン操作の処理**。<br>

<br>- 各ボタンID (`kB_START`, `kB_EXIT` 等) に応じた処理関数 (`ButtonStart`, `CloseWindow` 等) を呼び出します。 |
| `HandleTimer` | **イベントループ / ハートビート**。<br>

<br>- 定期的にメッセージキューを確認し、バックエンドからの通知を処理します。またX11の接続監視も行います。 |

## 3. メインウィンドウの実装 (Application Specific)

**パス**: `src/dtt/gui/diagmain.cc` (Header: `diagmain.hh`)

`TLGMainWindow` を継承し、`diaggui` 特有のメニュー項目や動作を実装しているクラスです。
*(※ファイルリストと依存関係からの推測)*

| クラス/役割 | 情報 |
| --- | --- |
| **`DiagMainWindow`** | `diaggui` のメインウィンドウクラス。`TLGMainWindow` を継承。<br>

<br>診断カーネル固有のコマンド処理やメニュー構成が含まれると推測されます。 |

## 4. GUI構成要素 (Widgets & Components)

**ディレクトリ**: `src/dtt/gui/dttgui/`

メインウィンドウ内で使用される個別のGUI部品群です。

| ファイル | クラス | 役割・情報 |
| --- | --- | --- |
| `TLGPlot.cc` | `TLGPlot` | **グラフ描画ウィジェット**。ROOTの描画機能をラップし、スペクトルや時系列データを表示。 |
| `TLGMainMenu.cc` | `TLGMainMenu` | **メニューバー**。「File」や「Help」などのメニュー項目の構築とイベント処理。 |
| `TLGChannelBox.cc` | `TLGChannelBox` | **チャンネル選択画面**。ツリー構造などで信号チャンネルを選択するUI。 |
| `TLGEntry.cc` | `TLGEntry` | **入力フィールド**。数値やテキストの入力用ウィジェット。 |
| `TLGFilterWizard.cc` | `TLGFilterWizard` | **フィルタ設計ウィザード**。デジタルフィルタの係数設計を行う対話画面。 |

## 5. 制御・ロジック (Logic)

**ディレクトリ**: `src/dtt/gui/`

GUIの操作を実際の診断コマンドに変換するロジック部分です。

| ファイル | 役割・情報 |
| --- | --- |
| `diagctrl.cc` | **診断制御 (Controller)**。<br>

<br>GUIからの「開始」「停止」等のアクションを受け取り、バックエンド（診断カーネル）へコマンドを送信する役割と推測されます。 |
| `diagplot.cc` | **描画ロジック**。<br>

<br>取得した診断データをどのように `TLGPlot` に渡して描画するかを制御するロジックが含まれます。 |

---

**補足**:

* GUIのベースフレームワークには **ROOT (CERN)** が使用されています。
* クラス名のプレフィックス `TLG` は "The LIGO GUI" を意味し、LIGO独自の拡張ウィジェットであることを示しています。