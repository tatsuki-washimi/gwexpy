# gwexpy GUI (pyaggui)

> ⚠️ **警告 / Warning**
>
> このGUIモジュールは**開発中**であり、**不安定**です。
> APIやUIは予告なく変更される可能性があります。
> 本番環境での使用は推奨しません。
>
> This GUI module is **under active development** and is **unstable**.
> The API and UI may change without notice.
> Not recommended for production use.

## 概要 / Overview

`pyaggui` は LIGO DTT (Diagnostic Test Tools) の `diaggui` を模したGUIツールで、`gwexpy` ライブラリを用いた時系列データのリアルタイム可視化、解析、およびファイルデータの閲覧を行います。

## 主な機能 / Features

### 1. データソース (Data Sources)
- **NDS (Online)**: KAGRA NDSサーバー等からリアルタイムでデータを取得・表示します（**※現在、接続動作およびチャンネルリスト取得機能は未検証・未実装です**）。
- **SIM (Simulation)**: 内部で生成されたダミーデータ（サイン波、ノイズ等）を使用して動作確認が可能です。
- **FILE (File Load)**: ローカルファイル（DTT XML, GWF, HDF5 等）を読み込んでデータを表示します。

### 2. チャンネルハンドリング (Channel Handling)
`diaggui` の設計思想に基づき、計測対象と表示対象を分離して管理します。

- **Measurement タブ**:
    - **計測対象の定義**: データ取得を行うチャンネルを選択し、`Active` に設定します。
    - ファイル読み込み時（特にXML）は、ファイル内の設定がここに復元されます。
- **Results タブ**:
    - **表示の選択**: Measurement タブで `Active` になっているチャンネルの中から、グラフに描画するチャンネルを選択します。

### 3. ファイルサポート (File Support)
- **LIGO Light Weight XML (`.xml`)**:
    - DTT互換形式。チャンネル名に加え、**Active状態（計測設定）も復元**されます。
    - 現状、このリッチな機能（Measurementタブとの連携）は XML形式のみでサポートされています。
- **その他 (`.gwf`, `.miniseed`, `.h5` 等)**:
    - **申し送り事項 (Future Work)**: これらの形式については、現状読み込みは可能ですが、DTT互換のActive状態復元は**未対応**です。将来的な実装課題となります。

## 起動方法 / How to Run

```bash
cd gwexpy/gui
python pyaggui.py
```

## 依存関係 / Dependencies

- PyQt5
- pyqtgraph
- gwexpy (numpy, scipy, astropy, gwpy 等に依存)

## ディレクトリ構造 / Directory Structure

- `pyaggui.py`: エントリーポイント。アプリケーションの初期化を行います。
- `ui/`: 各タブやウィンドウのUI定義ファイル群（`main_window.py`, `tabs.py`, `graph_panel.py` 等）。
- `io/`: データ読み込み・パース関連（`loaders.py`, `products.py`）。
    - ※ **重要**: 現在、DTT XMLのパースロジック (`extract_xml_channels`) はワークスペース制限のため `gui/io/loaders.py` に一時的に実装されています。将来的には **`gwexpy/io/dttxml_common.py` へ統合・修正する必要があります**。
- `engine.py`: シミュレーションデータの生成や信号処理を行うバックエンド。

## 既知の問題 / 今後の課題 (Known Issues / Future Work)

- **NDS接続**: 接続動作およびチャンネルリスト取得機能は未検証・未実装です。
- **ファイル対応**: XML以外の形式（GWF等）におけるチャンネルハンドリングのDTT互換化。
- **コード統合**: `gui/io/loaders.py` 内のXMLパースロジックを `gwexpy/io/dttxml_common.py` へ移動。
