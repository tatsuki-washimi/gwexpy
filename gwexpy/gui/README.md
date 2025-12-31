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
- **NDS (Online)**: KAGRA NDSサーバー等からリアルタイムでデータを取得・表示します。（**※現在、接続動作およびチャンネルリスト取得機能は未検証・未実装です**）。
    - NDS接続がない場合でも、**Excitation (Simulation)** タブで信号生成を有効にすることで、擬似信号を用いた動作確認（シミュレーション）が可能です。
- **FILE (File Load)**: ローカルファイル（DTT XML, GWF, HDF5 等）を読み込んでデータを表示します。

## 既存ツールとの比較 / Comparison with Existing Tools

* **ndscope** (LIGO開発, KAGRA利用):
    * 時系列波形（もしくはトレンド）のみ対応で、スペクトル等の描画機能はありません。
* **diaggui** (LIGO DTT, KAGRA利用):
    * 時系列波形、スペクトル(ASD, CSD, coherence, TF)描画機能がありますが、スペクトログラムは非対応です。
    * ARM系CPUのコンピュータ（Apple Silicon Mac等）は非対応です。
* **Virgo dataDisplay**:
    * リアルタイムスペクトログラムにも対応していますが、導入障壁が高いです（RedHat系Linux限定、cmake必須、サポート終了した外部ツール依存など）。

## 本ツールと LIGO DTT (diaggui) との互換性 (目標) / Compatibility Goals

### UIの再現性
`diaggui` のXMLファイルのI/Oだけでなく、特に**UI部分を忠実に再現すること**を目的としています。
これは、LIGO, KAGRAの現場コミッショナーがスムーズに利用できるようにするためです。
（現状、見た目は似ていますが、未実装の機能も残されています。）

### Excitation（加振）機能の扱い
`gwexpy` はあくまでデータ表示・分析ツールであり、装置への操作は目的外です（安全面も理由の一つ）。
そのため、**Excitation (Simulation) タブ** は以下のようなローカルシミュレーション機能として実装されています：

1.  **読み取り専用 (Read-Only)**: 装置へは信号を送りません。
2.  **高性能信号生成器**:
    - Sine, Square, Ramp, Triangle, Impulse, Noise (Gauss/Uniform), Arbitrary, Sweep (Linear/Log) などの多様な波形を生成可能です。
    - Band-limited Noise (フィルタリング) や 周波数スイープ機能も実装されており、`diaggui` と同等のパラメータ設定でシミュレーションを行えます。
    - 生成された信号は、"Excitation-0" などのチャンネル名で Result タブから選択・表示可能です。

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
    - `extract_xml_channels` などの共通ロジックは **`gwexpy/io/dttxml_common.py`** を利用します。
- `engine.py`: シミュレーションデータの生成や信号処理を行うバックエンド。

## 既知の問題 / 今後の課題 (Known Issues / Future Work)

- **NDS接続**: 接続動作およびチャンネルリスト取得機能は未検証・未実装です。
- **ファイル対応**: XML以外の形式（GWF等）におけるチャンネルハンドリングのDTT互換化。

