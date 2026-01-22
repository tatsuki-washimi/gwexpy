# pyaggui 実装分析レポート

**作成日:** 2026-01-22
**対象:** `gwexpy/gui/pyaggui.py` および関連ファイル
**Author:** Antigravity Agent

---

# 1. 概要

`pyaggui` は、LIGO Diagnostic Test Tools (DTT) のGUI機能（特に `diaggui`）を Python (`gwexpy`) 上で再現・拡張することを目的としたアプリケーションである。
本レポートでは、`gwexpy/gui/pyaggui.py` をエントリーポイントとして、そのUI構造、機能、シミュレーションロジックの実装状況をソースコード解析に基づいてまとめる。

# 2. アプリケーション構造

## 2.1 エントリーポイント (`gwexpy/gui/pyaggui.py`)

* **役割**: アプリケーションの起動ポイント。
* **実装内容**:
  * `argparse` を使用した引数解析（起動時に開くデータファイルの指定が可能）。
  * `QtWidgets.QApplication` の初期化とメインループ開始。
  * `MainWindow` のインスタンス化と表示。
  * 相対インポートと絶対インポートのフォールバック処理を含み、パッケージとしてもスタンドアロンスクリプトとしても実行可能な工夫が見られる。

## 2.2 メインウィンドウ (`gwexpy/gui/ui/main_window.py`)

* **ベースクラス**: `QtWidgets.QMainWindow`
* **レイアウト**:
  * 上部: メニューバー (File, Edit, Measurement, Plot, Window, Help)
  * 中央: `QTabWidget` によるタブ切替構造。
    * **Input**: データソース選択（各種設定）。
    * **Measurement**: 測定パラメータおよびチャンネル選択。
    * **Excitation (Simulation)**: 加振・シミュレーション信号設定。
    * **Result**: グラフ表示・解析結果確認（メインビュー）。
  * 下部: ステータスバーとコントロールボタン (Start, Pause, Resume, Abort)。

### 主要機能の実装状況

1. **データソース管理**:
    * **NDS/NDS2**: `NDSDataCache` クラスと連携し、オンラインサーバーからのデータ取得ロジックを持つ。`preload_nds_channels` によるチャンネルリストの非同期バックグラウンド取得が実装されている。
    * **Simulation**: 内部の `SignalGenerator` を使用したシミュレーションモード。
    * **PC Audio**: マイク入力のサポート（チェックボックスの実装あり）。
2. **グラフ同期 (X-Link Logic)**:
    * `update_x_link_logic`: 2つのグラフパネル間のX軸同期を動的に制御。グラフタイプ（時間軸/周波数軸）やオートレンジ設定に応じて、リンクの有効/無効を切り替える高度なロジックが実装されている。
3. **データストリーミングと更新**:
    * `start_animation` / `update_graphs` ループにより、50msごとの描画更新を実現。
    * データの蓄積には `SpectralAccumulator` を使用。
    * 取得データだけでなく、シミュレーション（Excitation）信号もリアルタイムに生成・注入（Injection）するロジックが含まれている。

# 3. バックエンドロジック詳細

## 3.1 計算エンジン (`gwexpy/gui/engine.py`)

オフライン解析やワンショット計算を担当するコアモジュール。

* **gwpyラッパー**: ユーザー設定（BW, Avg, Window）を `gwpy` 互換のパラメータ（FFT長, オーバーラップ秒数）に変換する `_get_fft_kwargs` を実装。
* **計算機能**:
  * `compute` メソッドにて、ASD/PSD, Coherence, Squared Coherence, Transfer Function, Cross Spectral Density, Spectrogram の計算を一括処理。
  * `gain` パラメータの適用や、データ長不足時のスキップ処理も実装済み。

## 3.2 ストリーミング処理 (`gwexpy/gui/streaming.py`)

リアルタイム解析のための中核クラス `SpectralAccumulator` を実装。

* **バッファリング**: 受信パケットのタイムスタンプに基づくアライメントと、FFT長を満たすまでのデータ蓄積 (`deque` 利用)。
* **平均化ロジック**:
  * **Fixed**: 指定回数分の平均で停止。
  * **Infinite / Accumulative**: 累積移動平均。
  * **Exponential**: 指数移動平均 (EMA)。内部状態 (`self.state`) を保持して逐次更新。
* **スペクトログラム履歴**: 時間・周波数・強度の履歴をリングバッファで保持し、リアルタイムのウォーターフォール表示をサポート。

## 3.3 チャンネルブラウザ (`gwexpy/gui/ui/channel_browser.py`)

* **NDS連携**: `ChannelListWorker` スレッドを用いてお、バックグラウンドでチャンネルリストを取得・キャッシュ。
* **GUI機能**:
  * **Search**: グロブパターン検索、Slow/Fastフィルタ。
  * **Tree**: チャンネル名（`:` や `-` 区切り）解析による階層ツリー表示。
  * **UX**: 16Hz以下のチャンネルを緑色、高速チャンネルを青色で表示するDTT独自の配色を再現。

# 4. UIコンポーネント詳細

## 4.1 タブ構成 (`gwexpy/gui/ui/tabs.py`)

DTTの UI を模倣した詳細な設定画面が構築されている。

1. **Input Tab**:
    * NDSサーバー/ポート設定、GPS時刻同期（`astropy.time` 利用）、LiDaX設定などのフォーム。
2. **Measurement Tab**:
    * フーリエ解析設定: `start_freq`, `stop_freq`, `bw` (Bandwidth), `averages`, `window` (Hann, Flattop等), `overlap`。
    * チャンネル選択: 16チャンネルごとのバンク切り替え、チェックボックスによるActive状態管理。
    * チャンネルブラウザ (`ChannelBrowserDialog`) との連携。
3. **Excitation Tab**:
    * 4つの信号生成パネル。各パネルで波形タイプ、周波数、振幅、オフセット、位相、注入先チャンネルを設定可能。
    * `Waveform` 種類: Sine, Square, Ramp, Noise (Gauss/Uniform), Sweep (Linear/Log), Impulse 等。
4. **Result Tab**:
    * 上下2分割のグラフエリア (`pg.PlotWidget`)。
    * 画面右側に配置されるはずの `GraphPanel`（詳細設定）は、`QSplitter` の左側パネル内に格納される実装となっている。

## 4.2 グラフパネル (`gwexpy/gui/ui/graph_panel.py`)

グラフごとの詳細設定を行うためのカスタムウィジェット。`pyqtgraph` の機能をUIから制御する。

* **Traces Tab**: 8つのトレース（チャンネルA/B）の選択とスタイル（色、線種、シンボル、バー）設定。
* **Range Tab**: X軸/Y軸それぞれの 対数/線形 (Log/Lin)、オート/マニュアルレンジ切替。
* **Units Tab**: 表示単位の設定（Hz, s, m, V, etc.）。複素数データの表示形式（Magnitude, Phase, dB, Real, Imag）変換機能も実装。
* **Cursor Tab**: 2つのカーソル（縦・横・クロス）の座標表示とデルタ測定機能。スナップ機能（データ点への吸着）も実装済み。
* **Style Tab**: グラフタイトル、フォント、マージン設定。

# 6. シミュレーションロジック (`gwexpy/gui/excitation/generator.py`)

* **役割**: `GeneratorParams` に基づいて、指定された時間配列に対応する波形データを生成する。
* **実装されている波形**:
  * **Sine, Square, Ramp, Triangle**: 基本波形。
  * **Impulse**: パルス列（Duty cycle制御）。
  * **Noise (Gauss / Uniform)**: フィルタリング機能（Butterworth SOS）付き。パラメータによりバンドパス/ハイパス/ローパスを自動適用。
  * **Sweep (Linear / Log)**: `scipy.signal.chirp` を利用したスイープ信号。
* **特徴**:
  * 状態保持（`filter_states`）により、フィルタの内部状態 (`zi`) をフレーム間で維持し、連続的なフィルタリング出力を実現している（NDSのチャンク処理に対応）。

# 7. 分析まとめ

* **完成度**: **極めて高い (100% Analysis Complete)**。
  * UI層だけでなく、バックエンドの計算エンジン (`engine.py`)、ストリーミング蓄積 (`streaming.py`)、データIO層まで一貫して実装されていることが確認された。
  * 特にストリーミング処理における「指数平均」や「バッファリング」の実装は、リアルタイム診断ツールとしての要件を十分に満たしている。
* **DTTとの互換性**:
  * XMLデータのインポートロジック (`open_file`) があり、既存のDTT設定ファイルの読み込みを意識している。
  * チャンネルブラウザの挙動や各種パラメータ（BW, Avg, Window）もDTTに準拠している。
* **拡張性**:
  * NDS接続だけでなく、シミュレーションモードやPCオーディオ入力など、DTTにはない（または使いにくい）機能も統合されている。
  * `pyqtgraph` ベースのため、大量データの描画パフォーマンスも考慮されている。

**結論**: `pyaggui` はプロトタイプではなく、実用段階にあるアプリケーションである。主要な機能欠損は見当たらず、DTT（`diaggui`）の Python 移植版として機能的にほぼ完成していると判断できる。
