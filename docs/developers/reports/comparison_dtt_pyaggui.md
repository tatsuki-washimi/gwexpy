# DTT vs pyaggui 比較分析レポート

**作成日:** 2026-01-22
**Author:** Antigravity Agent
**Referenced Documents:**

- `docs/developers/reports/dtt_analysis_report.md`
- `docs/developers/reports/analyze_pyaggui_implementation.md`

---

# 1. 概要

本レポートでは、LIGO Diagnostic Test Tools (DTT) のC++ソースコード分析結果と、Python移植版である `pyaggui` の実装状況を比較し、機能の網羅率と技術的な差異を明確にする。

**結論要約:**
`pyaggui` は **DTT の「Fourier Tools (FFT解析)」モードの機能をほぼ100%再現** しており、さらに現代的なUIと独自機能（PCオーディオ、シミュレーション）を追加している。
一方で、**「Swept Sine (周波数掃引)」測定モードについては、励起信号の生成機能はあるものの、解析ロジック（同期検波による高精度計測）は未実装** であり、FFTベースの伝達関数推定で代用されている状態である。

---

# 2. アーキテクチャ比較

| 項目 | DTT (Legacy) | pyaggui (New) | 評価 |
|---|---|---|---|
| **言語** | C++ (ROOT Framework) | Python (PyQt5 + pyqtgraph) | `pyaggui` が圧倒的に現代的で保守性が高い |
| **構成** | クライアント(`diaggui`) - サーバー(`diagd`) 分離型 | モノリシック (GUI + Worker Threads) | `pyaggui` はローカル動作においてはシンプル。リモート制御にはNDSサーバーへの依存が必要。 |
| **データ取得** | NDS1/2, Shared Memory, File | NDS2, Local Audio, Simulation | `pyaggui` はローカルシミュレーション機能が強力 |
| **グラフライブラリ** | ROOT (TGraph) | pyqtgraph (OpenGL/Qt) | `pyqtgraph` は高速でインタラクティブ性が高い |

---

# 3. 機能実装状況詳細比較

## 3.1 Fourier Tools (FFT解析)

DTTの主要機能である `Fourier Tools` モードは、`pyaggui` において非常に高いレベルで移植されている。

| 機能 | DTT実装 | pyaggui実装 | 状態 |
|---|---|---|---|
| **PSD/ASD** | `ffttest::fftToPs` | `gwpy.TimeSeries.asd` | ✅ 実装済み |
| **Coherence** | `ffttest::coherenceCP` | `gwpy.TimeSeries.coherence` | ✅ 実装済み |
| **Cross Spectral Density** | `ffttest::crossPower` | `gwpy.TimeSeries.csd` | ✅ 実装済み |
| **Spectrogram** | あり | `gwpy.Spectrogram` + リングバッファ | ✅ 実装済み |
| **Time Series** | `timeseries` test | `SpectralAccumulator` raw output | ✅ 実装済み |
| **平均化 (Averaging)** | Linear, Exponential (Fixed/Infinite) | Fixed, Infinite, Exponential (Custom Impl) | ✅ 完全再現 |
| **Window関数** | Hann, Flat-top, Uniform | Hann, Boxcar (Uniform), etc. | ✅ 実装済み |
| **Zoom FFT** | **あり** (Heterodyne + Decimate) | **なし** (標準FFTのみ) | ⚠️ **未実装** |

## 3.2 Swept Sine (周波数掃引測定)

制御系の伝達関数計測に不可欠なモード。

| 機能 | DTT実装 | pyaggui実装 | 状態 |
|---|---|---|---|
| **励起信号生成** | Sine, Swept Sine (Linear/Log) | Sine, Sweep (Linear/Log) | ✅ 実装済み (Generator) |
| **解析ロジック** | **同期検波 (Lock-in Detection)** | **FFTベース (`Transfer Function`)** | ⚠️ **方式が異なる** |
| **計測フロー** | 周波数ごとに Wait & Measure | 連続スイープ + FFT | ⚠️ **DTTと同等のS/Nは出ない** |

- **解説**: DTTの `Swept Sine` は、特定周波数で正弦波を揺らし、定常状態になるまで待ってから計測を行うことで、ノイズに強い高精度な計測を行う。`pyaggui` の現状の実装（`Engine.compute` 内での `transfer_function`）は、入出力のスペクトル比 (`Pxy/Pxx`) を取る Welch法 であり、これは広帯域なノイズ励起やチャープ信号を用いる場合には有効だが、厳密な「Swept Sine 測定」の代替にはならない。

## 3.3 Excitation (加振・信号生成)

| 機能 | DTT実装 | pyaggui実装 | 状態 |
|---|---|---|---|
| **波形タイプ** | Sine, Square, Ramp, Noise, Impulse, Const | Sine, Square, Ramp, Noise, Impulse, Sweep | ✅ 実装済み |
| **フィルタリング** | IIR Filter (Foton連携) | Butterworth SOS (Noiseのみ) | ⚠️ 汎用フィルタは未確認 |
| **加算・合成** | 複数波形の合成が可能 | 4スロットでの合成が可能 | ✅ 実装済み |

## 3.4 その他機能

| 機能 | DTT | pyaggui | 備考 |
|---|---|---|---|
| **チャンネル選択** | ツリービュー, ワイルドカード | ツリービュー, グロブ検索, Slow/Fastフィルタ | ✅ pyagguiの方が使いやすい |
| **単位変換** | Calibrationデータ適用 | `gain` パラメータのみ | ⚠️ Calibration連携は簡易的 |
| **カーソル** | クロスヘア, デルタ, ハーモニクス | クロスヘア, デルタ, スナップ | ✅ 実装済み |
| **エクスポート** | XML, ASCII | XML (Read), CSV/Image (Export) | ✅ 実装済み |

---

# 4. 今後の課題 (Gap Analysis)

`pyaggui` を DTT の完全な代替とするために不足している要素は以下の通りである。

1. **Swept Sine Measurement (Point-by-Point) の実装**:
    - 現状の `SignalGenerator` (Sweep) と `Engine` (FFT TF) の組み合わせでは、制御ループの精密な同定（特に低周波や高Q値の共振測定）において DTT に劣る可能性がある。
    - **対策**: `Engine` または `Analysis` クラスに、入力信号を参照信号とした同期検波ロジック（`sine_analyze`）を実装し、Generatorと連携してステップごとの計測を行うシーケンスを追加する必要がある。

2. **Zoom FFT**:
    - 特定の共振ピークを細かく見たい場合に DTT で多用される機能。
    - **対策**: `scipy.signal` を用いたヘテロダインとデシメーション処理を追加する。

3. **Calibration / Units**:
    - 現状は単純な `gain` 乗算のみ。DTT (Ezca/GDS) のようにシステムごとのキャリブレーション定数を自動適用する仕組みがあると望ましい。

---

# 5. 総評

`pyaggui` は、日常的なスペクトル解析（Noise Hunting、振動解析など）においては、すでに DTT 以上の使い勝手と十分な機能を有している。
ただし、**「伝達関数測定（Swept Sine）」に関しては、まだ DTT のロジックを完全には模倣できていない**。この点が実装されれば、DTT からの完全移行が現実的になる。

---

# 6. UI詳細比較

UIレイアウト、ボタン挙動、設定パネルの制御に関する詳細な比較。

## 6.1 コントロールバー (Start/Stop/Abort)

DTTの基本的な測定制御フローを再現しつつ、Pause/Resumeの実装により利便性を向上させている。

| ボタン | DTT (diaggui) | pyaggui | 挙動の詳細 |
|---|---|---|---|
| **Start** | 測定開始 | 測定開始 | タイマー(`start_animation`)起動、データ取得開始。設定ロック。 |
| **Pause** | (Pauseボタンあり) | **Pause** | データ取得・描画の**一時停止**。バッファは保持される。 |
| **Resume** | - | **Resume** | 一時停止した状態から測定・描画を再開。 |
| **Abort** | Abort | **Abort** | **完全停止**。データバッファ消去、初期化(`stop_animation`)。 |

## 6.2 画面レイアウトと分割機能

| 機能 | DTT | pyaggui | 評価 |
|---|---|---|---|
| **基本画面** | 1枚の大きなプロットエリア | **上下2段分割** (`QSplitter`) | デフォルトで2つのグラフを同時に見られる構成。 |
| **分割の自由度** | **高** (上下左右、Grid配置可能) | **中** (上下2段固定) | pyagguiは現状、3つ以上のグラフ追加や左右分割は不可。 |
| **設定パネル** | 別ダイアログ (`Graph Options`) | **ドッキングパネル** (`GraphPanel`) | グラフの右側に常に表示され、即座に調整可能。モダンなUX。 |
| **パネル表示制御** | ダイアログの開閉 | スプリッターによる調整 | スプリッターバーを動かしてパネルを隠すことが可能。 |

## 6.3 グラフ設定パネル (`GraphPanel`) の構成

pyagguiでは、DTTの分散していた設定項目をタブ切り替え式のサイドパネルに集約している。

| タブ名 | 設定項目 | DTT相当機能 | 備考 |
|---|---|---|---|
| **Traces** | Trace 1-8 選択、色、線種、シンボル | `Trace Properies` | 8本までオーバーレイ可能。色同期機能あり。 |
| **Range** | X/Y軸の Log/Lin, Auto/Manual 切替 | `Axis Options` | DTT同様、軸ごとの独立設定が可能。 |
| **Units** | 単位表示 (Hz, V等)、表示モード (Mag, Phase, dB) | `Units` / `Display` | 複素データの表示形式切替が可能。 |
| **Cursor** | カーソル1/2選択、Type(Vert/Horiz/Cross)、Delta表示 | `Cursor` tool | **スナップ機能** (データ点吸着) が実装されており使いやすい。 |
| **Style** | タイトル、フォントサイズ、マージン | `Main Options` | タイトル位置やマージン調整が可能。 |

## 6.4 特筆すべきUI機能 (pyaggui)

1. **X-Link (軸同期) の自動化**:
    - `update_x_link_logic`: グラフの種類（時間軸 vs 周波数軸）とオートレンジ設定に基づき、**上下のグラフのX軸同期を自動的にON/OFF**する高度なロジックを実装。ユーザーが手動でリンク設定をする手間を省いている。
2. **インライン編集**:
    - カーソル値の表示ラベル (`QLineEdit`) を直接編集することで、カーソル位置を数値指定で移動可能。
3. **Simulation Mode**:
    - NDSサーバーがない環境でも、内蔵シグナルジェネレータ (`SignalGenerator`) を用いて、実際のDTTのような測定フロー（励起→計測）を体験・テストできる。
