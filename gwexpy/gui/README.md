# gwexpy GUI (pyaggui)

> ⚠️ **警告 / Warning**
>
> このGUIモジュールは**開発中**です。APIやUIは改善のため予告なく変更される可能性があります。
>
> This GUI module is **under active development**. The API and UI may change without notice for improvements.

## 概要 / Overview

`pyaggui` は LIGO DTT (Diagnostic Test Tools) の `diaggui` を模したGUIツールで、`gwexpy` ライブラリを用いた時系列データのリアルタイム可視化、解析、およびファイルデータの閲覧を行います。

## 主な機能 / Features

### 1. データソース (Data Sources)

- **NDS / NDS2 (Online)**: KAGRA NDSサーバー等からデータを取得・表示します。
  - **Channel Browser**: サーバー上のチャンネルリストをブラウズし、計測対象を選択可能です。
  - **PC Audio**: PCのマイク入力やスピーカー出力をデータソースとして利用可能です。
- **Excitation (Simulation)**: 多彩な波形生成器を用いて、オンラインデータへの信号注入や単独のシミュレーション信号生成が可能です。
- **FILE (File Load)**: ローカルファイル（DTT XML, GWF, HDF5, CSV 等）を読み込んでデータを表示します。

### 2. チャンネルハンドリング (Channel Handling)

`diaggui` の設計思想に基づき、計測対象と表示対象を分離して管理します。

- **Measurement タブ**:
  - データ取得を行うチャンネルを選択し、`Active` に設定します。
  - チャンネルブラウザからの追加や、ファイル読み込み時の設定復元に対応しています。
- **Results タブ**:
  - **表示の選択**: Measurement タブで `Active` になっているチャンネルから、グラフに描画する対象を選択します。
  - **複数グラフ**: 2つの独立したグラフパネルを持ち、それぞれで異なる解析（時系列、ASD、コヒーレンス、スペクトログラム等）を同時に表示できます。

### 3. 多彩な解析と表示 (Analysis & Plotting)

- **解析タイプ**: Time Series, ASD, CSD, Coherence, Squared Coherence, Transfer Function, Spectrogram.
- **豊かな表示オプション**:
  - **Style**: 線種、シンボル、色、軸のスケール（Linear/Log）のカスタマイズ。
  - **Legend / Cursor**: 凡例の表示や、データポイントへのスナップ機能を持つカーソルによる値の読み取り。
  - **Display**: dB表示、位相表示、アンラップ表示等への切り替え。

### 4. ファイルサポート (File Support)

`gwexpy` の強力なI/O機能を利用して、LIGO/KAGRAで標準的なデータ形式に加え、各種計測器や汎用的なデータ形式の読み込みに対応しています。

| 形式 / Format | 拡張子 / Extension | 読み込み可能な情報 / Data Types | 備考 / Remarks |
| :--- | :--- | :--- | :--- |
| **DTT XML** | `.xml` | TS, ASD, CSD, COH, TF, **計測状態** | **推奨形式**。計測設定や解析結果を完全に復元。 |
| **LIGO LW XML** | `.xml` | TS, ASD, CSD, COH, TF | DTT形式でない一般のLIGO LW形式。 |
| **GW Frame** | `.gwf` | TS | 重力波観測データの標準形式。 |
| **HDF5** | `.h5`, `.hdf5` | TS, ASD, CSD, COH, TF, Spectrogram | 階層的な汎用データ形式。 |
| **MiniSEED / SAC** | `.mseed`, `.sac` | TS | 地震計等（ObsPy依存）で用いられる形式。 |
| **WAV Audio** | `.wav` | TS | 音声データ。マイク録音等の解析に利用可能。 |
| **NI TDMS** | `.tdms` | TS | **National Instruments** 製機器のデータ形式。 |
| **Graphtec GBD** | `.gbd` | TS | **Graphtec** 製データロガーのバイナリ形式。 |
| **Metronix ATS** | `.ats` | TS | **Metronix ADU** 製機器のデータ形式。 |
| **Text / CSV** | `.txt`, `.csv`, `.dat` | TS | カンマまたはスペース区切りのテキストデータ。 |
| **その他 / Others** | `.npy`, `.mat`, `.fits`, `.pkl`, `.ffl`, `.sdb` | TS | NumPy, MATLAB, FFL(Frame File List)等。 |

> [!NOTE]
>
> - **TS**: Time Series (時系列), **ASD**: Amplitude Spectral Density (振幅スペクトル密度)
> - **CSD**: Cross Spectral Density, **COH**: Coherence, **TF**: Transfer Function
> - DTT XML形式以外のファイルでは、チャンネル設定（どのチャンネルがActiveか等）の保存・復元には対応していません。

## 既存ツールとの比較 / Comparison with Existing Tools

- **ndscope** (LIGO開発, KAGRA利用):
  - 時系列波形（もしくはトレンド）のみ対応ですが、本ツールはスペクトル解析やスペクトログラムにも対応しています。
- **diaggui** (LIGO DTT, KAGRA利用):
  - 本ツールは `diaggui` の操作感を再現しつつ、Apple Silicon Mac (ARM系) 等の現代的な環境でも動作し、さらに `diaggui` にはないスペクトログラム表示機能も備えています。

## 起動方法 / How to Run

```bash
cd gwexpy/gui
python pyaggui.py [filename]
```

## 依存関係 / Dependencies

- PyQt5
- pyqtgraph
- qtpy
- sounddevice (PC Audio利用時)
- nds2-client (NDS接続時)
- gwexpy (numpy, scipy, astropy, gwpy 等に依存)

## ディレクトリ構造 / Directory Structure

- `pyaggui.py`: エントリーポイント。
- `ui/`: メインウィンドウ、タブ、グラフパネル等のUI定義。
- `loaders/`: 各種ファイルフォーマットのローダーと、XMLパースロジック。
- `nds/`: NDS通信、PC Audio通信、およびデータバッファリング管理。
- `excitation/`: 信号生成エンジンとパラメータ管理。
- `plotting/`: グラフ描画のユーティリティ。

## 今後の課題 (Future Work)

- **NDS接続の堅牢化**: 特定のネットワーク環境下でのタイムアウト処理の改善。
- **コード統合**: `gui/loaders/` 内のXMLパースロジックの `gwexpy/io/` への共通化 (一部完了)。
- **機能拡張**: Fotonフィルタの直接読み込みや、より高度な統計解析機能の実装。
