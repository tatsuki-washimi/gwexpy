# GWexpy 外部ライブラリ統合の実装調査レポート（I/O とクラス変換）

## エグゼクティブサマリー

GWexpy（tatsuki-washimi/gwexpy）は **GWpy の TimeSeries/FrequencySeries/Spectrogram を継承しつつ**、行列型（`TimeSeriesMatrix`/`FrequencySeriesMatrix`/`SpectrogramMatrix`）と空間場（`ScalarField`/`VectorField`/`TensorField`）を追加し、さらに **多数の外部ライブラリとの相互変換（interop）**を `gwexpy/interop` に集中実装しています。fileciteturn26file0L1-L1 fileciteturn34file0L1-L1 fileciteturn35file0L1-L1  

今回指定されたライブラリ群のうち、**pyroomacoustics / PySpice / scikit-rf / SimPEG / gwinc(=pygwinc)** は既に GWexpy 側に実装が存在し、主に「変換対象の拡張」「メタデータ/単位/軸の厳密化」「（可能なら）GWpy I/O レジストリへのフォーマット登録」を追加する段階です。fileciteturn19file0L1-L1 fileciteturn20file0L1-L1 fileciteturn21file0L1-L1 fileciteturn22file0L1-L1 fileciteturn33file0L1-L1  

未実装（または直接統合が薄い）領域は、**LALSuite / PyCBC との往復変換の「GWexpy 型で返す」保証**、**Meep/openEMS/FEniCSx/emg3d の空間場データ（グリッド/メッシュ/複素場）の吸収**、さらに **振動モーダル（SDynPy/SDyPy/pyOMA/OpenSeesPy/Exudyn）と気象（MetPy/wrf-python）**の「標準表現（スキーマ）」策定です。  
実装優先度としては、(1) GW系（LAL/PyCBC）互換の最小差分、(2) HDF5/VTK/XDMF/NetCDF といった **中間フォーマット**を軸に空間場・メッシュ場を取り込む、(3) モーダル・気象を xarray/NetCDF/CSV で統一、の順が最短で効果が高いです。citeturn13search5turn15search4turn1search1turn5search5turn4search2  

---

## GWexpy の既存アーキテクチャと現在の統合状況

GWexpy の中核は、GWpy の系列クラスを拡張した **系列（Series）・行列（Matrix）・場（Field）**の3層です。

- 系列：`TimeSeries` は GWpy `TimeSeries` を継承し、追加ミキシン（解析・相互運用・前処理など）を統合しています。fileciteturn34file0L1-L1  
- 周波数系列：`FrequencySeries` は GWpy `FrequencySeries` を継承し、位相・微積分・dB変換・各種 interop を追加しています。fileciteturn35file0L1-L1  
- 行列：`FrequencySeriesMatrix` は「共通の周波数軸を持つ 2D コンテナ」で、CSD 行列や MIMO 伝達関数などに自然に対応します。fileciteturn36file0L1-L1  
- 空間場：`ScalarField`（4D：axis0 + xyz）を基底に、`VectorField`/`TensorField` を成分集合として表現します。fileciteturn23file0L1-L1 fileciteturn24file0L1-L1 fileciteturn41file0L1-L1  

I/O は基本的に GWpy の `read/write`（GWF/ASCII/HDF5 等）を継承しつつ、GWexpy 独自形式（例：DTTXML から周波数系列を読む）などを追加します。DTTXML 読み込みは `gwpy.io.registry.register_reader` を使う実装例になっています。fileciteturn11file0L1-L1  

相互運用（interop）は `gwexpy/interop` にまとまり、**ConverterRegistry** により “GWexpy側コンストラクタ（TimeSeries 等）” を遅延参照できる設計です。これにより、外部ライブラリ → GWexpy の変換で循環 import を避けられます。fileciteturn19file0L1-L1 fileciteturn21file0L1-L1  

既に存在する（今回の必須リストに直接関係する）統合例：
- pyroomacoustics：RIR（`room.rir`）、マイク信号（`room.mic_array.signals`）、STFT（`pyroomacoustics.transform.STFT`）などの変換。fileciteturn20file0L1-L1 fileciteturn44file0L1-L1  
- PySpice：`TransientAnalysis`→TimeSeries、`AcAnalysis/NoiseAnalysis`→FrequencySeries、全ノードを Dict として返す変換。fileciteturn21file0L1-L1  
- scikit-rf：`Network`→FrequencySeries/Matrix、`impulse_response/step_response`→TimeSeries/Dict、書き戻し（Network生成）も一部対応。fileciteturn22file0L1-L1  
- SimPEG：`simpeg.data.Data` への変換（Time/Frequency Domain EM）と逆変換。fileciteturn23file0L1-L1  
- gwinc（pygwinc）：`gwinc.load_budget(...).run()` の `trace.psd` から ASD を作るヘルパー。fileciteturn33file0L1-L1 citeturn3search2  

---

## 統合候補のデータタイプと GWexpy 側の推奨表現

要求されたデータタイプを、GWexpy の「既存クラス」でどう表すべきか（不足がある場合は追加提案も含む）を整理します。

**時系列（Time series）**  
- 単一チャネル：`TimeSeries(data, t0, dt, unit, name, channel, epoch)`  
- 多チャネル：`TimeSeriesDict` / `TimeSeriesMatrix`（センサ配列、ノード多数、マイクアレイなど）  

**周波数系列（Frequency series）**  
- PSD/ASD：`FrequencySeries`（実数、単位は `1/sqrt(Hz)` や `m/sqrt(Hz)` 等）  
- 複素スペクトル：`FrequencySeries`（複素）  
- クロススペクトル・コヒーレンス：`FrequencySeries`（2系列で出るもの）または `FrequencySeriesMatrix`（多チャネルCSD行列）fileciteturn36file0L1-L1  

**伝達関数（Transfer function / FRF / Network parameter）**  
- SISO：`FrequencySeries`（複素）  
- MIMO：`FrequencySeriesMatrix`（例：形状 `(n_out, n_in, n_freq)` を想定）  
- 例：scikit-rf の S行列、SDynPy の FRF 行列、FEniCSx から抽出した周波数応答など  

**時間周波数（Spectrogram / STFT）**  
- `Spectrogram`（複素STFT/パワー。pyroomacoustics STFT とは相互変換が既にあります）fileciteturn44file0L1-L1  

**インパルス応答（RIR / IR）**  
- `TimeSeries`（単一）または `TimeSeriesDict`（mic×src などペアが多い）  
- 伝達関数化：FFT → `FrequencySeries`  

**空間場（Spatial fields：部屋・格子・場グリッド）**  
- 正則グリッド（FDTD等、Meep/openEMS(emg3dのTensorMeshも近い)）：  
  - スカラー場：`ScalarField(axis0 + x + y + z)`  
  - ベクトル場：`VectorField({'x': ScalarField, 'y':..., 'z':...})`fileciteturn24file0L1-L1  
- 非正則（有限要素・非構造格子：FEniCSx/OpenSees等）：  
  - **提案：`MeshField`（新規）**または **xarray Dataset スキーマ**で “node/element + geometry + values” を保持  
  - 最小実装では「XDMF/VTK→meshio→xarray→（必要に応じて）正則グリッドへ補間→ScalarField」の2段階が現実的  

**振動モーダル（modal shapes / damping / mode tables）**  
- 推奨：  
  - モード周波数・減衰比：pandas DataFrame（列：`mode, f_Hz, zeta, ...`）  
  - モード形状：xarray Dataset（dims：`mode, node, dof`、coords：`x,y,z`）  
  - FRF/伝達：`FrequencySeriesMatrix`（in/out DOF を行列に）citeturn7search5  

**気象・流体・地球物理グリッド（meteorological fields / gravity-mag grids）**  
- 気象（WRF/MetPy）：xarray DataArray/Dataset を中心に、必要部分を `ScalarField`（時間×水平×鉛直）へ写像  
  - MetPy は `.metpy` アクセサと `.quantify()` による単位・座標処理が主戦場citeturn11search0  
  - wrf-python は `getvar` が xarray 対応で座標キャッシュ等を持つciteturn9search1  
- 重力・磁場（Harmonica）：xarray DataArray を前提に、正則グリッド変換の道具を提供しているciteturn5search1turn5search4  

---

## ライブラリ別の統合設計（I/O・クラス変換・ギャップ）

下表は、指定ライブラリ（最低限リスト＋関連）について「入出力形式」「インメモリ型」「GWexpy 推奨変換先」「GWexpy の現状」「不足点（追加実装）」を要約したものです。

### 主要統合マトリクス（ライブラリ×型×I/O）

| ライブラリ | 代表的インメモリ型 | 代表的I/O形式（公式/事実上） | GWexpy 推奨変換先 | GWexpy現状 | 追加実装ギャップ（要点） |
|---|---|---|---|---|---|
| **GWpy** | `TimeSeries`, `FrequencySeries`, `Spectrogram` | GWF, HDF5, ASCII, WAV 等（read/write）citeturn14search5turn14search0 | （同型）GWexpy は継承して互換fileciteturn34file0L1-L1 | 既存（基盤） | 追加不要。I/O レジストリ登録の拡張（例：Touchstone/VTKなど）を検討 |
| **LALSuite** | `REAL8TimeSeries`, `COMPLEX16FrequencySeries` 等（構造体）citeturn13search7turn15search0 | LALFrame（GWF関連）、内部構造は C API | TimeSeries/FrequencySeries（メタ：epoch, deltaT/F, unit, name, f0） | **未明示（GWpy経由で可）**：GWpyに `from_lal` ありciteturn13search4turn13search5turn15search4 | **GWexpy型で返す保証**（`TimeSeries.from_lal` を override し GWexpyへラップ）。FrequencySeries側も同様に追加 |
| **PyCBC** | `pycbc.types.TimeSeries/FrequencySeries`citeturn1search1turn1search2 | HDF5/ASCII/NPY 等のロード関数 | TimeSeries/FrequencySeries（dt/epoch/df等を厳密移送） | **未明示（GWpyに `from_pycbc`）**citeturn13search4 | `from_pycbc` を GWexpy 型で返すラッパー追加。周波数系列（df, epoch, complex）も対応表整備 |
| **pygwinc / gwinc** | `Budget`, `BudgetTrace`（freq, psd, asd, subtraces）citeturn3search2 | YAML, Python API | 総ASD：`FrequencySeries`、内訳：`FrequencySeriesDict`、（可能なら）CSD/TFはMatrix | **ASD生成ヘルパーあり**fileciteturn33file0L1-L1 | (1) `trace.asd`/`trace.psd` の選択、(2) サブトレース全展開（辞書/属性アクセス）を `FrequencySeriesDict` で返す、(3) 主要パラメータ（IFO, YAMLパス）を MetaData に保存 |
| **pyroomacoustics** | `Room`, `MicrophoneArray`, `STFT` | RIR/信号は numpy。Roomは内部状態 | RIR→TimeSeries/Dict、mic→TimeSeriesDict、STFT→Spectrogram/Dict、room field→ScalarField | **既に実装**fileciteturn20file0L1-L1 fileciteturn44file0L1-L1 | (1) `room.rir` の mic→src の順序をメタデータに明示（公式に list-of-lists で outer=mic, inner=src）citeturn16search0 (2) RT60等の派生値をMetaへ、(3) 空間離散化情報（room.fs, mic位置, src位置）の保持 |
| **PySpice** | `TransientAnalysis`, `AcAnalysis`, `NoiseAnalysis`（WaveForm）citeturn17search1turn17search7 | SPICE netlist、ngspice | Transient→TimeSeries/Dict、AC/Noise→FrequencySeries/Dict | **既に実装**fileciteturn21file0L1-L1 | (1) 単位の扱い（V/A/√Hz）を明示し `unit=` を自動設定、(2) Noise の「入力換算/出力換算」の区別を name/meta へ、(3) WaveFormの複素表現（ACの位相）に対応 |
| **scikit-rf** | `Network`, `Frequency` | Touchstone `.sNp`、pickle `.ntwk`citeturn0search1turn0search0turn0search7 | Network→FrequencySeries/Matrix、IR/step→TimeSeries/Dict、逆変換は `to_skrf_network` | **既に実装**fileciteturn22file0L1-L1 | (1) `z/y/h/a/t` 等の全パラメータ一般化、(2) Touchstone を GWpy I/O registry へ登録するか（`FrequencySeries.read("x.s2p")` 的UX）、(3) port名・z0 のメタ移送 |
| **Meep** | `Simulation`、場出力は HDF5（datasets） | HDF5出力、`output_field_function` が real/imag dataset を作るciteturn3search6 | HDF5→ScalarField/VectorField（Ex/Ey/Ez 等）、時間依存なら axis0=time | **未実装** | (1) HDF5 dataset 命名規約（`name*.r/.i` 等）を読んで複素場再構成citeturn3search6 (2) 格子座標（dx, origin）を軸に入れる（メタ or coords）、(3) 複数コンポーネントを `VectorField` へ束ねる |
| **openEMS** | `CSXCAD.ContinuousStructure`、dump設定は `AddDump` | Field dump：VTK または HDF5（時間/周波数、dump_typeで選択）citeturn3search7turn3search5turn3search9 | dump→ScalarField/VectorField、周波数dump→FrequencySeries/Field | **未実装** | (1) dumpファイル（VTK/HDF5）の読込ラッパー、(2) `DumpType` と物理量（E/H/J等）対応表の実装citeturn3search5turn3search9 (3) CSXCAD XML（形状/材料）をメタとして保存 |
| **FEniCSx（dolfinx）** | `dolfinx.fem.Function`, `Mesh` | XDMF(HDF5) / VTK / VTXWriter(ADIOS2)citeturn4search2turn4search0 | **非構造格子場**：推奨は xarray Dataset か `MeshField` 新設 | **未実装** | (1) XDMF/VTK 読み込み（meshio等）→統一表現、(2) `write_function` の制約（低次要素等）を考慮した補間ヘルパーciteturn4search0turn2search5 (3) GWexpy側に MeshField または Field↔xarray API を追加 |
| **SimPEG** | `simpeg.data.Data`, survey/source/rx, mesh | 基本はnumpy。EMモジュールの枠組みありciteturn4search1turn4search5 | Data↔Time/FrequencySeries、Fields/Model/mesh↔（xarray/MeshField） | **Dataのみ既存**fileciteturn23file0L1-L1 | (1) `Fields` や mesh（discretize）からの空間場取り込み、(2) 単位体系の整備（SI前提）、(3) 多受信点データを Matrix/Dict へ自然に割当 |
| **emg3d** | `emg3d.fields.Field`（fx,fy,fz view）、`TensorMesh` | `.h5/.npz/.json` で save/load（公式）citeturn5search2turn5search5turn5search0 | Field→VectorField（各成分をScalarFieldに）、I/OはHDF5経由で直接変換可 | **未実装** | (1) `Field.f{x,y,z}` を3D arraysとして軸/セル位置を決める、(2) frequency属性を axis0=frequency or metaに、(3) emg3d.io.load の出力 dict→GWexpy objects auto-cast |
| **SDynPy** | FRF生成関数、ShapeArray（UNV） | UNV/UFF（ユニバーサルファイル）等 | FRF→FrequencySeriesMatrix、Shape→xarray Dataset（mode,node,dof） | **未実装** | (1) `timedata2frf` の出力行列を GWexpy Matrixへマップciteturn7search5 (2) UNV読込→ShapeArray→標準スキーマ化citeturn7search6 |
| **SDyPy** | `sdypy.FRF`、`sdypy.io.uff` | UFF wrapper（pyuff）citeturn7search2turn7search0 | FRF→FrequencySeries/Matrix、UFF→（TimeSeries/Meta/Mode shapes） | **未実装** | UFFの dataset 種別→GWexpy表現の対応表を作る（最低：時系列/FRF/形状） |
| **Exudyn** | `SimulationSettings`, センサ、solution file | solution file（txt）、センサは [time, values...] で出力citeturn8search6turn8search1 | solution/センサ→TimeSeriesMatrix/Dict、（必要なら）CSV互換 | **未実装** | (1) 出力テキストのパーサ（時間列＋多列）→TimeSeriesMatrix、(2) センサ名/変数名のメタ保持 |
| **OpenSeesPy** | モデル状態、レコーダがファイル出力 | node recorder：text/xml/binary 等、time列オプションciteturn6search4turn6search1 | recorder出力→TimeSeriesMatrix（列=DOF@node） | **未実装** | (1) レコーダ出力の列定義（node/dof/respType）→列名生成、(2) XML出力にも対応する場合は lxml等 |
| **pyOMA** | OMA解析結果（独自構造） | プロジェクトディレクトリ（入出力形式は拡張中）citeturn6search0 | 時系列→TimeSeriesMatrix、安定化図/推定結果→DataFrame/xarray | **未実装** | (1) pyOMAが扱う入出力の最低共通（時系列/PSD/モード表）をGWexpy側で用意、(2) “結果セット”をHDF5/NetCDF で保存する枠組み |
| **acoustics（python-acoustics）** | 関数中心（dB演算等） | 特に標準I/Oなし | dB列→FrequencySeries（単位dB）など | **未実装** | 例：`dbsum/dbmean` 等で得たスペクトル/帯域値を Series化するユーティリティciteturn12search3 |
| **Harmonica** | xarray DataArray（正則グリッド） | NetCDF/xarray 互換が主流 | DataArray→ScalarField（2D/3D）または FrequencySeries | **未実装** | (1) CF属性/単位の扱い、(2) 高さ一定グリッド→ScalarField(axis0をtime/freq無しで扱う設計)citeturn5search1turn5search4 |
| **MetPy** | xarray + Pint（`.metpy` accessor、`.quantify()`）citeturn11search0 | NetCDF/GRIB は xarray 経由 | DataArray→ScalarField（気象場）、時系列→TimeSeries | **未実装** | (1) Pint単位→astropy単位への橋渡し、(2) CRS/座標をScalarFieldの axis/attrs に落とす |
| **wrf-python** | `wrf.getvar` が xarray DataArray 対応、座標キャッシュ等citeturn9search1turn10search3 | WRF出力 NetCDF | DataArray→ScalarField（time×z×y×x 等） | **未実装** | (1) WRF座標（XLAT/XLONG等）→Field軸へ、(2) 大規模データの lazy（dask）に配慮 |
| **mtspec / multitaper** | `mtspec.MTSpec/MTSine`（freq/spec/err）citeturn10search1 | 入力は配列、出力はオブジェクト | スペクトル→FrequencySeries、時間窓スペログラム→Spectrogram | **未実装** | (1) `MTSpec.spec` などを FrequencySeries に、誤差帯（err/CI）を Meta or 追加Seriesとして保持 |
| **python-control**（参考） | `FRD`, `TimeResponseData` | なし（pickle等） | FRD→FrequencySeries/Matrix、時応答→TimeSeries | 既存（GWexpy側に多い）fileciteturn35file0L1-L1 | 伝達の行列次元・入出力名の扱いを統一（Matrixメタ） |

---

## 互換ファイル形式とシリアライズ方針

「外部ライブラリを直接依存させず、現場で回せる」ためには、**中間フォーマット**をはっきり決めるのが最重要です。GWexpy は既に HDF5 / Zarr / netCDF4 への低レベル書き込み（dataset + attrs）を持つため、これを “共通の保存層” にするのが合理的です。fileciteturn43file0L1-L1 fileciteturn28file0L1-L1 fileciteturn42file0L1-L1 fileciteturn29file0L1-L1  

### 推奨する「標準保存」セット

- **GWF（GWフレーム）**：GW系時系列の標準（GWpyが read/write）。citeturn14search3  
- **HDF5**：  
  - GWpyは TimeSeries/FrequencySeries を HDF5 に保存する仕様を持つciteturn14search0turn14search5  
  - 空間場：Meep/emg3d/openEMS も HDF5 に寄っているため最優先citeturn3search6turn5search5turn3search7  
- **NetCDF（CF）**：気象・地球物理の配列は NetCDF が標準（wrf-python/MetPy/xarray）citeturn9search1turn11search0  
- **Zarr**：クラウド/分散・遅延ロードに強い保存（GWexpyに実装あり）fileciteturn42file0L1-L1  
- **Touchstone**：RF/ネットワークパラメータは scikit-rf の標準（`.sNp`）citeturn0search0turn0search2turn0search7  
- **CSV/TSV**：OpenSeesPy/Exudyn のテキスト出力、簡易交換（pandas経由）citeturn6search4turn8search6  
- **VTK / XDMF**：メッシュ場。FEniCSx/openEMSのdumpにも関係。XDMFはHDF5参照XMLで並列も可能citeturn4search2turn2search3turn3search9  

### 形式別の設計指針（GWexpyで実装すべき差分）

- **HDF5**：  
  - 既存の `to_hdf5_dataset/from_hdf5_dataset` は TimeSeries/FrequencySeries にあり、attrs に `t0/dt/unit/name` 等を保存しています。fileciteturn43file0L1-L1 fileciteturn28file0L1-L1  
  - 空間場・行列型にも同じ思想を拡張（attrs：axis座標配列、domain、component名、complex表現方式など）
- **NetCDF**：  
  - 現状は TimeSeries の最低限（t0/dt/unit）を variable attrs として保存。気象用途に拡張するなら CF準拠（`units`, `standard_name`, `coordinates` 等）への寄せが必要。fileciteturn29file0L1-L1  
- **VTK/XDMF**：  
  - dolfinx は XDMF/VTK を公式に提供し、XDMF は「XML + HDF5」を生成します。citeturn4search2turn2search3  
  - openEMS はダンプを VTK/HDF5 で出力可能であることを明示しています。citeturn3search7turn3search9  
  - GWexpy側は「読み込み時に mesh と値を分離して保持する」方が安全（MeshField/xarray）。
- **Touchstone**：  
  - scikit-rf は `Network.read_touchstone/write_touchstone` を提供し、Networkコンストラクタで読み込み可能です。citeturn0search2turn0search0turn0search1  
  - GWexpyは既に Network↔Series の変換があるため、ファイルI/Oは scikit-rf に任せ “GWexpy.read” には無理に統合しない（=薄いラッパーで十分）という選択も妥当です。

---

## 実装ロードマップとテスト戦略

### 優先度付きロードマップ（MVP / 中期 / 長期）

| フェーズ | タスク | 実装内容（最小→拡張） | 規模見積 | 主な単体テスト案 |
|---|---|---|---|---|
| **MVP** | LALSuite / PyCBC 変換の “GWexpy型で返す” | `TimeSeries.from_lal/from_pycbc` を override（GWpyの結果をGWexpyへ再構築）。FrequencySeries側も同様 | 小 | 乱数データで epoch/dt/df/unit/name が保たれること、complexも一致 |
| **MVP** | gwinc trace 展開 | totalだけでなく `trace['Quantum']` 等を `FrequencySeriesDict` 化（asd/psd選択可） | 小〜中 | サブトレース数・周波数一致、単位（ASD/PSD）一致 |
| **MVP** | Meep HDF5 field reader | `output_field_function` の `name*.r/.i` などを読んで複素場を再構成→ScalarField/VectorFieldciteturn3search6 | 中 | dataset名パターン、complex再構成、軸・shape検証 |
| **中期** | openEMS dump reader（VTK/HDF5） | dump_type→物理量(E/H/J)マップ、VTK/HDF5 をFieldへ | 中 | dump_typeごとの読み分け、単位・コンポーネント数 |
| **中期** | emg3d Field統合 | `Field.f{x,y,z}` view→VectorField、`emg3d.io.load/save` dict→GWexpy auto-castciteturn5search2turn5search0 | 中 | f{x,y,z} shape、grid coords、周波数属性の保持 |
| **中期** | FEniCSx（dolfinx）メッシュ場 | XDMF/VTK から mesh + function を読み込み、xarray Dataset or MeshFieldに格納。XDMFの制約も補間で回避citeturn4search0turn4search2 | 大 | 最低次要素のwrite_function、補間後の整合性、parallel時のpiece対応（将来） |
| **長期** | モーダル統一スキーマ | SDynPy/SDyPy/pyOMA/OpenSeesPy/Exudyn の “時系列/FRF/モード表/形状” を統一（xarray+DataFrame）。FRF→FrequencySeriesMatrix | 大 | 既知例（小梁など）でFRF shape、mode shapes dims、units |
| **長期** | 気象・環境場スキーマ | wrf-python/MetPy の xarray を ScalarField へ写像（座標、鉛直、CRS、単位）citeturn11search0turn9search1 | 大 | CF座標推定、単位変換(pint↔astropy)、lazy/dask保全 |
| **長期** | multitaper/mtspec 統合 | MTSpec/MTSine の `freq/spec/err` を FrequencySeries + error band として保持citeturn10search1 | 中 | 既知信号でピーク周波数一致、CIメタが保存される |

### 実装パターン（GWexpy流）

- 新規統合は `gwexpy/interop/<libname>_.py` に `from_<lib>` / `to_<lib>` を置き、`require_optional()` で依存を遅延読み込みする（既存実装と同型）。fileciteturn21file0L1-L1 fileciteturn22file0L1-L1  
- 主要クラス側（TimeSeries/FrequencySeries/Spectrogram/Field）には **薄いクラスメソッド**（`from_...`）を追加し、内部で interop 関数を呼ぶ。既に pyroomacoustics / skrf / pyspice でこの形式が使われています。fileciteturn43file0L1-L1 fileciteturn35file0L1-L1 fileciteturn44file0L1-L1  

---

## 図とコードパターンと前提条件

### 比較表：データタイプ対応（要約）

| データタイプ | 推奨GWexpy型 | 代表ライブラリ |
|---|---|---|
| 時系列（単一/多ch） | TimeSeries / Dict / Matrix | GWpy, PyCBC, PySpice(Transient), pyroomacoustics(mic), OpenSeesPy(recorder), Exudyn(sensor/solution)citeturn0search5turn1search1turn17search7turn16search0turn6search4turn8search6 |
| PSD/ASD | FrequencySeries | gwinc, PyCBC, multitaper, GWpy(asd/psd)citeturn3search2turn1search0turn10search1turn0search5 |
| CSD/FRF/TF/MIMO | FrequencySeriesMatrix | scikit-rf(Network S行列), SDynPy(FRF), SimPEG多Rx拡張citeturn0search1turn7search5turn4search1 |
| STFT/スペクトログラム | Spectrogram | GWpy, pyroomacoustics(STFT), multitaper(spectrogram関数)citeturn0search5turn0search3turn10search1 |
| RIR/IR | TimeSeries/Dict | pyroomacoustics(Room.rir), scikit-rf(impulse_response)citeturn16search0turn0search1 |
| 正則グリッド空間場 | ScalarField / VectorField | Meep(HDF5 field), openEMS dump, Harmonica(xarray grid)citeturn3search6turn3search9turn5search1 |
| 非構造メッシュ場 | xarray Dataset / MeshField(提案) | FEniCSx(dolfinx), OpenSees(要素場), SimPEG mesh | citeturn4search2turn2search3turn4search1 |
| モーダル/形状 | xarray Dataset + DataFrame | SDynPy/SDyPy/pyOMA | citeturn7search6turn6search0turn7search2 |
| 気象場 | xarray DataArray/Dataset→ScalarField | MetPy, wrf-python | citeturn11search0turn9search1 |

### Mermaid：エンティティ関係（GWexpy ↔ 外部型/ファイル）

```mermaid
flowchart LR
  subgraph GWexpy[GWexpy core]
    TS[TimeSeries]
    FS[FrequencySeries]
    SG[Spectrogram]
    FSM[FrequencySeriesMatrix]
    SF[ScalarField]
    VF[VectorField]
  end

  subgraph External[External libs / formats]
    GWpy[GWpy TimeSeries/FrequencySeries]
    LAL[LALSuite TimeSeries/FrequencySeries structs]
    PyCBC[PyCBC TimeSeries/FrequencySeries]
    GWINC[gwinc BudgetTrace]
    PRA[pyroomacoustics Room/STFT]
    PySpice[PySpice WaveForm Analyses]
    SKRF[scikit-rf Network/Frequency]
    MEEP[Meep HDF5 field outputs]
    OEMS[openEMS VTK/HDF5 dumps]
    DFX[dolfinx Function/Mesh]
    EMG3D[emg3d Field + io.save/load]
    WRF[wrf-python xarray]
    MetPy[MetPy xarray accessor]
    H5[(HDF5)]
    NC[(NetCDF)]
    GWF[(GWF Frame)]
    TSN[(Touchstone .sNp)]
    VTK[(VTK/XDMF)]
  end

  TS <--> GWpy
  FS <--> GWpy
  SG <--> GWpy

  TS <--> LAL
  FS <--> LAL
  TS <--> PyCBC
  FS <--> PyCBC

  FS <--> GWINC
  TS <--> PRA
  SG <--> PRA

  TS <--> PySpice
  FS <--> PySpice

  FS <--> SKRF
  FSM <--> SKRF

  SF <--> MEEP
  VF <--> MEEP
  SF <--> OEMS
  VF <--> OEMS

  SF <--> EMG3D
  VF <--> EMG3D

  SF <--> WRF
  SF <--> MetPy

  TS --> GWF
  TS --> H5
  FS --> H5
  SF --> H5
  SF --> NC
  FS --> TSN
  SF --> VTK
  DFX --> VTK
```

### Mermaid：ロードマップ（ガント）

```mermaid
gantt
  title GWexpy integration roadmap (proposal)
  dateFormat  YYYY-MM-DD
  axisFormat  %m/%d

  section MVP
  LALSuite/PyCBC conversions (return GWexpy types) :a1, 2026-03-25, 10d
  gwinc BudgetTrace -> FrequencySeriesDict/Series :a2, after a1, 7d
  Meep HDF5 Field reader (Scalar/VectorField) :a3, after a1, 14d

  section Mid-term
  openEMS dump reader (VTK/HDF5) :b1, 2026-04-20, 20d
  emg3d Field + io.load/save glue :b2, after b1, 15d

  section Long-term
  FEniCSx mesh-field schema (XDMF/VTK -> MeshField/xarray) :c1, 2026-06-01, 45d
  Modal schema (SDynPy/SDyPy/pyOMA/OpenSeesPy/Exudyn) :c2, 2026-06-15, 60d
  Meteorology schema (MetPy/wrf-python) :c3, 2026-06-15, 45d
  multitaper/mtspec wrappers :c4, 2026-07-15, 20d
```

### コードパターン例（読み書き/変換）

#### HDF5（GWpy互換 + GWexpyの dataset API）

GWpyは `TimeSeries.write('output.hdf', ...)` に対応し、既存ファイルへの append/overwrite も可能です。citeturn14search5turn14search1  
GWexpyはさらに `to_hdf5_dataset(group, path)` 形式の低レベルAPIを持ちます。fileciteturn43file0L1-L1  

```python
import h5py
from gwexpy import TimeSeries

ts = TimeSeries([0.0, 1.0, 0.5], t0=0, dt=1/1024, unit="m", name="test")

with h5py.File("out.h5", "w") as f:
    ts.to_hdf5_dataset(f, "chan/test", compression="gzip")

with h5py.File("out.h5", "r") as f:
    ts2 = TimeSeries.from_hdf5_dataset(f, "chan/test")
```

#### GWF（GWフレーム）

GWpyの `TimeSeries.write('output.gwf')` / `TimeSeries.read(file, channel)` パターンがそのまま使えます。citeturn14search3  

```python
from gwexpy import TimeSeries

channel = "X1:TEST-CHANNEL"
ts = TimeSeries([1,2,3], t0=0, dt=1, name=channel)
ts.write("out.gwf")          # GWFへ
ts2 = TimeSeries.read("out.gwf", channel)  # 読み戻し
```

#### gwpy FrequencySeries の I/O（ASCII/HDF5）

GWpyの周波数系列は ASCII/HDF5/XML に read/write でき、HDF5 では “ファイル内パス名” を指定します。citeturn14search0  

```python
from gwexpy import FrequencySeries
fs = FrequencySeries.read("psd.txt")                 # ASCII
fs.write("psd.h5", "psd", overwrite=True)            # HDF5 内の "psd" に保存
fs2 = FrequencySeries.read("psd.h5", "psd")          # 読み戻し
```

#### pyroomacoustics：RIR と STFT

- `room.rir` は “outer list=mic、inner list=source” の list-of-lists と明記されています。citeturn16search0  
- `STFT(N, hop, analysis_window, ...)` が公式 API。citeturn0search3  

```python
import pyroomacoustics as pra
from gwexpy import TimeSeries, Spectrogram

room = pra.ShoeBox([5, 4, 3], fs=16000, max_order=3)
# ... add_source / add_microphone_array ...
room.compute_rir()

rir_dict = TimeSeries.from_pyroomacoustics_rir(room)   # dict: mic/srcペアが展開される（既存実装）

stft = pra.transform.STFT(N=512, hop=128, channels=1)
# stft.analysis(signal) ... などで stft.X が埋まる想定
spec = Spectrogram.from_pyroomacoustics_stft(stft, fs=room.fs)
```

#### Meep：場の HDF5 ダンプ（読み取り側の方針）

Meepは `output_field_function(...)` で HDF5 に `name*.r` と `name*.i`（実部/虚部）の dataset を作る、と説明されています。citeturn3search6  
GWexpy側はこれを読んで `ScalarField/VectorField` を復元するのが最短です。

```python
# 読み取り側（GWexpyに from_meep_hdf5 を実装する想定の例）
import h5py
import numpy as np
from gwexpy.fields import ScalarField, VectorField

def load_meep_complex_dataset(h5path: str, base: str):
    with h5py.File(h5path, "r") as f:
        real = f[f"{base}.r"][...]
        imag = f[f"{base}.i"][...]
    return real + 1j * imag

Ez = load_meep_complex_dataset("Ez.h5", "Ez")  # shape: (nx, ny) 등
field = ScalarField(Ez, axis_names=("t_or_f", "x", "y", "z"))  # 座標は attrs から埋める設計を推奨
```

#### FEniCSx（dolfinx）：Function/Mesh のエクスポート

dolfinx の IO は `dolfinx.io` にまとまり、`VTKFile.write_mesh/write_function`、`XDMFFile.read_mesh/read_meshtags` などが提供されます。citeturn4search2turn4search4  
また XDMF 書き込みには要素次数等の前提条件があり、満たせない場合は補間や VTXWriter 推奨が明記されています。citeturn4search0turn2search5  

```python
from mpi4py import MPI
from dolfinx import io

# mesh, u: dolfinx.mesh.Mesh / dolfinx.fem.Function がある想定
with io.XDMFFile(MPI.COMM_WORLD, "out.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u, t=0.0)  # 要件を満たさない場合は補間してから
```

### 優先参照先（URLはコード内に限定して列挙）

```text
https://github.com/tatsuki-washimi/gwexpy

https://gwpy.readthedocs.io/
https://lscsoft.docs.ligo.org/lalsuite/
https://pycbc.org/pycbc/latest/

https://gwinc.docs.ligo.org/pygwinc/
https://pypi.org/project/gwinc/

https://pyroomacoustics.readthedocs.io/
https://pyspice.fabrice-salvaire.fr/
https://scikit-rf.org/
https://scikit-rf.readthedocs.io/

https://meep.readthedocs.io/  (または meep-hr.readthedocs.io)
https://openems.readthedocs.io/  / https://wiki.openems.de/
https://docs.fenicsproject.org/dolfinx/main/python/
https://simpegdocs.appspot.com/  (SimPEG docs)
https://emg3d.emsig.xyz/

https://sandialabs.github.io/sdynpy/
https://sdypy.readthedocs.io/
https://openseespydoc.readthedocs.io/
https://py-oma.readthedocs.io/

https://unidata.github.io/MetPy/
https://wrf-python.readthedocs.io/
https://www.fatiando.org/harmonica/
https://multitaper.readthedocs.io/
```

---

## 前提条件（未確定点の仮定を明示）

1. **GWexpy の系列型は GWpy の系列型を継承**しているため、GWpy の `read/write`・多くの classmethod は原理的に利用可能だが、`from_lal/from_pycbc` の返り値型が **GWpy型になってしまう場合がある**ため、GWexpy側で override して「必ず GWexpy型で返す」方針を推奨しました。fileciteturn34file0L1-L1 citeturn13search4  
2. `ScalarField` は（少なくとも設計上）**axis0 + 3D空間の正則グリッド**を主対象とし、FEniCSx のような非構造メッシュは **新スキーマ（xarray/MeshField）**が必要、という前提でロードマップを組みました。fileciteturn23file0L1-L1 citeturn4search2  
3. 単位系は GWpy/GWexpy が **astropy.units** を基調とする前提で、MetPy の Pint 単位は相互変換（または “attrs.units に落とす”）が必要と仮定しました。citeturn11search0  
4. Meep/openEMS の場データは HDF5/VTK で得られるとして、**まずはファイル読込→GWexpy場**を最短導線（シミュレーションオブジェクトを直接抱えない）としています。citeturn3search6turn3search9  
5. “日本語の一次資料” は重力波解析/数値計算系ライブラリでは限定的なため、原則は公式（英語）文書を一次として引用し、補助的に日本語ページ（例：jpドメイン）も混ぜています（ただし本レポートの主根拠は公式docsです）。

---

## MVP 実装計画（2026-03-25 追記）

### 概要

調査レポートの優先度付きロードマップに基づき、**MVP フェーズ 3 タスク** の具体的な実装手順を定義します。
3 タスクは互いに独立であり、並行実装が可能です。

---

### 既存 interop パターン（定型）

既存 interop（37モジュール）は以下の定型に従います：

- `gwexpy/interop/<libname>_.py` に `from_<lib>_<type>()` / `to_<lib>_<type>()` を配置
- `require_optional(“libname”)` で遅延 import
- `ConverterRegistry.get_constructor(“TimeSeries”)` で型取得（循環回避）
- `cls` パラメータで返り値型をディスパッチ（Series / Dict / Matrix）
- `__init__.py` の `__all__` にエクスポート
- テストは `tests/interop/test_interop_<libname>.py`

参照実装：`skrf_.py`（双方向）、`pyspice_.py`（片方向）、`finesse_.py`（ディスパッチ）

---

### Task 1: LALSuite / PyCBC 変換

**目的**：GWpy 継承の `from_lal` / `from_pycbc` は既に動作するが、interop モジュールとして
明示的に管理し、(a) メタデータ拡充、(b) `to_lal` / `to_pycbc` 逆変換、(c) テスト整備を行う。

#### 変更ファイル

| 操作 | ファイル |
|---|---|
| 新規 | `gwexpy/interop/lal_.py` (~80行) |
| 新規 | `gwexpy/interop/pycbc_.py` (~80行) |
| 新規 | `tests/interop/test_interop_lal.py` (~120行) |
| 新規 | `tests/interop/test_interop_pycbc.py` (~120行) |
| 変更 | `gwexpy/interop/_optional.py` — `”pycbc”` マッピング追加 |
| 変更 | `gwexpy/interop/__init__.py` — import + `__all__` 追加 |

#### 関数シグネチャ

```python
# gwexpy/interop/lal_.py
def from_lal_timeseries(cls, lalts, *, copy=True) -> TimeSeries
def to_lal_timeseries(ts, *, dtype=None) -> lal.REAL8TimeSeries
def from_lal_frequencyseries(cls, lalfs, *, copy=True) -> FrequencySeries
def to_lal_frequencyseries(fs) -> lal.REAL8FrequencySeries

# gwexpy/interop/pycbc_.py — 同構造
def from_pycbc_timeseries(cls, pycbc_ts, *, copy=True) -> TimeSeries
def to_pycbc_timeseries(ts) -> pycbc.types.TimeSeries
def from_pycbc_frequencyseries(cls, pycbc_fs, *, copy=True) -> FrequencySeries
def to_pycbc_frequencyseries(fs) -> pycbc.types.FrequencySeries
```

#### 実装方針

- `from_lal_timeseries`: `lalts.data.data`, `lalts.epoch`, `lalts.deltaT`, `from_lal_unit(lalts.sampleUnits)` を取り出し GWexpy 型で構築（`gwexpy/utils/lal.py` の既存ユーティリティを活用）
- `to_lal_timeseries`: `lal.Create<TYPE>TimeSeries` で逆構築
- PyCBC も同様（`pycbc_ts.data`, `pycbc_ts.start_time`, `pycbc_ts.delta_t`）

#### テストケース

- 乱数データの roundtrip（epoch/dt/df/unit/name 保持）
- 複素データの変換
- 単位変換の正確性（LAL unit ↔ astropy unit）

---

### Task 2: gwinc サブトレース展開

**目的**：既存 `gwexpy/noise/gwinc_.py` は total ASD のみ返す。
新規 interop モジュールとして、サブトレース（Quantum, Thermal, Seismic 等）を
`FrequencySeriesDict` で返す機能を追加する。既存 noise モジュールは維持（共存）。

#### 変更ファイル

| 操作 | ファイル |
|---|---|
| 新規 | `gwexpy/interop/gwinc_.py` (~150行) |
| 新規 | `tests/interop/test_interop_gwinc.py` (~150行) |
| 変更 | `gwexpy/interop/__init__.py` — import + `__all__` 追加 |

#### 関数シグネチャ

```python
def from_gwinc_budget(
    cls: type,                          # FrequencySeries or FrequencySeriesDict
    budget_or_model: Any,               # gwinc.Budget or str (“aLIGO”, “Aplus”)
    *,
    frequencies: np.ndarray | None = None,
    quantity: Literal[“asd”, “psd”] = “asd”,
    trace_name: str | None = None,      # None → total or all traces
    fmin: float = 10.0,
    fmax: float = 4000.0,
    df: float = 1.0,
) -> FrequencySeries | FrequencySeriesDict
```

#### 実装方針

- `budget_or_model` が str なら `gwinc.load_budget(model)` → `budget.run(freq=frequencies)`
- `trace` の再帰走査でサブトレース名と PSD を収集
- `quantity == “asd”` なら `np.sqrt(psd)`
- ディスパッチ: `trace_name` 指定 → 単一 `FrequencySeries`、`cls` が Dict → 全展開 `FrequencySeriesDict`
- metadata に model 名、quantity を保持

#### テストケース

- Total ASD/PSD の返り値と単位
- FrequencySeriesDict での全トレース展開
- 特定 trace_name の抽出
- 全トレースの周波数軸一致
- metadata 保持
- 不正な trace_name で ValueError

---

### Task 3: Meep HDF5 field reader

**目的**：Meep の `output_field_function` が生成する HDF5（`name.r` / `name.i` ペア）を読み込み、
`ScalarField` / `VectorField` を構築する。

#### 変更ファイル

| 操作 | ファイル |
|---|---|
| 新規 | `gwexpy/interop/meep_.py` (~200行) |
| 新規 | `tests/interop/test_interop_meep.py` (~200行) |
| 変更 | `gwexpy/interop/_optional.py` — `”meep”` マッピング追加 |
| 変更 | `gwexpy/interop/__init__.py` — import + `__all__` 追加 |

#### ScalarField コンストラクタ仕様

```python
ScalarField(
    data,               # 4D array (axis0, x, y, z)
    unit=None,
    axis0=None,         # 時刻 or 周波数の座標配列
    axis1=None, axis2=None, axis3=None,
    axis_names=None,    # 4要素リスト
    axis0_domain=”time”,    # “time” | “frequency”
    space_domain=”real”,    # “real” | “k”
)
```

#### 関数シグネチャ

```python
def from_meep_hdf5(
    cls: type,                          # ScalarField or VectorField
    filepath: str | Path,
    *,
    field_name: str | None = None,      # 自動検出 or 指定
    component: str | None = None,       # “ex” 等。None → 全コンポーネント
    resolution: float | None = None,    # pixels/unit length
    origin: tuple[float, ...] | None = None,
    axis0_domain: Literal[“time”, “frequency”] = “frequency”,
    unit: Any | None = None,
) -> ScalarField | VectorField
```

#### 実装方針

- `require_optional(“h5py”)` のみ（meep 本体は不要）
- `*.r` / `*.i` ペア → 複素場、suffix なし → 実数場
- 1D/2D/3D データは 4D に拡張（axis0 を singleton）
- 複数コンポーネント (ex, ey, ez) → `VectorField({“x”: sf_ex, “y”: sf_ey, “z”: sf_ez})`
- `resolution` と `origin` から等間隔空間座標を生成

#### テストケース（h5py で一時ファイルを生成）

- 3D 実数場 → ScalarField (shape: 1, nx, ny, nz)
- 3D 複素場 → ScalarField (complex dtype)
- 空間座標の正確性（resolution, origin）
- ex/ey/ez → VectorField (3 components)
- データセット未検出 → ValueError

---

### 実装順序と検証

3タスクは完全に独立。推奨マージ順序：Task 1 → Task 2 → Task 3

```bash
# 各タスクのテスト
pytest tests/interop/test_interop_lal.py -v
pytest tests/interop/test_interop_pycbc.py -v
pytest tests/interop/test_interop_gwinc.py -v
pytest tests/interop/test_interop_meep.py -v

# 回帰確認
pytest tests/interop/ -v
ruff check gwexpy/interop/ tests/interop/
mypy gwexpy/interop/lal_.py gwexpy/interop/pycbc_.py gwexpy/interop/gwinc_.py gwexpy/interop/meep_.py
```