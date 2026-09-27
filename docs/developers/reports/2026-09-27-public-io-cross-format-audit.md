# Public I/O 横断監査

**実施日**：2026-09-27

**対象ソース**：GWexpy 0.2.4、`ae10234a1e37853508c54901bf7c9e80878f25aa`

**結果**：runtime実測した11形式の74シナリオを評価し、36シナリオでcorrectness defectを確認した。これらは独立した36件のbugではなく、9つのfix boundaryに集約できる。

実装、恒久テスト、公開契約は変更していない。

## 監査範囲

公開契約の25 canonical formatsは、静的な公開ルート一覧として全件記録した。

runtime characterizationは実測根拠がある11形式に限定し、未実測の14形式は `INVENTORY_ONLY` とした。

`BLOCKED` はruntime対象で有効な判定に至れなかった個別シナリオだけに付けている。

25形式のroute一覧は [public-route-inventory.json](2026-09-27-public-io-cross-format-audit/public-route-inventory.json)、runtime対象の区別は [runtime-scope-inventory.json](2026-09-27-public-io-cross-format-audit/runtime-scope-inventory.json) にある。

実測形式は `hdf5`、`sdb`、`wav`、`flac`、`gbd`、`tdms`、`mseed`、`win`、`ats`、`nc`、`zarr` である。

`gwf`、`hdf.ndscope`、`xml.diaggui`、`csv`、`txt`、`pickle`、`root`、`ogg`、`mp3`、`m4a`、`sac`、`gse2`、`knet`、`ats.mth5` は今回runtime対象にしていない。

ObsPyではminiSEEDだけを実測した。

圧縮音声ではFLACだけを実測したため、SAC/GSE2/KNETやOGG/MP3/M4Aの結果を同じbackend familyから推定していない。

## 判定方法

静的解析で生じた疑いは、独立生成fixtureをpublic entrypointへ渡して確認した。

GWexpy自身のwrite→readだけをoracleにせず、xarray、Zarr、h5py、npTDMS、ObsPy、SciPy、TinyTagの独立読取結果またはformat semanticsと比較した。

判定authorityは次の順に適用した。

1. 文書化されたGWexpy public contract
2. ファイル形式およびnative libraryのsemantics
3. GWpyが有限な正常結果を返す場合の挙動
4. 既存のGWexpy互換性方針
5. repositoryのテストとドキュメント

GWpyが正常な有限値を返す経路でGWexpyだけが別の有限結果を返す場合は、強い不具合根拠とした。

GWpyが例外、NaN、または非対応を返す場合、GWexpyの明示的なfail-closedだけを理由に不具合とは判定していない。

確認済みdefectは、public entrypointで到達できる場合にそのrouteから再現した。

個別の根拠、fixture、expected/actualは [runtime-characterization-matrix.json](2026-09-27-public-io-cross-format-audit/runtime-characterization-matrix.json) に記録した。

### 判定件数

74件の実測シナリオを次のように分類した。

| 判定 | 件数 |
| --- | ---: |
| `CONFIRMED_CORRECTNESS_DEFECT` | 36 |
| `CONFIRMED_COMPATIBILITY_BEHAVIOR` | 6 |
| `CONFIRMED_INTENTIONAL_BEHAVIOR` | 8 |
| `REFUTED_STATIC_SUSPICION` | 12 |
| `BLOCKED` | 12 |

件数はmatrixのシナリオ行数であり、同一修正境界にまとめられる複数routeを含む。

## 環境と再現性

各laneは同じソースSHAから独立worktreeを作り、characterization中の更新やrebaseを行っていない。

Python環境もworktreeとは別に分離した。

`present` はoptional packageを含む環境であり、`base` は同じcore環境をcloneして対象optional packageを実際にuninstallした環境である。

missing caseにmonkeypatchは使っていない。

両環境はPython 3.11.14、GWexpy 0.2.4、GWpy 4.0.2、NumPy 1.26.4を使用した。

NetCDF/Zarr/HDF5 laneはnetCDF4 1.7.4、xarray 2026.2.0、zarr 3.1.5、h5py 3.16.0を記録した。

instrument laneはnpTDMS 1.11.0、Astropy 6.1.7、MTH5 0.6.8を使用した。

ObsPy/audio laneはObsPy 1.5.0、SciPy 1.12.0、pydub 0.25.1、tinytag 2.3.0、`/usr/bin/ffmpeg` を記録した。

実行ファイル、optional packageの有無、対象versionは [environment-matrix.json](2026-09-27-public-io-cross-format-audit/environment-matrix.json) にある。

各findingにはfixture recipe、public entrypoint、native oracle、期待値、実測値、command ID、evidence file、source SHAをJSONで記録した。

`runtime-characterization-matrix.json` の `formats` は、常にcanonical format IDの配列である。複数形式にまたがるscenarioは複数要素を持つ。

再実行scriptは [repro](2026-09-27-public-io-cross-format-audit/repro/) に、JSONL観測とlane報告は [evidence](2026-09-27-public-io-cross-format-audit/evidence/) に保存した。

report bundleを含むcheckoutのrepository rootを作業ディレクトリにし、matrixのcommand IDに記載したコマンドを使う。

再実行コマンドは環境prefixを絶対パスで指定している。

別環境で再実行する場合はenvironment matrixのpackage versionとoptionalの有無を揃える。

## 確認した正当性不具合

### NetCDF

`NC-MATRIX-001`〜`NC-MATRIX-007` では、欠損cell、重複cell、key/indexの衝突、負index、sparse/out-of-range index、単位不一致を `TimeSeriesMatrix.read(format='nc')` が拒否せず、不完全または矛盾した2×2行列を返した。

欠損cellの判定はreaderがtopologyを拒否せず未代入領域を含むmatrixを返した点に基づく。

実行ごとに変わる `np.empty` の数値を再現条件にはしていない。

`NC-AXIS-001` と `NC-AXIS-002` では、legacy coordinate `[0,1,2,4,5]` と2秒gapを含むdatetime軸がregularizeされ、source axisと返却axisが一致しなかった。

`NC-MATRIX-UNIT-001` と `NC-MATRIX-UNIT-002` では、Matrix writeで `V` が空文字列になり、legacy readでもsource `units=V` が失われた。

`NC-DTYPE-001` はfix boundary確認中に追加したbaseline follow-upである。独立したxarray fixtureは先頭cellがint32、次cellが小数を含むfloat64で、sourceの値はxarrayから確認した。基準SHAのpublic `TimeSeriesMatrix.read(path, format='nc')` は行列dtypeをint32にし、後続cellを `[1, 2, 3]` へ黙って変換した。fixture、source/actual、dependency versionは `evidence/lane-a/heterogeneous_dtype.jsonl` にあり、再実行scriptは `repro/lane-a/heterogeneous_dtype_probe.py` にある。

### Zarr

`ZARR-DTYPE-001` では、int64値 `9007199254740993` が `9007199254740992` に丸められた。

`ZARR-DTYPE-002` では、complex値の虚部とphaseが失われ、実部だけのfloat64となった。

`ZARR-AXIS-001`、`ZARR-UNIT-001`、`ZARR-MULTISTORE-DTYPE-001`、`ZARR-UINT64-001` はfix boundary確認中に追加したbaseline follow-upである。public Matrix readは1 msずれたcell軸を受理して先頭時刻を全体へ割り当て、native multi-channel ZarrのV単位をdimensionlessとして返し、複数storeのint64および単独uint64の2**53超整数をfloat64へ丸めた。native Zarr fixture、source/actual、dependency versionsは `evidence/lane-a/zarr_axis_units_multistore.jsonl` にあり、再実行scriptは `repro/lane-a/zarr_axis_units_multistore_probe.py` にある。

`ZARR-AUTO-001` では、契約上auto identifyが公開された `.zarr` directoryを `TimeSeries.read(path)` が `IsADirectoryError` で拒否した。

`OPTIONAL-002` と `OPTIONAL-003` では、base環境のMatrix auto routeがmissing dependencyをgeneric `ValueError` に隠し、single-series auto Zarr routeはpresent/base両方で `IsADirectoryError` を返した。

### TDMS

`TDMS-TIME-001`〜`TDMS-TIME-005` では、`wf_increment` absent、0、NaN、正のInf、負のInfをpublic TimeSeries/Dict/Matrix routesが `dt=1.0` に置き換えた。

`TDMS-TIME-010` はfix boundary確認中に追加したbaseline follow-upである。`wf_increment=-0.1` のpublic TimeSeries/Dict routeは負のdtと下降するsample timesを返し、Matrix routeは読み込み後のalignmentで遅れて失敗した。fixtureと全route結果は `evidence/lane-b/tdms.jsonl` にある。

native npTDMSでは、absentはtime trackを生成できず、0/NaN/Infは各入力に応じた別のrelative coordinateとなる。

GWexpyの有限な1秒間隔はsourceにない情報を付加するため、axis corruptionまたはfabricated metadataと判定した。

### GBD

`GBD-HEADER-001` では必須の `Sample` 欄を欠くGL500 fixtureをreaderが警告なしで受け入れ、valid controlの100 msに対して `dt=1.0` を返した。

`GBD-HEADER-003` では `Order` 欄を欠くと `CH0` を作り、`CH1` と `Alarm` を失った。

`GBD-HEADER-004` では `Counts` 欄を欠くと、空のchannelを警告なしで返した。

`GBD-HEADER-005` では `$Amp` scaleを欠くとraw countをvoltとして返し、valid controlの `[1,-2,5] V` に対して `[2000,-4000,10000] V` となった。

これらの結論はGraphtec-authored GL500 specification sheetが対象にするfirmware 1.00–1.21のmalformed fixtureに限定した。

仕様書はメーカー著者文書のStudylib mirrorである。

[Graphtec公式FAQ](https://www.graphteccorp.com/logger_qa/qaqr_gl_029/) はGBDが同社機器のbinary formatであることを確認するが、byte layoutは裏付けない。

このため他Graphtec modelへ結論を一般化していない。

### HDF5 collection

`HDF5-TS-COLL-001`、`HDF5-FS-COLL-001`、`HDF5-SPEC-COLL-001`、`HDF5-HIST-COLL-001` では、manifestがA/B/Cを列挙したcollectionからBだけを壊すと、public Dict/List readは成功してBを落とし、List indexを詰めて返した。

manifestは独立にC/B/Aへ並べ替え、Dict key mapも書き換えたfixtureを使った。

C1 authoritative manifest routeとC2 manifestなしlegacy discovery routeを分離している。

C1は公開manifestに記載されたentryの欠損なのでsilent entry lossと判定した。

C2で同様のskipが起きるrouteは既存のtolerant discovery policyとして別のcompatibility behaviorに分類した。

skip処理はshared HDF5 helperではなくclass別readerにある。

### Audio metadata

`AUDIO-METADATA-001` では、tagged FLACのtitle `D audit title`、bitrate、durationがnative TinyTagとdirect readerでは得られたが、`TimeSeriesDict.read(..., extract_metadata=True)` 経由ではprovenanceが空になり、存在しないpathへの警告が出た。

registry routeはreaderに `FileIO` objectを渡すが、metadata extractorはpath文字列として扱う。

WAVでもavailableなbitrate/durationが失われる同じrouteを観測した。

metadataを保存するpublic docstringは [wav.py](../../../gwexpy/timeseries/io/wav.py#L56) と [audio.py](../../../gwexpy/timeseries/io/audio.py#L104) にある。

sample valuesとaxisはdirect/publicで一致した。

## 互換性、refuted、blocked

SDBとWINの `epoch=999` はdirect/public routeで受理後に無視された。

契約は両形式で `epoch_arg=none` としているため、正当性不具合とはせず `CONFIRMED_COMPATIBILITY_BEHAVIOR` にした。

今後、非対応kwargsを拒否するAPI policyを採る場合は別途契約を定める。

manifestなしHDF5 discoveryが壊れたentryをskipする挙動も、manifest-backed routeと混ぜずcompatibility behaviorとして残した。

ObsPyでは同じminiSEED trace IDを持つ連続区間、gap、overlapをpublic Dict readに渡したが、辞書構築前にObsPy mergeが実行された。

異なるsampling rateではmergeが明示的に例外を出し、重複IDの辞書上書き仮説は再現しなかった。

WAVとFLACのsample values、channel order、rate、relative `t0` はnative decoderと一致した。

blockedシナリオには、NetCDF v2で正当な異なるcell lengthまたはcell axisを作れなかった2件、TDMS missing absolute epochとroot DateTimeの意味が未規定な2件、TDMS unit importの契約不足、ATS truncated payloadのauthority不足、GBD surplus payloadのmalformed-input policy不足、他modelのGBD authority不足、Zarr dtype-only wideningの契約不足、HDF5 legacy fixture/layout不足が含まれる。

これらはstatusだけを `BLOCKED` とし、形式全体を未監査扱いにはしていない。

TDMS property semanticsは [NI TDMS internal structure](https://www.ni.com/en/support/documentation/supplemental/07/tdms-file-format-internal-structure.html#predefined-properties) と [npTDMS time-track documentation](https://nptdms.readthedocs.io/en/stable/reading.html) を参照した。

ObsPy merge behaviorは [ObsPy Stream API](https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.html) を参照した。

GL500 malformed-header findingsは [manufacturer-authored sheet mirror](https://studylib.net/doc/8138865/gl500-binary-data-file-format-specification-sheet) の対象範囲に限る。

## Issue draft

Issueは作成していない。

以下は具体的なfailure modeで整理したdraft titleと本文要点であり、severity scoreは付けていない。

| Draft title | 対象findingと本文要点 |
| --- | --- |
| Reject invalid or lossy NetCDF matrix cells | `NC-MATRIX-001`〜`NC-MATRIX-007`、`NC-DTYPE-001`。欠損/duplicate cell、row/column key-index矛盾、negative/sparse/out-of-range index、incompatible unitを検証し、異種dtype cellを損失変換せず保持または明示的に拒否する。 |
| Preserve or reject irregular legacy NetCDF time coordinates | `NC-AXIS-001`、`NC-AXIS-002`。numeric/datetime coordinate gapをmedian `dt` に置き換えず、明示軸で保持するか、regular `TimeSeries`で表現できない場合は明示的に拒否する。 |
| Preserve TimeSeriesMatrix units in NetCDF reads and writes | `NC-MATRIX-UNIT-001`、`NC-MATRIX-UNIT-002`。Matrix writer/readerで単位Vを保ち、mixed-unit cellsを黙って一単位に揃えない。 |
| Preserve Zarr matrix values, axes, and units | `ZARR-DTYPE-001`、`ZARR-DTYPE-002`、`ZARR-AXIS-001`、`ZARR-UNIT-001`、`ZARR-MULTISTORE-DTYPE-001`、`ZARR-UINT64-001`。large signed/unsigned integersとcomplex valuesを正確に保持し、cell軸とunitを保つ。対応できないdtypeやaxisは変換前に明示的に拒否する。 |
| Restore TimeSeries Zarr auto-read and missing-dependency errors | `ZARR-AUTO-001`、`OPTIONAL-002`、`OPTIONAL-003`。`.zarr` directoryのsingle-series auto routeを公開契約に合わせ、auto routeのmissing backend errorもImportErrorとして伝える。 |
| Reject TDMS files with absent or invalid waveform increments | `TDMS-TIME-001`〜`TDMS-TIME-005`、`TDMS-TIME-010`。absent、nonpositive、NaN、±Infの `wf_increment` を既定値へ置き換えず、public single/Dict/Matrix routesでfail-closedにする。 |
| Fail closed on malformed GL500 channel and scale headers | `GBD-HEADER-001`、`GBD-HEADER-003`〜`GBD-HEADER-005`。GL500 firmware 1.00–1.21のmalformed headersでdt、channel identity、sample count、scaleを捏造せず、明示的なerrorを返す。 |
| Fail HDF5 manifest collection reads when a listed entry is unreadable | `HDF5-TS-COLL-001`、`HDF5-FS-COLL-001`、`HDF5-SPEC-COLL-001`、`HDF5-HIST-COLL-001`。authoritative manifest-listed entryをskipせずcollection全体を失敗させる。manifestなしlegacy discovery policyは分けて扱う。 |
| Preserve audio metadata through public registry reads | `AUDIO-METADATA-001`。`extract_metadata=True` のregistry routeが`FileIO` objectをmetadata extractorへ渡すとpath検査に失敗し、title/bitrate/durationをprovenanceから落とす。public routeでnative/direct readerと同等にavailable metadataを保存する。 |

`GBD-COUNT-001`、`ATS-TRUNC-001`、`TDMS-TIME-007/008` はauthorityまたはcontract decisionが不足しているためissue draft化していない。

## conformance追加候補

`tests/io_conformance/contract.py` のblocking baselineは `gwf`、`hdf.ndscope`、`hdf5`、`csv`、`txt`、`wav` の6形式である。

このbaselineは25 canonical formats全件のruntime correctness coverageを意味しない。

将来のfix laneでは、次のindependent generator/scenarioを追加する。

1. NetCDF matrix generatorで欠損/duplicate/key-index衝突/negative/sparse/out-of-range cell、unit不一致、異種dtype cellを作り、拒否または値を損失なく保持する契約を検証する。
2. Legacy NetCDF numericとdatetime axis generatorでirregular intervalとgapを作り、sourceとreturned coordinatesを比較する。
3. NetCDF Matrix unit fixtureでsingle/Dict/Matrix write/readをxarrayで検証する。
4. Native Zarr fixtureでsigned/unsigned 64-bitとcomplex64/128の値、multi-store dtype、cell軸、unit、single/Dict/Matrixおよびauto routeを検証する。
5. Optional-missing jobでNC/Zarr explicit/autoのexception classを検証する。
6. TDMS waveform metadata fixtureを`ABSENT`、`INVALID_ZERO`、`INVALID_NEGATIVE`、`INVALID_NAN`、`INVALID_POSINF`、`INVALID_NEGINF`、`VALID`に分け、timestamp absent/root present/explicit overrideも独立にする。
7. GL500 versioned fixtureでvalid controlとmalformed `Sample`、`Order`、`Counts`、`$Amp` casesを分ける。他機種へ拡張する前に仕様authorityを揃える。
8. HDF5 collection fixtureでmanifest authorityとmanifestなしdiscoveryを別generatorにし、class × layoutの壊れたentryを検証する。
9. Audio metadata fixtureでtagged WAV/FLACを作り、direct/registry/auto routeとTinyTag present/missingを比較する。
10. ObsPy duplicate-ID fixtureでgap/overlap/contiguousとsample-rate mismatchを確認する。
11. SDB/WIN `epoch` kwargsの受理方針を契約に明記し、受理するならno-op挙動が意図的であることを記録する。

調査中に確認したdefectを「期待挙動」として恒久testへ先行登録してはいない。

次段階で契約とfix boundaryを決定してから、red reproductionをregression testへ変換する。
