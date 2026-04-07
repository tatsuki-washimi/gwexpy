# GWexpy Class List (Categorized Usage)

このドキュメントは、`gwexpy/` ソースコード全体から抽出したクラスを、「公開クラス（ユーザー利用想定）」と「内部・基底クラス（実装用）」に分類したものです。

## 1. 公開クラス (Public Classes)
ユーザーが直接 `import` してインスタンス化、あるいは主要な解析フローで明示的に型を意識するクラスです。

### 1.1 主要な解析・アルゴリズム (Main Analysis)
- `analysis/bruco: Bruco` (干渉計コヒーレンス解析)
- `analysis/coupling: CouplingFunctionAnalysis` (結合関数解析)
- `analysis/response: ResponseFunctionAnalysis` (応答関数解析)
- `analysis/stats: SpectralStats` (スペクトル統計)
- `analysis/threshold: PercentileThreshold`, `RatioThreshold`, `SigmaThreshold` (閾値判定戦略)
- `fitting/core: ComplexLeastSquares`, `RealLeastSquares`, `Fitter` (フィッティング)
- `fitting/gls: GLS`, `GeneralizedLeastSquares` (一般化最小二乗法)
- `fitting/models: Polynomial` (多項式モデル)

### 1.2 コアデータ構造 (Core Data Containers)
- `timeseries/timeseries: TimeSeries` (時系列データ)
- `timeseries/matrix: TimeSeriesMatrix` (マルチチャネル時系列)
- `timeseries/collections: TimeSeriesDict`, `TimeSeriesList`
- `frequencyseries/frequencyseries: FrequencySeries` (周波数領域データ)
- `frequencyseries/matrix: FrequencySeriesMatrix` (スペクトル行列)
- `frequencyseries/collections: FrequencySeriesDict`, `FrequencySeriesList`
- `spectrogram/spectrogram: Spectrogram` (スペクトログラム)
- `spectrogram/matrix: SpectrogramMatrix` (スペクトログラム行列)
- `spectrogram/collections: SpectrogramDict`, `SpectrogramList`
- `fields/scalar: ScalarField` (スカラー場)
- `fields/vector: VectorField` (ベクトル場)
- `fields/tensor: TensorField` (テンソル場)
- `fields/collections: FieldDict`, `FieldList`
- `histogram/histogram: Histogram` (ヒストグラム)
- `histogram/collections: HistogramDict`, `HistogramList`
- `table/segment_table: SegmentTable` (セグメント管理テーブル)

### 1.3 信号処理・パイプライン (Signal Processing)
- `timeseries/pipeline: Pipeline` (処理パイプライン)
- `timeseries/pipeline: WhitenTransform`, `StandardizeTransform`, `PCATransform`, `ICATransform`, `ImputeTransform`, `Transform` (変換器)
- `signal/preprocessing/ml: MLPreprocessor` (機械学習前処理)
- `signal/preprocessing/standardization: StandardizationModel`
- `signal/preprocessing/whitening: WhiteningModel`
- `numerics/scaling: AutoScaler` (スケーリング)

### 1.4 描画 (Visualization)
- `plot/plot: Plot` (基本プロット)
- `plot/field: FieldPlot` (フィールド描画)
- `plot/geomap: GeoMap` (地図プロット)
- `plot/skymap: SkyMap` (全天マップ)
- `plot/pairplot: PairPlot` (ペアプロット)

---

## 2. 内部実装・基底クラス (Internal / Base Classes)
主にライブラリ内部、または新機能開発時の基底として使用されるクラスです。

### 2.1 抽象・基底クラス (Abstract & Base)
- `types/array: Array`, `types/array2d: Array2D`, `types/array3d: Array3D`, `types/array4d: Array4D`
- `types/series: Series` (系列データの基底)
- `types/seriesmatrix_base: SeriesMatrix` (行列データの基底)
- `timeseries/_core: TimeSeriesCore`
- `frequencyseries/frequencyseries: SeriesType`
- `fields/base: FieldBase`
- `fitting/models: Model` (フィッティングモデル基底)
- `analysis/threshold: ThresholdStrategy` (ストラテジー基底)

### 2.2 Mixin (機能拡張用)
- `*Mixin` (例: `FittingMixin`, `TimeSeriesAnalysisMixin`, `PlotMixin`, `StatisticalMethodsMixin`, `SeriesMatrixCoreMixin` 等の全 Mixin クラス)
- 構造・IO・数学演算用の Mixin 群 (`SeriesMatrixIndexingMixin`, `SeriesMatrixIOMixin`, `SeriesMatrixMathMixin` 等)

### 2.3 プロトコル・インターフェース (Protocols)
- `types/mixin/_protocols: Supports*`, `Has*` (例: `SupportsTimeSeries`, `HasSeriesData` 等)
- `histogram/_typing: HistogramProtocol`
- `types/typing: XIndex`, `MetaDataLike`, `MetaDataDictLike`
- `types/mixin/_collection_mixin: _DictLike`, `_ListLike`

### 2.4 GUI 関連（内部実装）
- `gui/` 配下の全クラス (ユーザーの直接利用を想定しない)
- GUI スレッド・ワーカー: `NDSThread`, `AudioThread`, `SimulationThread`, `ChannelListWorker`
- UI コンポーネント: `MainWindow`, `ChannelBrowserDialog`, `ExcitationManager`, `GraphPanel`, `PlotRenderer`
- 解析・信号生成: `Engine`, `SignalGenerator`, `GeneratorParams`, `PayloadPacket`
- 内部バッファ・キャッシュ: `DataBuffer`, `DataBufferDict`, `ChannelListCache`, `NDSDataCache`, `SpectralAccumulator`

### 2.5 その他内部実装部品 (Miscellaneous Internals)
- `types/metadata: MetaData`, `MetaDataDict`, `MetaDataMatrix` (内部メタデータ管理)
- `table/segment_table: RowProxy`
- `table/segment_cell: SegmentCell`
- `types/axis: AxisDescriptor`
- `timeseries/utils: AxisInfo`, `FreqAxisInfo`, `SeriesType`
- `interop/_registry: ConverterRegistry`
- `timeseries/_typing: TimeSeriesAttrs`

---

## 3. 特殊・境界クラス (Context / Result Objects)
ユーザーに返されるオブジェクトですが、直接インスタンス化することは稀なものです。

### 3.1 解析結果オブジェクト (Analysis Results)
- `analysis/bruco: BrucoResult`
- `analysis/coupling_result: CouplingResult`, `CouplingResultCollection`
- `analysis/response: ResponseFunctionResult`
- `timeseries/arima: ArimaResult`, `ArimaForecastResult`
- `timeseries/decomposition: ICAResult`, `PCAResult`
- `timeseries/hurst: HurstResult`
- `timeseries/_statistics: GrangerResult`
- `statistics/gauch: GauChResult`

### 3.2 構成・設定オブジェクト (Configuration)
- `timeseries/io/csv_config: CSVFormatConfig`, `ColumnSpec`
- `timeseries/io/gbd: GBDHeader`
- `io/dttxml_common: ChannelInfo`

---

## 備考 (Note)
- `_` で始まるクラス（例: `_FrequencySeriesMatrixLike`, `_LocAccessor` 等）は、すべて内部実装用として Section 2 に分類されます。
- クラス名に `Protocol` や `Interface` を含むものは、 Section 2.3 に分類されています。
