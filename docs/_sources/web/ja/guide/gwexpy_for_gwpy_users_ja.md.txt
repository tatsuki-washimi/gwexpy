# GWexpy 機能一覧 (GWpyユーザー向け)

本ドキュメントでは、GWexpy で GWpy に対して拡張・追加された機能を網羅的にリストアップします。
ここに含まれる機能は、`gwexpy` をインポートすることで利用可能になります。

## 1. GWpy既存クラスの機能拡張 (軽微なもの)

`TimeSeries`, `FrequencySeries`, `Spectrogram` 等の基本クラスに対する、利便性向上のためのメソッド追加や機能強化です。

### 基本メソッドの拡張 (TimeSeries / FrequencySeries / Spectrogram)
*   **位相・角度**:
    *   `.phase(unwrap=False)`: 位相の計算
    *   `.angle(unwrap=False)`: `.phase()` のエイリアス
    *   `.radian(unwrap=False)`: ラジアン単位での位相
    *   `.degree(unwrap=False)`: 度数法での位相
    *   `np.unwrap` のサポート
*   **リサンプリング・補間・編集**:
    *   `.interpolate(new_times, method='linear')`: データの補間
    *   `.resample(rate)`: `scipy` または `obspy` (Lanczos) バックエンドを用いたリサンプリングの強化
    *   `.decimate(factor)`: 間引き
    *   `.tail(n)`: 末尾 `n` サンプルの抽出
    *   `.crop(start, end)`: (`gwexpy.time.to_gps` による柔軟な時刻指定をサポート)
    *   `.append(other)`: 結合処理の強化
*   **前処理 (Preprocessing)**:
    *   `.impute(method=...)`: 欠損値補完 (線形補間, ffill, bfill)。`max_gap` 制約なども指定可能。
    *   `.standardize()`: 標準化 (Z-score normalization)
    *   `.whiten(method='pca'|'zca')`: 白色化 (Whitening)。`return_model=True` でモデルを取得可能。
*   **ピーク検出**:
    *   `.find_peaks(...)`: 信号内のピーク検出 (`scipy.signal.find_peaks` ラッパー)

### GPS Time 関連
*   GPS時刻のベクトル演算やフォーマット変換をサポートするユーティリティ群 (`gwexpy.time`, `gwexpy.plot.gps` 等で利用可能)。

### 描画関連 (Plotting)
*   **日時軸**: プロット時の横軸を自動的に日時 (datetime) 形式にフォーマットする機能。
*   **一括描画**: `Spectrogram.plot_summary()` などによる、スペクトログラムと時間平均ASDの同時プロット。
*   **地図・全天図**: `gwexpy.plot.GeoMap`, `gwexpy.plot.SkyMap` による地図上へのデータプロット機能。
*   **PairPlot**: `gwexpy.plot.PairPlot` による多変量データのペアプロット（散布図行列/相関図）。

### コレクションクラスの拡張 (TimeSeriesList / TimeSeriesDict)
GWpyに存在する `TimeSeriesList`, `TimeSeriesDict` に対しても、一括処理機能を追加拡張しています。
*   一括信号処理: `.fft_time_all()`, `.resample_all(rate)`, `.filter_all(*args)`
*   一括スライス: `.sel_all(...)`, `.isel_all(...)`
*   算術演算のブロードキャスト

## 2. 新規クラスの導入

多チャンネルデータ、行列データ、物理場などを扱うための新しいデータ構造です。

### 新規コレクション (New Collections)
GWpyには標準実装されていない、周波数・時間周波数データのためのコレクションクラスです。GWpyの `TimeSeries` に対する `TimeSeriesList`, `TimeSeriesDict` の立ち位置になります。
*   `FrequencySeriesList`, `FrequencySeriesDict`
*   `SpectrogramList`, `SpectrogramDict`

### マトリックス (SeriesMatrix)
データを「行列」の要素として持つ多次元データ構造です (Row, Col, Axis...)。
*   `TimeSeriesMatrix`: (Rows, Cols, Time)
*   `FrequencySeriesMatrix`: (Rows, Cols, Frequency)
*   `SpectrogramMatrix`: (Rows, Cols, Time, Frequency)

**特徴・機能**:
*   **行列演算**: 要素ごとの行列積 (`@`)、逆行列 (`.inv()`)、行列式 (`.det()`)、トレース (`.trace()`)、シュア補行列 (`.schur()`)。MIMOシステムの伝達関数解析などに強力。
*   **統計処理**: 行・列方向の平均・分散演算 (`.mean(axis)`, `.std(axis)` 等) により、センサーアレイ全体の統計量を一括計算。
*   **バッチ処理**: 各要素に対する信号処理（フィルタリング、リサンプリング、白色化など）の一括適用。

### 物理フィールド (Fields)
4次元 (時間/周波数 + 3次元空間/波数) の物理場を扱うクラスです。
*   `ScalarField` / `VectorField` / `TensorField`

**特徴・機能**:
*   **ドメイン管理**: 時間/周波数軸 (`axis0`) と空間/波数軸 (`space_domain`) の状態を管理し、相互変換 (`.fft_time`, `.fft_space`) をサポート。
*   **空間抽出**: `.extract_points()` により任意の空間座標での時系列データを抽出・補間。
*   **ベクトル・テンソル解析**: 内積、外積、発散、回転などの物理演算と、座標変換（デカルト座標系など）。
*   **シミュレーション**: `gwexpy.noise.field` と連携したノイズ場の生成（等方性ノイズ、平面波など）。

### その他
*   `BifrequencyMap`: 2つの周波数軸を持つマップデータ (STLTGram等)。

## 3. 高度な解析メソッド (単チャンネル)

### ノイズシミュレーション (`gwexpy.noise`)
*   `from_asd(asd, ...)`: ASD (振幅スペクトル密度) からの時系列生成
*   `colored_noise(psd, ...)`: 有色ノイズの生成
*   `field.simulate(...)`: ノイズ場の生成
*   `pygwinc` 連携: `gwexpy.noise.gwinc_` を通じた干渉計ノイズバジェットモデルの利用。

### 時系列解析 (TimeSeries)
*   `.hilbert()`: ヒルベルト変換（解析信号）
*   `.envelope()`: 包絡線 (`abs(hilbert)`)
*   `.demodulate(f)`: ロックインアンプ・復調
*   `.arima(order, ...)`: AR/MA/ARMA/ARIMAモデル解析 (`statsmodels`, `pmdarima` ラッパー)
*   `.ar(p)`, `.ma(q)`, `.arma(p, q)`: AR/MA/ARMAのショートカット
*   `.hurst()`: ハースト指数の計算
*   `.skewness()`: 歪度 (Skewness)
*   `.kurtosis()`: 尖度 (Kurtosis)

### 周波数スペクトル解析 (FrequencySeries)
*   `.differentiate()`, `.integrate()`: 周波数ドメインでの微分・積分
*   `.differentiate_time()`, `.integrate_time()`: 周波数ドメインでの時間微分・積分（Displacement ↔ Velocity ↔ Acceleration 変換等）
*   `.group_delay()`: 群遅延の計算

### 時間周波数解析 (Spectrogram / Special Transforms)
GWpyのスペクトログラムとQ変換に加えて、様々な時間周波数解析手法を提供します。
*   `hht()`: ヒルベルト・ファン変換 (HHT)。EMD (Empirical Mode Decomposition) によるIMF分解と瞬時周波数解析。
*   `stlt()`: 短時間ラプラス変換 (Short-Time Laplace Transform)。減衰定数 $\sigma$ を考慮した解析。
*   `cepstrum()`: ケプストラム解析 (倒周波数)。
*   `cwt()`: 連続ウェーブレット変換 (`pywt` ラッパー)。

### 統計的推定・フィッティング
*   **Fitting (`gwexpy.fitting`)**:
    *   `FitResult`, `Model`: iminuitによるchi-square Fitting (実数、複素数) とMCMC解析
    *   `TimeSeries`, `FrequencySeries` にも `.fit()` 等のメソッドがMixされています。
*   **Bootstrap**:
    *   `bootstrap_spectrogram`: スペクトログラムのブートストラップ推定

## 4. 高度な解析メソッド (多チャンネル)

### 成分分析 (`gwexpy.timeseries.decomposition`)
*   `PCA`: 主成分分析
*   `ICA`: 独立成分分析
*   `ZCA`: 白色化 (`.whiten(method='zca')` として利用)

### 相関・統計解析
*   **相関指標**:
    *   `dCor`: 距離相関 (Distance Correlation)
    *   `MIC`: 最大情報係数 (Maximal Information Coefficient)
    *   `Pearson`, `Kendall`: 基本的な相関係数
*   **Bruco**:
    *   多チャンネルコヒーレンス探索によるノイズ源特定ツール (`gwexpy.analysis.bruco`)

### システム解析・制御
*   **MIMO / 状態空間モデル**:
    *   `python-control` 連携: `TimeSeries.from_control(response)`, `to_control_frd` などによる制御系設計ライブラリとの相互変換。
    *   行列クラス (`SeriesMatrix`) を用いたMIMO伝達関数・時系列解析
*   **雑音注入試験解析**:
    *   `gwexpy.analysis.response`: 伝達関数推定や結合係数算出のための実験データ解析ツール
*   **行列演算**:
    *   逆行列 (`.inv()`)、行列式 (`.det()`)、トレース (`.trace()`)、シュア補行列 (`.schur()`) などの基本操作。

## 5. 外部ツール連携 (Interop)

外部ライブラリとのオブジェクト相互変換機能です (`.to_*()` / `.from_*()`)。

*   **Deep Learning / Array Libraries**:
    *   `PyTorch` (`Tensor`, `Dataset`/`DataLoader`): `to_torch`, `to_torch_dataset`等
    *   `TensorFlow` (`Tensor`): `to_tf`
    *   `JAX` (`Array`): `to_jax`
    *   `CuPy` (`ndarray`): `to_cupy`
    *   `Dask` (`dask.array`): `to_dask`
    *   `Zarr`: `to_zarr` / `from_zarr`
*   **Data Structures**:
    *   `pandas` (`Series`/`DataFrame`): `to_pandas`, `to_pandas_dataframe`
    *   `xarray` (`DataArray`): `to_xarray`
    *   `polars` (`Series`/`DataFrame`): `to_polars_series`
    *   `SQLite`, `JSON`: 軽量な保存・読み込み
*   **Domain Specific**:
    *   `Obspy` (Seismology): `Trace`, `Stream`
    *   `MNE` (EEG/MEG): `Raw`, `RawArray` (`to_mne`)
    *   `Neo` (Electrophysiology): `AnalogSignal` (`to_neo`)
    *   `Librosa` / `Pydub` (Audio): 音声データ変換
    *   `SimPEG` (Geophysics): `Data` オブジェクト
    *   `python-control`: `FrequencyResponseData` (FRD) 等
    *   `ROOT` (CERN): `TGraph`, `TH1D`, `TH2D`, `TMultiGraph`
    *   `Specutils`: `Spectrum1D` (`to_specutils`)
    *   `Pyspeckit`: `Spectrum` (`to_pyspeckit`)
    *   `Astropy`: `TimeSeries` (`to_astropy_timeseries`)
    *   `Quantities`: `Quantity` (`to_quantity`)

### データフォーマットI/O拡張
GWpyが標準サポートする形式に加え、以下に対応（または拡張）しています。
*   `ATS` (Metronix MTデータ)
*   `GBD` (GRAPHTEC データロガー)
    *   アナログCHはヘッダのアンプレンジ（例: `20mV` 等）を用いて `count -> V` に換算します。
    *   `Alarm` / `AlarmOut` / `Pulse*` / `Logic*` はデジタル/ステータスとして扱い、値を 0/1 に正規化し unit を dimensionless にします。
    *   必要に応じて `TimeSeriesDict.read(..., format="gbd", digital_channels=[...])` でデジタル扱いするチャンネル名を上書きできます。
*   `TDMS` (LabVIEW / NI)
*   `MiniSEED`, `SAC`, `GSE2` (地震波データ, Obspy連携)
*   `WIN` (地震波データ)
*   `DTTXML` (LIGO Diagnostic Test Tools)
*   `SDB` (気象計 Davis データ)
*   `Midas` (PSI/DAQフォーマット)
*   `Parquet`, `Feather`, `Pickle` (pandas連携)
*   `WAV` (Audio, 拡張サポート)
*   `ROOT` file (.root)

## 6. GUI アプリケーション

*   **pyaggui (`gwexpy.gui.pyaggui`)**:
    *   DTT diaggui (C++, ROOT) を模したpyqt5ベースのGUIアプリケーション。
