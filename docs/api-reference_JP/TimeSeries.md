# TimeSeries

**継承元:** TimeSeriesInteropMixin, TimeSeriesAnalysisMixin, TimeSeriesResamplingMixin, TimeSeriesSignalMixin, TimeSeriesSpectralMixin, gwpy.timeseries.TimeSeries

`gwexpy` の全機能を備えた、拡張された TimeSeries クラスです。
GWpy の TimeSeries と完全な互換性を持ちつつ、以下の機能を追加・統合しています。

- **信号処理**: 解析信号 (analytic_signal), 周波数復調 (demodulate), 相互相関 (correlate) など。
- **スペクトル変換**: FFT, PSD, ASD, CWT (連続ウェーブレット変換), STLT (短時間ラプラス変換), DCT (離散コサイン変換) など。
- **データ分析**: 欠損値補完 (impute), 標準化 (standardize), 移動平均などのローリング統計 (rolling_*) など。
- **相互運用性**: PyTorch, TensorFlow, JAX, pandas, xarray, MNE, ObsPy などの他ライブラリへの変換。

## 主要メソッド

### `abs`

要素ごとの絶対値を計算します。

### `analytic_signal`

解析信号（ヒルベルト変換）を計算し、複素数形式の信号を返します。

### `asd`, `psd`

ASD（振幅スペクトル密度）または PSD（パワースペクトル密度）を計算します。

### `asfreq`

指定された時間間隔やルール（'1s' など）に基づいて、シリーズを補間せずに再サンプリング（リインデックス）します。補間が必要な場合は `resample` を使用します。

### `bandpass`, `highpass`, `lowpass`, `notch`

各種フィルタ（バンドパス、ハイパス、ローパス、ノッチフィルタ）を適用します。

### `cepstrum`

ケプストラムを計算します。

### `coherence`, `coherence_spectrogram`

他の信号とのコヒーレンス、またはコヒーレンス・スペクトログラムを計算します。

### `convolve`

FIR フィルタとの畳み込み（overlap-save 法）を行います。

### `correlate`

他の信号との相互相関（SNR タイムシリーズ）を計算します。

### `crop`

指定した GPS 時刻の範囲で信号を切り抜きます。

### `cwt`

連続ウェーブレット変換 (CWT) を計算します。

### `stlt`

短時間ラプラス変換 (STLT) を計算します。
(Time, Sigma, Frequency) の3次元データ (`LaplaceGram`) を返します。

### `dct`

離散コサイン変換 (DCT) を計算します。

### `demodulate`

指定した周波数で信号を復調し、振幅と位相のトレンドを取得します。

### `detrend`

トレンド除去（平均値や線形成分の除去）を行います。

### `fft`, `ifft`

高速フーリエ変換 (FFT) および逆 FFT を実行します。

### `emd`

経験的モード分解 (Empirical Mode Decomposition) を行い、固有モード関数 (IMF) を抽出します。

### `hht`

ヒルベルト・ファン変換 (Hilbert-Huang Transform) を計算します。分析結果として、各 IMF のヒルベルト・スペクトルを取得できます。

### `filter`, `filterba`

一般的なデジタルフィルタ、または [b, a] 係数によるフィルタを適用します。

### `find_peaks`

信号内のピークを検出します。(戻り値: `TimeSeries` オブジェクトとプロパティのタプル)

### `fit`

モデル関数へのフィッティングを行います（最小二乗法またはMCMC）。

### `arima`, `ar`, `ma`, `arma`

ARIMA（自己回帰移動平均）モデルなどの時系列モデルを適合させます。予測 (forecast) や残差分析が可能です。

### `gate`

指定した閾値に基づいて信号にゲート（ゼロ埋めやウィンドウ適用）をかけます。

### `heterodyne`

ヘテロダイン検波（デジタル復調）を行います。

### `lock_in`

ロックインアンプ方式による復調と平均化を行います。

### `mix_down`, `baseband`

信号の周波数シフト（複素復調）を行い、ベースバンド信号を取得します。

### `histogram`

データの値のヒストグラムを計算します。

### `hurst`, `local_hurst`

ハースト指数を計算します。時系列の長期記憶性や自己相関の強さを評価します。`local_hurst` は時間変化するハースト指数を推定します。

### `impute`

NaN などの欠損値を補完します。

### `interpolate`

指定したサンプリングレートへ補間します。

### `resample`

信号処理的な再サンプリング、または時間軸のリサンプリングを行います。

### `rms`, `std`, `mean`, `median`

各種統計量（実効値、標準偏差、平均、中央値）を計算します。

### `rolling_mean` (他 `rolling_*`)

移動窓を用いた統計計算（平均、中央値、標準偏差、最大値、最小値）を行います。

### `spectrogram`

スペクトログラムを計算します。

### `standardize`

データを標準化（z-score など）します。

### `to_torch`, `to_tensorflow`, `to_jax`, `to_cupy`

各種ディープラーニング・数値計算フレームワークのテンソル/配列に変換します。

### `to_pandas`, `to_xarray`, `to_polars`

データ分析ライブラリの DataFrame や DataArray に変換します。

### `to_mne`, `to_obspy`, `to_neo`, `to_simpeg`

脳磁計/脳波計 (MNE)、地震学 (ObsPy)、地球物理学 (SimPEG) などの専門分野のデータ型に変換します。

### `whiten`

信号をホワイトニング（白色化）します。

### `transfer_function`

他の信号との間の伝達関数（TF）を推定します。

### `write`, `read`

各種フォーマット（GWF, HDF5, WAV, CSV, ROOT など）でファイルの読み書きを行います。

### `zpk`

零点・極・ゲイン形式でフィルタを適用します。
