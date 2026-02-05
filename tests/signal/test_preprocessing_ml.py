"""Tests for gwexpy.signal.preprocessing.ml module."""

import astropy.units as u
import numpy as np
import pytest

from gwexpy.signal.preprocessing import MLPreprocessor
from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix


class TestMLPreprocessorSplit:
    """Tests for MLPreprocessor.split() method."""

    def test_split_integer_seconds(self):
        """分割点が整数秒になることを確認."""
        # ダミーデータ作成
        sample_rate = 4096  # Hz
        duration = 10  # 秒
        n_samples = sample_rate * duration
        n_channels = 3

        X = TimeSeriesMatrix(
            np.random.randn(n_channels, n_samples), t0=0, dt=1 / sample_rate, unit=""
        )
        y = TimeSeries(np.random.randn(n_samples), t0=0, dt=1 / sample_rate, unit="")

        # 33%を検証用に
        preprocessor = MLPreprocessor(sample_rate=sample_rate, valid_frac=0.33)
        X_train, y_train, X_valid, y_valid = preprocessor.split(X, y)

        # 検証データのサイズが整数秒であることを確認
        valid_duration_sec = len(y_valid) / sample_rate
        assert valid_duration_sec == int(valid_duration_sec), (
            f"Validation duration {valid_duration_sec} is not an integer number of seconds"
        )

        # 訓練データと検証データの合計がもとのデータと一致することを確認
        assert len(y_train) + len(y_valid) <= len(y)

    def test_split_zero_valid_frac(self):
        """valid_frac=0.0の場合、空の検証データを返すことを確認."""
        # ダミーデータ作成
        sample_rate = 4096
        n_samples = 8192
        n_channels = 2

        X = TimeSeriesMatrix(
            np.random.randn(n_channels, n_samples), t0=0, dt=1 / sample_rate, unit=""
        )
        y = TimeSeries(np.random.randn(n_samples), t0=0, dt=1 / sample_rate, unit="")

        # valid_frac=0.0で分割
        preprocessor = MLPreprocessor(sample_rate=sample_rate, valid_frac=0.0)
        X_train, y_train, X_valid, y_valid = preprocessor.split(X, y)

        # 訓練データは元のデータと同じサンプル数
        assert X_train.shape[-1] == X.shape[-1]
        assert len(y_train) == len(y)

        # 検証データのサンプル数は0
        assert X_valid.shape[-1] == 0
        assert len(y_valid) == 0

    def test_split_preserves_metadata(self):
        """分割後にメタデータ（t0, dt）が保持されることを確認."""
        sample_rate = 4096
        n_samples = 8192
        n_channels = 2
        t0 = 1000.0  # GPS time

        X = TimeSeriesMatrix(
            np.random.randn(n_channels, n_samples),
            t0=t0,
            dt=1 / sample_rate,
            unit="",
        )
        y = TimeSeries(
            np.random.randn(n_samples), t0=t0, dt=1 / sample_rate, unit=""
        )

        preprocessor = MLPreprocessor(sample_rate=sample_rate, valid_frac=0.2)
        X_train, y_train, X_valid, y_valid = preprocessor.split(X, y)

        # dtが保持されていることを確認（訓練データのみ、検証データは空の可能性があるため）
        assert X_train.dt == X.dt
        assert y_train.dt == y.dt
        # 検証データが空でない場合のみdtを確認
        if X_valid.shape[-1] > 0:
            assert X_valid.dt == X.dt
            assert y_valid.dt == y.dt

        # t0が適切に設定されていることを確認
        assert X_train.t0 == X.t0
        assert y_train.t0 == y.t0
        # 検証データのt0は訓練データの終了時刻と一致
        if X_valid.shape[-1] > 0:
            expected_valid_t0 = X_train.t0 + X_train.shape[-1] * X_train.dt
            assert abs(float(X_valid.t0 - expected_valid_t0)) < 1e-6
            assert abs(float(y_valid.t0 - expected_valid_t0)) < 1e-6


class TestMLPreprocessorHelpers:
    """Tests for MLPreprocessor helper methods."""

    def test_get_sample_rate_hz_from_quantity(self):
        """Quantity型のsample_rateをfloatに変換できることを確認."""
        sample_rate_quantity = 4096 * u.Hz
        preprocessor = MLPreprocessor(sample_rate=sample_rate_quantity)
        sample_rate_hz = preprocessor._get_sample_rate_hz()

        assert isinstance(sample_rate_hz, float)
        assert sample_rate_hz == 4096.0

    def test_get_sample_rate_hz_from_float(self):
        """float型のsample_rateをそのまま返すことを確認."""
        sample_rate_float = 4096.0
        preprocessor = MLPreprocessor(sample_rate=sample_rate_float)
        sample_rate_hz = preprocessor._get_sample_rate_hz()

        assert isinstance(sample_rate_hz, float)
        assert sample_rate_hz == 4096.0

    def test_extract_value_from_timeseries(self):
        """TimeSeriesからndarrayを取得できることを確認."""
        data = np.array([1.0, 2.0, 3.0])
        ts = TimeSeries(data, t0=0, dt=1, unit="")
        preprocessor = MLPreprocessor(sample_rate=1.0)

        extracted = preprocessor._extract_value(ts)
        np.testing.assert_array_equal(extracted, data)

    def test_reconstruct_timeseries_preserves_metadata(self):
        """TimeSeriesの再構築でメタデータが保持されることを確認."""
        data = np.array([1.0, 2.0, 3.0])
        ts = TimeSeries(data, t0=1000, dt=0.5, unit="m", name="test_channel")
        preprocessor = MLPreprocessor(sample_rate=2.0)

        new_data = np.array([4.0, 5.0, 6.0])
        reconstructed = preprocessor._reconstruct_timeseries(
            new_data, ts, unit=u.dimensionless_unscaled
        )

        assert reconstructed.t0 == ts.t0
        assert reconstructed.dt == ts.dt
        assert reconstructed.unit == u.dimensionless_unscaled
        assert reconstructed.name == ts.name
        np.testing.assert_array_equal(reconstructed.value, new_data)


class TestMLPreprocessorFitTransform:
    """Tests for MLPreprocessor fit() and transform() methods."""

    def test_fit_transform_basic_workflow(self):
        """基本的なfit→transformワークフローが動作することを確認."""
        # ダミーデータ作成
        sample_rate = 4096
        n_samples = 8192
        n_channels = 3

        X = TimeSeriesMatrix(
            np.random.randn(n_channels, n_samples), t0=0, dt=1 / sample_rate, unit="m"
        )
        y = TimeSeries(np.random.randn(n_samples), t0=0, dt=1 / sample_rate, unit="m")

        # 前処理パイプライン（フィルタなし）
        preprocessor = MLPreprocessor(sample_rate=sample_rate)
        preprocessor.fit(X, y)

        # transform実行
        X_proc, y_proc = preprocessor.transform(X, y)

        # 出力の形状が保持されていることを確認
        assert X_proc.shape == X.shape
        assert len(y_proc) == len(y)

        # 出力単位がdimensionless_unscaledであることを確認
        # TimeSeriesMatrixはunits（複数形）、TimeSeriesはunit（単数形）
        assert all(unit == u.dimensionless_unscaled for unit in X_proc.units.flat)
        assert y_proc.unit == u.dimensionless_unscaled

    def test_fit_transform_with_bandpass(self):
        """バンドパスフィルタ付きでfit→transformが動作することを確認."""
        # ダミーデータ作成
        sample_rate = 4096
        n_samples = 16384  # 4秒分
        n_channels = 2

        X = TimeSeriesMatrix(
            np.random.randn(n_channels, n_samples), t0=0, dt=1 / sample_rate, unit=""
        )
        y = TimeSeries(np.random.randn(n_samples), t0=0, dt=1 / sample_rate, unit="")

        # バンドパスフィルタ付き前処理
        preprocessor = MLPreprocessor(
            sample_rate=sample_rate,
            freq_low=[55.0],
            freq_high=[65.0],
            filt_order=4,
        )
        preprocessor.fit(X, y)

        # transform実行
        X_proc, y_proc = preprocessor.transform(X, y)

        # フィルタ係数が設定されていることを確認
        assert preprocessor.filter_coeffs_ is not None
        assert len(preprocessor.filter_coeffs_) == 1

        # 出力が適切に処理されていることを確認
        assert all(unit == u.dimensionless_unscaled for unit in X_proc.units.flat)
        assert y_proc.unit == u.dimensionless_unscaled

    def test_transform_before_fit_raises_error(self):
        """fit()前にtransform()を呼ぶとエラーが発生することを確認."""
        X = TimeSeriesMatrix(
            np.random.randn(2, 1000), t0=0, dt=0.001, unit=""
        )

        preprocessor = MLPreprocessor(sample_rate=1000)

        with pytest.raises(RuntimeError, match="fit\\(\\)を先に呼び出してください"):
            preprocessor.transform(X)

    def test_output_is_standardized(self):
        """出力が標準化されていることを確認（平均0、標準偏差1に近い）."""
        # ダミーデータ作成（平均と標準偏差が異なる）
        sample_rate = 1000
        n_samples = 10000
        n_channels = 2

        X_data = np.random.randn(n_channels, n_samples) * 10 + 50  # 平均50、std約10
        X = TimeSeriesMatrix(X_data, t0=0, dt=1 / sample_rate, unit="")

        y_data = np.random.randn(n_samples) * 5 + 20  # 平均20、std約5
        y = TimeSeries(y_data, t0=0, dt=1 / sample_rate, unit="")

        # 前処理
        preprocessor = MLPreprocessor(sample_rate=sample_rate)
        preprocessor.fit(X, y)
        X_proc, y_proc = preprocessor.transform(X, y)

        # 標準化されているか確認（平均0、std1に近い）
        np.testing.assert_allclose(
            np.mean(X_proc.value, axis=-1), 0.0, atol=1e-10
        )
        np.testing.assert_allclose(
            np.std(X_proc.value, axis=-1, ddof=0), 1.0, atol=1e-10
        )
        np.testing.assert_allclose(np.mean(y_proc.value), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.std(y_proc.value, ddof=0), 1.0, atol=1e-10)


class TestMLPreprocessorDeepCleanCompatibility:
    """DeepClean参考実装との互換性テスト."""

    def test_split_logic_matches_deepclean(self):
        """分割ロジックがDeepClean参考実装と一致することを確認."""
        # DeepClean参考実装の分割ロジック（data.py:174-176）:
        # valid_size = int(valid_frac * len(y))
        # valid_length = int(valid_size / sample_rate)  # 整数秒に丸め
        # valid_size = int(valid_length * sample_rate)   # サンプル数に戻す

        sample_rate = 4096
        duration = 10
        n_samples = sample_rate * duration
        valid_frac = 0.33

        X = TimeSeriesMatrix(
            np.random.randn(2, n_samples), t0=0, dt=1 / sample_rate, unit=""
        )
        y = TimeSeries(np.random.randn(n_samples), t0=0, dt=1 / sample_rate, unit="")

        # DeepClean参考実装のロジックを直接実装
        valid_size_deepclean = int(valid_frac * len(y))
        valid_length_deepclean = int(valid_size_deepclean / sample_rate)
        valid_size_deepclean = int(valid_length_deepclean * sample_rate)
        train_size_deepclean = len(y) - valid_size_deepclean

        # MLPreprocessorの分割
        preprocessor = MLPreprocessor(sample_rate=sample_rate, valid_frac=valid_frac)
        X_train, y_train, X_valid, y_valid = preprocessor.split(X, y)

        # サイズが一致することを確認
        assert len(y_train) == train_size_deepclean
        assert len(y_valid) == valid_size_deepclean

    def test_bandpass_filter_matches_deepclean(self):
        """バンドパスフィルタがDeepClean参考実装と一致することを確認."""
        from scipy.signal import butter, sosfiltfilt

        # テストデータ作成
        sample_rate = 4096
        n_samples = 16384
        freq_low = [55.0]
        freq_high = [65.0]
        filt_order = 8

        # 疑似ランダムデータ（シード固定で再現性確保）
        np.random.seed(42)
        y_data = np.random.randn(n_samples)
        y = TimeSeries(y_data, t0=0, dt=1 / sample_rate, unit="")

        # DeepClean参考実装のフィルタリング（filt.py:126-130）
        coeffs_deepclean = []
        for f_low, f_high in zip(freq_low, freq_high):
            sos = butter(filt_order, [f_low, f_high], btype="bandpass", fs=sample_rate, output="sos")
            coeffs_deepclean.append(sos)
        y_filt_deepclean = np.zeros_like(y_data)
        for coeff in coeffs_deepclean:
            y_filt_deepclean += sosfiltfilt(coeff, y_data, axis=-1)

        # MLPreprocessorのフィルタリング
        preprocessor = MLPreprocessor(
            sample_rate=sample_rate,
            freq_low=freq_low,
            freq_high=freq_high,
            filt_order=filt_order,
        )
        # ダミーデータでfit（フィルタ係数を設計）
        X_dummy = TimeSeriesMatrix(np.random.randn(2, n_samples), t0=0, dt=1 / sample_rate, unit="")
        preprocessor.fit(X_dummy, y)
        y_filt_mlprep = preprocessor._apply_bandpass(y)

        # フィルタ結果が一致することを確認（相対誤差1e-5、絶対誤差1e-6）
        np.testing.assert_allclose(
            y_filt_mlprep.value, y_filt_deepclean, atol=1e-6, rtol=1e-5
        )

    def test_standardization_matches_deepclean(self):
        """標準化がDeepClean参考実装と一致することを確認."""
        # DeepClean参考実装では、Scalerクラスがz-score標準化を行う
        # std[std == 0] = 1 の処理も含む（data.py:195-196）

        sample_rate = 4096
        n_samples = 8192
        n_channels = 3

        # テストデータ作成（1チャンネルは定数）
        np.random.seed(42)
        X_data = np.random.randn(n_channels, n_samples) * 10 + 50
        X_data[1, :] = 100.0  # 定数チャンネル（std=0）
        X = TimeSeriesMatrix(X_data, t0=0, dt=1 / sample_rate, unit="")

        # MLPreprocessorの標準化
        preprocessor = MLPreprocessor(sample_rate=sample_rate, standardization_method="zscore")
        preprocessor.fit(X, y=None)
        X_proc = preprocessor.transform(X, y=None)

        # 各チャンネルの統計量を確認
        for i in range(n_channels):
            if i == 1:
                # 定数チャンネルは標準化しても定数のまま（std=1で割る）
                # 期待値: (100 - 100) / 1 = 0
                np.testing.assert_allclose(X_proc.value[i], 0.0, atol=1e-10)
            else:
                # 通常チャンネルは平均0、std1に標準化される
                np.testing.assert_allclose(np.mean(X_proc.value[i]), 0.0, atol=1e-10)
                np.testing.assert_allclose(np.std(X_proc.value[i], ddof=0), 1.0, atol=1e-10)


class TestMLPreprocessorEdgeCases:
    """エッジケースのテスト."""

    def test_single_channel_X(self):
        """単一チャンネルの参照データで動作することを確認."""
        sample_rate = 1000
        n_samples = 5000
        n_channels = 1

        X = TimeSeriesMatrix(
            np.random.randn(n_channels, n_samples), t0=0, dt=1 / sample_rate, unit=""
        )
        y = TimeSeries(np.random.randn(n_samples), t0=0, dt=1 / sample_rate, unit="")

        preprocessor = MLPreprocessor(sample_rate=sample_rate)
        preprocessor.fit(X, y)
        X_proc, y_proc = preprocessor.transform(X, y)

        # 形状が保持されていることを確認
        assert X_proc.shape == X.shape
        assert len(y_proc) == len(y)

        # 標準化されていることを確認
        np.testing.assert_allclose(np.mean(X_proc.value), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.std(X_proc.value, ddof=0), 1.0, atol=1e-10)

    def test_multiple_bandpass_bands(self):
        """複数帯域のバンドパスフィルタで動作することを確認."""
        sample_rate = 4096
        n_samples = 16384
        n_channels = 2

        X = TimeSeriesMatrix(
            np.random.randn(n_channels, n_samples), t0=0, dt=1 / sample_rate, unit=""
        )
        y = TimeSeries(np.random.randn(n_samples), t0=0, dt=1 / sample_rate, unit="")

        # 2つの帯域を指定
        preprocessor = MLPreprocessor(
            sample_rate=sample_rate,
            freq_low=[55.0, 110.0],
            freq_high=[65.0, 120.0],
            filt_order=4,
        )
        preprocessor.fit(X, y)
        X_proc, y_proc = preprocessor.transform(X, y)

        # フィルタ係数が2つ設定されていることを確認
        assert len(preprocessor.filter_coeffs_) == 2

        # 形状が保持されていることを確認
        assert X_proc.shape == X.shape
        assert len(y_proc) == len(y)

    def test_valid_frac_boundary_one(self):
        """valid_frac=1.0の境界値ケースをテスト."""
        sample_rate = 1000
        n_samples = 5000
        n_channels = 2

        X = TimeSeriesMatrix(
            np.random.randn(n_channels, n_samples), t0=0, dt=1 / sample_rate, unit=""
        )
        y = TimeSeries(np.random.randn(n_samples), t0=0, dt=1 / sample_rate, unit="")

        # valid_frac=1.0（すべて検証データ）
        preprocessor = MLPreprocessor(sample_rate=sample_rate, valid_frac=1.0)
        X_train, y_train, X_valid, y_valid = preprocessor.split(X, y)

        # 検証データが全データ
        assert X_valid.shape[-1] == X.shape[-1]
        assert len(y_valid) == len(y)

        # 訓練データのサンプル数は0
        assert X_train.shape[-1] == 0
        assert len(y_train) == 0

    def test_robust_standardization(self):
        """Robust標準化（median/MAD）が動作することを確認."""
        sample_rate = 1000
        n_samples = 5000
        n_channels = 2

        # 外れ値を含むデータ作成
        np.random.seed(42)
        X_data = np.random.randn(n_channels, n_samples)
        X_data[0, 100:110] = 100.0  # 外れ値
        X = TimeSeriesMatrix(X_data, t0=0, dt=1 / sample_rate, unit="")

        y_data = np.random.randn(n_samples)
        y_data[200:210] = -100.0  # 外れ値
        y = TimeSeries(y_data, t0=0, dt=1 / sample_rate, unit="")

        # Robust標準化
        preprocessor = MLPreprocessor(
            sample_rate=sample_rate, standardization_method="robust"
        )
        preprocessor.fit(X, y)
        X_proc, y_proc = preprocessor.transform(X, y)

        # 形状が保持されていることを確認
        assert X_proc.shape == X.shape
        assert len(y_proc) == len(y)

        # 外れ値の影響が抑えられていることを確認（中央値が0に近い）
        np.testing.assert_allclose(np.median(X_proc.value, axis=-1), 0.0, atol=0.1)
        np.testing.assert_allclose(np.median(y_proc.value), 0.0, atol=0.1)

    def test_fit_without_y(self):
        """yなしでfitできることを確認（X変換のみのユースケース）."""
        sample_rate = 1000
        n_samples = 5000
        n_channels = 2

        X = TimeSeriesMatrix(
            np.random.randn(n_channels, n_samples), t0=0, dt=1 / sample_rate, unit=""
        )

        # yなしでfit
        preprocessor = MLPreprocessor(sample_rate=sample_rate)
        preprocessor.fit(X, y=None)

        # y_scaler_はNone
        assert preprocessor.y_scaler_ is None

        # transformでもyなしで動作
        X_proc = preprocessor.transform(X, y=None)
        assert X_proc.shape == X.shape

    def test_transform_y_without_fit_y_raises(self):
        """fit時にyを使用せずにtransform時にyを指定するとエラーが発生することを確認."""
        sample_rate = 1000
        n_samples = 5000
        n_channels = 2

        X = TimeSeriesMatrix(
            np.random.randn(n_channels, n_samples), t0=0, dt=1 / sample_rate, unit=""
        )
        y = TimeSeries(np.random.randn(n_samples), t0=0, dt=1 / sample_rate, unit="")

        # yなしでfit
        preprocessor = MLPreprocessor(sample_rate=sample_rate)
        preprocessor.fit(X, y=None)

        # transform時にyを指定するとエラー
        with pytest.raises(RuntimeError, match="yが指定されましたが、fit\\(\\)でyを使用していません"):
            preprocessor.transform(X, y)
