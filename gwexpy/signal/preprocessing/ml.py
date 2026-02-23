"""
gwexpy.signal.preprocessing.ml
-------------------------------

機械学習用の前処理パイプライン。

このモジュールは、DeepCleanなどのノイズ除去タスクで使用される前処理
（データ分割、バンドパスフィルタリング、標準化）を、汎用的な
scikit-learn風のTransformer APIとして提供します。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import astropy.units as u
import numpy as np

if TYPE_CHECKING:
    from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix

    from .standardize import StandardizationModel


class MLPreprocessor:
    """
    機械学習用の前処理パイプライン。

    データ分割、バンドパスフィルタリング、チャンネルごと標準化を行う
    scikit-learn風のTransformer。DeepCleanなどのノイズ除去タスクだけでなく、
    ランダムフォレスト、XGBoost、その他の機械学習モデルでも使用可能。

    Parameters
    ----------
    sample_rate : Quantity or float
        サンプリングレート（Hz単位）
    freq_low : list[float] or None, optional
        バンドパスフィルタの低周波カットオフ（複数帯域対応）。
        Noneの場合はフィルタリングをスキップ。
    freq_high : list[float] or None, optional
        バンドパスフィルタの高周波カットオフ（複数帯域対応）。
        Noneの場合はフィルタリングをスキップ。
    filt_order : int, optional
        バターワースフィルタの次数（デフォルト: 8）
    valid_frac : float, optional
        検証データの割合（0.0～1.0、デフォルト: 0.0）
        0.0の場合は分割なし。
    standardization_method : str, optional
        標準化手法（'zscore'または'robust'、デフォルト: 'zscore'）

    Attributes
    ----------
    X_scaler_ : StandardizationModel or None
        参照チャンネル用の標準化モデル（fit後に設定）
    y_scaler_ : StandardizationModel or None
        ターゲットチャンネル用の標準化モデル（fit後に設定）
    filter_coeffs_ : list[np.ndarray] or None
        バンドパスフィルタ係数（SOS形式、fit後に設定）
    is_fitted_ : bool
        fitが完了しているかのフラグ

    Examples
    --------
    基本的な使用例：

    >>> from gwexpy.timeseries import TimeSeriesMatrix, TimeSeries
    >>> from gwexpy.signal.preprocessing import MLPreprocessor
    >>>
    >>> # データ読み込み
    >>> witnesses = TimeSeriesMatrix(...)  # (n_channels, n_samples)
    >>> strain = TimeSeries(...)            # (n_samples,)
    >>>
    >>> # 前処理パイプライン
    >>> preprocessor = MLPreprocessor(
    ...     sample_rate=4096,
    ...     freq_low=[55.0],
    ...     freq_high=[65.0],
    ...     valid_frac=0.2
    ... )
    >>>
    >>> # 分割 → fit → transform
    >>> X_train, y_train, X_valid, y_valid = preprocessor.split(witnesses, strain)
    >>> preprocessor.fit(X_train, y_train)
    >>> X_train_proc, y_train_proc = preprocessor.transform(X_train, y_train)
    >>> X_valid_proc, y_valid_proc = preprocessor.transform(X_valid, y_valid)

    Notes
    -----
    処理順序はDeepClean v2の実装に準拠：
    1. データ分割（時系列順）
    2. X標準化パラメータ学習（フィルタリングなし）
    3. フィルタ係数設計
    4. yフィルタリング → y標準化パラメータ学習

    重要な注意点：
    - **Xはフィルタリングしない**（参照チャンネルは生データのまま標準化）
    - **yはフィルタリングしてから標準化**（ターゲットチャンネルは帯域制限）
    """

    def __init__(
        self,
        sample_rate: u.Quantity | float,
        freq_low: list[float] | None = None,
        freq_high: list[float] | None = None,
        filt_order: int = 8,
        valid_frac: float = 0.0,
        standardization_method: str = "zscore",
    ):
        self.sample_rate = sample_rate
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.filt_order = filt_order
        self.valid_frac = valid_frac
        self.standardization_method = standardization_method

        # 内部状態（fit後に設定）
        self.X_scaler_: Optional[StandardizationModel] = None
        self.y_scaler_: Optional[StandardizationModel] = None
        self.filter_coeffs_: Optional[list[np.ndarray]] = None
        self.is_fitted_ = False

    def split(
        self,
        X: TimeSeriesMatrix,
        y: TimeSeries,
    ) -> tuple[TimeSeriesMatrix, TimeSeries, TimeSeriesMatrix, TimeSeries]:
        """
        データを訓練/検証に分割。

        Parameters
        ----------
        X : TimeSeriesMatrix
            参照チャンネル (shape: (n_channels, n_samples))
        y : TimeSeries
            ターゲットチャンネル (shape: (n_samples,))

        Returns
        -------
        X_train : TimeSeriesMatrix
            訓練用参照チャンネル
        y_train : TimeSeries
            訓練用ターゲットチャンネル
        X_valid : TimeSeriesMatrix
            検証用参照チャンネル
        y_valid : TimeSeries
            検証用ターゲットチャンネル
        """
        if self.valid_frac == 0.0:
            # 分割なし: crop()を使って長さ0の検証データを作成
            empty_X = X.crop(start=X.t0, end=X.t0)
            empty_y = y.crop(start=y.t0, end=y.t0)
            return X, y, empty_X, empty_y

        # 整数秒単位に調整（DeepClean互換）
        sample_rate_hz = self._get_sample_rate_hz()
        total_length = len(y)
        valid_size_float = self.valid_frac * total_length
        valid_length_sec = int(valid_size_float / sample_rate_hz)
        valid_size = int(valid_length_sec * sample_rate_hz)
        train_size = total_length - valid_size

        # crop()メソッドで分割（時間情報を保持）
        train_end_time = X.t0 + train_size * X.dt
        valid_start_time = train_end_time

        X_train = X.crop(end=train_end_time)
        X_valid = X.crop(start=valid_start_time)
        y_train = y.crop(end=train_end_time)
        y_valid = y.crop(start=valid_start_time)

        return X_train, y_train, X_valid, y_valid

    def fit(
        self,
        X: TimeSeriesMatrix,
        y: TimeSeries | None = None,
    ) -> MLPreprocessor:
        """
        統計量とフィルタ係数を学習。

        Parameters
        ----------
        X : TimeSeriesMatrix
            参照チャンネル（訓練データ）
        y : TimeSeries or None, optional
            ターゲットチャンネル（訓練データ）
            Noneの場合はyの標準化をスキップ。

        Returns
        -------
        self : MLPreprocessor
            fitted preprocessor
        """
        from scipy.signal import butter

        from gwexpy.signal.preprocessing import standardize

        # 1. Xの標準化パラメータを学習
        X_val = self._extract_value(X)  # ndarray取得
        X_std, X_model = standardize(
            X_val,
            method=self.standardization_method,
            axis=-1,  # time軸で標準化（チャンネルごと独立）
            return_model=True,
        )
        self.X_scaler_ = X_model

        # 2. フィルタ係数を設計
        if self.freq_low is not None and self.freq_high is not None:
            sample_rate_hz = self._get_sample_rate_hz()
            self.filter_coeffs_ = []
            for f_low, f_high in zip(self.freq_low, self.freq_high):
                sos = butter(
                    self.filt_order,
                    [f_low, f_high],
                    btype="bandpass",
                    fs=sample_rate_hz,
                    output="sos",
                )
                self.filter_coeffs_.append(sos)
        else:
            self.filter_coeffs_ = None

        # 3. yの標準化パラメータを学習
        if y is not None:
            # yをフィルタリング
            y_filt = self._apply_bandpass(y)

            # yの標準化パラメータを学習
            y_val = self._extract_value(y_filt)
            y_std, y_model = standardize(
                y_val,
                method=self.standardization_method,
                axis=-1,
                return_model=True,
            )
            self.y_scaler_ = y_model
        else:
            self.y_scaler_ = None

        self.is_fitted_ = True
        return self

    def transform(
        self,
        X: TimeSeriesMatrix,
        y: TimeSeries | None = None,
    ) -> tuple[TimeSeriesMatrix, TimeSeries] | TimeSeriesMatrix:
        """
        フィルタリングと標準化を適用。

        Parameters
        ----------
        X : TimeSeriesMatrix
            参照チャンネル
        y : TimeSeries or None, optional
            ターゲットチャンネル
            Noneの場合はXのみ返す。

        Returns
        -------
        X_proc : TimeSeriesMatrix
            処理済みX（dimensionless_unscaled単位）
        y_proc : TimeSeries (yが指定された場合)
            処理済みy（dimensionless_unscaled単位）
        """
        if not self.is_fitted_:
            raise RuntimeError("fit()を先に呼び出してください")

        # 1. Xの標準化
        X_val = self._extract_value(X)
        # meanとscaleをブロードキャスト可能な形状にreshape
        # X_valの形状: (n_rows, n_cols, n_samples) or (n_channels, n_samples)
        # X_scaler_.meanの形状: (n_channels,) -> (n_channels, 1, ...)にreshapeが必要
        assert self.X_scaler_ is not None
        mean = self.X_scaler_.mean
        scale = self.X_scaler_.scale

        # スカラーの場合は配列に変換（単一チャンネルケース対応）
        mean = np.atleast_1d(mean)
        scale = np.atleast_1d(scale)

        # X_valの次元数に応じてreshape
        if X_val.ndim == 3:
            mean = mean[:, None, None]
            scale = scale[:, None, None]
        elif X_val.ndim == 2:
            mean = mean[:, None]
            scale = scale[:, None]
        X_std_val = (X_val - mean) / scale
        X_proc = self._reconstruct_timeseries_matrix(
            X_std_val, X, unit=u.dimensionless_unscaled
        )

        # 2. yの処理
        if y is not None:
            if self.y_scaler_ is None:
                raise RuntimeError(
                    "yが指定されましたが、fit()でyを使用していません。"
                )

            # yフィルタリング
            y_filt = self._apply_bandpass(y)

            # y標準化
            y_val = self._extract_value(y_filt)
            # y_scaler_.meanとscaleはスカラーまたは1次元配列
            # y_valが1次元の場合はそのまま、2次元の場合はreshapeが必要
            y_mean = self.y_scaler_.mean
            y_scale = self.y_scaler_.scale

            # スカラーの場合は配列に変換
            y_mean = np.atleast_1d(y_mean)
            y_scale = np.atleast_1d(y_scale)

            # y_valの次元数に応じてreshape
            if y_val.ndim > 1 and y_mean.ndim > 0:
                y_mean = y_mean[:, None]
                y_scale = y_scale[:, None]
            y_std_val = (y_val - y_mean) / y_scale
            y_proc = self._reconstruct_timeseries(
                y_std_val, y_filt, unit=u.dimensionless_unscaled
            )

            return X_proc, y_proc

        return X_proc

    # ヘルパーメソッド（プライベート）

    def _get_sample_rate_hz(self) -> float:
        """sample_rateをHz単位のfloatに変換。"""
        if hasattr(self.sample_rate, "to"):
            # Quantity型の場合はHz単位に変換
            return float(self.sample_rate.to(u.Hz).value)
        # float型の場合はそのまま返す
        return float(self.sample_rate)

    def _extract_value(self, ts) -> np.ndarray:
        """TimeSeries/TimeSeriesMatrixからndarrayを取得。"""
        if hasattr(ts, "value"):
            return ts.value
        return np.asarray(ts)

    def _reconstruct_timeseries_matrix(self, val, original, unit=None):
        """ndarrayからTimeSeriesMatrixを再構築。"""
        # 元のクラスを使用して新しいインスタンスを作成（コンストラクタで単位を設定）
        if unit is not None:
            new_mat = original.__class__(val, t0=original.t0, dt=original.dt, unit=unit)
        else:
            new_mat = original.__class__(val, t0=original.t0, dt=original.dt)

        # メタデータを保持
        if hasattr(original, "channel_names"):
            new_mat.channel_names = original.channel_names

        return new_mat

    def _reconstruct_timeseries(self, val, original, unit=None):
        """ndarrayからTimeSeriesを再構築。"""
        # 元のクラスを使用して新しいインスタンスを作成（コンストラクタで単位を設定）
        if unit is not None:
            new_ts = original.__class__(val, t0=original.t0, dt=original.dt, unit=unit)
        else:
            new_ts = original.__class__(val, t0=original.t0, dt=original.dt)

        # メタデータを保持
        if hasattr(original, "name"):
            new_ts.name = original.name

        return new_ts

    def _apply_bandpass(self, y):
        """バンドパスフィルタを適用。"""
        if self.filter_coeffs_ is None:
            # フィルタリングなし
            return y

        from scipy.signal import sosfiltfilt

        y_val = self._extract_value(y)
        y_filt_val = np.zeros_like(y_val)

        # 各帯域のフィルタを適用して加算
        for sos in self.filter_coeffs_:
            y_filt_val += sosfiltfilt(sos, y_val, axis=-1)

        # TimeSeriesとして再構築
        return self._reconstruct_timeseries(y_filt_val, y)


__all__ = ["MLPreprocessor"]
