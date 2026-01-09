import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u

from gwpy.types import Array2D, FrequencySeries

class BifrequencyMap(Array2D):
    """2つの異なる周波数軸を持つマップ（応答関数や相関行列）クラス。
    
    (行, 列) = (出力周波数, 入力周波数) としてデータを保持します。
    """

    @property
    def frequencies_in(self):
        """入力周波数軸 (X軸/Columns)"""
        return self.xindex

    @property
    def frequencies_out(self):
        """出力周波数軸 (Y軸/Rows)"""
        return self.yindex

    @classmethod
    def from_points(cls, data, f_out, f_in, **kwargs):
        """
        データと2つの周波数軸からインスタンスを生成します。
        
        Parameters
        ----------
        data : array-like
            形状が (len(f_out), len(f_in)) の2次元配列
        f_out : array-like
            出力周波数軸（Y軸）
        f_in : array-like
            入力周波数軸（X軸）
        """
        # Array2Dは (y, x) の順で初期化されるため、それに合わせる
        return cls(data, yindex=f_out, xindex=f_in, **kwargs)

    def propagate(self, input_spectrum, interpolate=True, fill_value=0):
        """
        入力スペクトルに応答関数を適用し、出力スペクトルを計算します。
        
        入力スペクトルの周波数軸がこのマップの入力軸(xindex)と異なる場合、
        自動的に線形補間を行って合わせることができます。

        Parameters
        ----------
        input_spectrum : FrequencySeries
            入力となるノイズスペクトル。
        interpolate : bool, optional
            Trueの場合、入力スペクトルの周波数軸をマップの入力軸に合わせて補間します。
            Falseの場合、サイズが一致していないとエラーになります。デフォルトは True。
        fill_value : float, optional
            補間時に範囲外となった場合の埋め込み値。デフォルトは 0。

        Returns
        -------
        FrequencySeries
            出力周波数軸(frequencies_out)を持つ計算結果。
        """
        # 入力データの検証
        if not isinstance(input_spectrum, FrequencySeries):
            raise ValueError("input_spectrum must be a gwpy.frequencyseries.FrequencySeries")

        # 演算用の入力データ配列
        in_val = input_spectrum.value
        in_freq = input_spectrum.frequencies.value
        map_in_freq = self.frequencies_in.value
        
        # 周波数軸の整合性チェックと補間
        if interpolate:
            # 軸が完全に一致していない場合は補間を行う
            if len(in_freq) != len(map_in_freq) or not np.allclose(in_freq, map_in_freq):
                # 線形補間関数を作成
                f = interp1d(in_freq, in_val, kind='linear', 
                             bounds_error=False, fill_value=fill_value)
                # マップのX軸に合わせてリサンプリング
                in_val = f(map_in_freq)
        else:
            # 補間しない場合はサイズチェックのみ
            if len(in_val) != self.shape[1]:
                raise ValueError(
                    f"Input spectrum size {len(in_val)} does not match "
                    f"map input size {self.shape[1]}. Enable `interpolate=True`."
                )

        # 行列積の計算: (N_out, N_in) @ (N_in,) -> (N_out,)
        # これにより、入力周波数軸方向の成分が積算され、出力周波数軸のデータになる
        out_val = self.value @ in_val

        # 単位の計算
        new_unit = self.unit * input_spectrum.unit

        # 結果をFrequencySeriesとして返す（軸はマップのY軸=出力軸を使用）
        return FrequencySeries(
            out_val,
            frequencies=self.frequencies_out,
            unit=new_unit,
            name=f"Projected: {self.name} x {input_spectrum.name}"
        )

    def plot(self, **kwargs):
        """ヒートマップ描画 (軸ラベルを自動設定)"""
        kwargs.setdefault("xlabel", "Input Frequency [Hz]")
        kwargs.setdefault("ylabel", "Output Frequency [Hz]")
        kwargs.setdefault("cmap", "inferno") # 非ゼロが見やすい色
        return super().plot(**kwargs)