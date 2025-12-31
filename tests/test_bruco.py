import numpy as np
import pandas as pd
from unittest.mock import patch
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from astropy import units as u

from gwexpy.analysis.bruco import Bruco

# モックデータの生成ヘルパー
def create_mock_timeseries(name, duration, sample_rate, signal_freq=None, noise_amp=0.1):
    t = np.linspace(0, duration, int(duration * sample_rate))
    data = np.random.normal(0, noise_amp, size=len(t))
    if signal_freq:
        data += np.sin(2 * np.pi * signal_freq * t)
    
    return TimeSeries(data, t0=0, sample_rate=sample_rate * u.Hz, name=name)

class TestBruco:
    def test_init(self):
        target = "TARGET"
        aux = ["AUX1", "AUX2", "AUX3"]
        excluded = ["AUX2"]
        
        bruco = Bruco(target, aux, excluded)
        
        assert bruco.target == target
        assert len(bruco.channels_to_scan) == 2
        assert "AUX1" in bruco.channels_to_scan
        assert "AUX3" in bruco.channels_to_scan
        assert "AUX2" not in bruco.channels_to_scan
        assert target not in bruco.channels_to_scan

    @patch('gwexpy.analysis.bruco.TimeSeries.get')
    @patch('gwexpy.analysis.bruco.TimeSeriesDict.get')
    def test_compute(self, mock_tsd_get, mock_ts_get):
        # 1. セットアップ
        target_channel = "H1:TARGET"
        aux_channels = ["H1:AUX1", "H1:AUX2"]
        duration = 10
        sample_rate = 256
        
        # ターゲット: 10Hzの信号を持つ
        mock_target_data = create_mock_timeseries(target_channel, duration, sample_rate, signal_freq=10)
        mock_ts_get.return_value = mock_target_data
        
        # AUX1: 10Hzの信号を持つ (高いコヒーレンスが期待される)
        mock_aux1_data = create_mock_timeseries("H1:AUX1", duration, sample_rate, signal_freq=10)
        
        # AUX2: ノイズのみ (低いコヒーレンスが期待される)
        mock_aux2_data = create_mock_timeseries("H1:AUX2", duration, sample_rate, signal_freq=None)
        
        # TimeSeriesDict.get は辞書っぽいものを返す
        mock_tsd_get.return_value = TimeSeriesDict({
            "H1:AUX1": mock_aux1_data,
            "H1:AUX2": mock_aux2_data
        })
        
        bruco = Bruco(target_channel, aux_channels)
        
        # 2. 実行
        # macなどでmultiprocessingがエラーになるのを防ぐためnproc=1でテストするか、
        # モックがpicklableでない可能性を考慮してシンプルな実行にする
        # ここではロジックのテストなので nproc=1 推奨
        df = bruco.compute(start=0, duration=duration, fftlength=1.0, overlap=0.5, nproc=1)
        
        # 3. 検証
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        
        # AUX1 が上位に来るはず
        assert df.iloc[0]['channel'] == "H1:AUX1"
        assert df.iloc[0]['max_coherence'] > 0.5  # 信号があるのでコヒーレンスは高いはず
        
        # 周波数もチェック (10Hz付近)
        assert np.isclose(df.iloc[0]['freq_at_max'], 10.0, atol=1.0)
        
        # AUX2 は下位
        assert df.iloc[1]['channel'] == "H1:AUX2"
        assert df.iloc[1]['max_coherence'] < 0.5  # ノイズだけなので低いはず

    def test_process_batch_error_handling(self):
        # バッチ処理中にエラーが発生しても全体が止まらないかテスト
        pass # 時間があれば実装

    def test_resampling_logic(self):
         # サンプリングレートが異なる場合のロジックテスト
        target_ts = create_mock_timeseries("T", 10, 512, signal_freq=10)
        aux_ts = create_mock_timeseries("A", 10, 256, signal_freq=10)
        
        # _calculate_pair_coherence は静的メソッドで private だがテストしたい
        # しかし実装上はインスタンスメソッドではないのでクラスから呼べる
        res = Bruco._calculate_pair_coherence(target_ts, aux_ts, fftlength=1, overlap=0)
        
        assert res is not None
        assert res['max_coherence'] > 0.8
