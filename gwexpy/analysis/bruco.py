
"""
Bruco: Brute force Coherence tool for gwexpy.

This module provides a modern implementation of the "Bruco" tool, originally developed
for noise hunting in gravitational wave detectors. It calculates the coherence between
a target channel and a large list of auxiliary channels to identify noise couplings.

Original Implementation:
    https://github.com/mikelovskij/bruco
    Based on the work by Gabriele Vajente (LIGO).

"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from gwpy.timeseries import TimeSeries, TimeSeriesDict
from astropy import units as u

# ロガーの設定
logger = logging.getLogger(__name__)

class Bruco:
    """
    Brute force Coherence (Bruco) scanner.

    Attributes:
        target (str): The name of the target channel (e.g., DARM).
        aux_channels (List[str]): List of auxiliary channels to scan.
        excluded (List[str]): List of channels to exclude from analysis.
    """

    def __init__(self, target_channel: str, aux_channels: List[str], excluded_channels: Optional[List[str]] = None):
        """
        Initialize the Bruco scanner.

        Args:
            target_channel (str): The main channel to analyze.
            aux_channels (List[str]): A list of all available auxiliary channels.
            excluded_channels (List[str], optional): Channels to ignore (e.g., calibration lines).
        """
        self.target = target_channel
        self.aux_channels = aux_channels
        self.excluded = set(excluded_channels) if excluded_channels else set()
        
        # 除外リストとターゲット自体をスキャン対象から外す
        self.channels_to_scan = sorted(list(
            set(self.aux_channels) - self.excluded - {self.target}
        ))
        
        logger.info(f"Bruco initialized. Target: {self.target}, "
                    f"Auxiliary channels: {len(self.channels_to_scan)} (after exclusions)")

    def compute(self, start: Union[int, float], duration: int, 
                fftlength: float = 2.0, overlap: float = 1.0, 
                nproc: int = 4, batch_size: int = 100) -> pd.DataFrame:
        """
        Execute the coherence scan.
        
        Fetches data in batches to optimize memory usage, computes coherence in parallel,
        and returns a ranked DataFrame.

        Args:
            start (int or float): GPS start time.
            duration (int): Duration of data in seconds.
            fftlength (float): FFT length in seconds for coherence calculation.
            overlap (float): Overlap in seconds.
            nproc (int): Number of parallel processes to use.
            batch_size (int): Number of channels to fetch and process at once.

        Returns:
            pd.DataFrame: A DataFrame containing 'channel', 'max_coherence', 
                          'mean_coherence', and 'freq_at_max'.
        """
        end = start + duration
        logger.info(f"Starting Bruco scan at {start} for {duration}s. Batch size: {batch_size}")

        # 1. ターゲットチャンネルのデータ取得 (これは一度だけ)
        try:
            logger.info(f"Fetching target data: {self.target}")
            target_ts = TimeSeries.get(self.target, start, end)
        except Exception as e:
            logger.error(f"Failed to fetch target channel {self.target}: {e}")
            return pd.DataFrame()

        results = []
        total_channels = len(self.channels_to_scan)

        # 2. バッチ処理ループ
        for i in range(0, total_channels, batch_size):
            batch_channels = self.channels_to_scan[i : i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(total_channels // batch_size) + 1} "
                        f"({len(batch_channels)} channels)")

            # バッチごとのデータ取得
            try:
                # verbose=False to reduce clutter
                aux_data = TimeSeriesDict.get(batch_channels, start, end, allow_tape=True, nproc=nproc)
            except Exception as e:
                logger.warning(f"Batch fetch failed or partial failure: {e}")
                # 失敗した場合、個別に再トライするかスキップする実装も可能だが、簡略化のためスキップ
                continue

            # 並列計算の実行
            batch_results = self._process_batch(target_ts, aux_data, fftlength, overlap, nproc)
            results.extend(batch_results)

        # 3. 結果の集計とソート
        df = pd.DataFrame(results)
        if not df.empty:
            # カラムの型を整理
            df = df.astype({
                'max_coherence': float,
                'mean_coherence': float,
                'freq_at_max': float
            })
            # コヒーレンス最大値で降順ソート
            df = df.sort_values(by='max_coherence', ascending=False).reset_index(drop=True)
        
        logger.info(f"Scan complete. {len(df)} channels analyzed.")
        return df

    def _process_batch(self, target_ts: TimeSeries, aux_dict: TimeSeriesDict, 
                       fftlength: float, overlap: float, nproc: int) -> List[Dict[str, Any]]:
        """
        Helper method to process a single batch of data using multiprocessing.
        """
        results = []
        with ProcessPoolExecutor(max_workers=nproc) as executor:
            future_to_ch = {
                executor.submit(
                    self._calculate_pair_coherence, 
                    target_ts, 
                    aux_dict[ch], 
                    fftlength, 
                    overlap
                ): ch for ch in aux_dict
            }

            for future in as_completed(future_to_ch):
                ch = future_to_ch[future]
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                except Exception as e:
                    logger.debug(f"Calculation failed for {ch}: {e}")
        
        return results

    @staticmethod
    def _calculate_pair_coherence(target: TimeSeries, aux: TimeSeries, 
                                  fftlength: float, overlap: float) -> Optional[Dict[str, Any]]:
        """
        Static worker function to calculate coherence between two series.
        Handles resampling if sampling rates differ.
        """
        try:
            # サンプリングレートの不一致を処理 (低い方に合わせる)
            if target.sample_rate != aux.sample_rate:
                # 単位付きの比較が必要な場合があるため .value を使うか astropy の比較を使う
                sr_target = target.sample_rate.value
                sr_aux = aux.sample_rate.value
                
                common_rate = min(sr_target, sr_aux)
                
                if sr_target > common_rate:
                    # targetをダウンサンプリング（コピーを作成して元のデータを破壊しない）
                    target = target.resample(common_rate * u.Hz)
                if sr_aux > common_rate:
                    aux = aux.resample(common_rate * u.Hz)

            # データ長が一致しない場合（欠損など）、短い方に合わせる（crop）
            if len(target) != len(aux):
                min_len = min(len(target), len(aux))
                target = target[:min_len]
                aux = aux[:min_len]

            # コヒーレンス計算 (gwpy native method)
            coh = target.coherence(aux, fftlength=fftlength, overlap=overlap)
            
            # メトリクス抽出
            max_val = coh.max().value
            mean_val = coh.mean().value
            idx_max = np.argmax(coh.value)
            freq_at_max = coh.frequencies[idx_max].value

            return {
                'channel': aux.name,
                'max_coherence': max_val,
                'mean_coherence': mean_val,
                'freq_at_max': freq_at_max
            }

        except Exception as e:
            # ログを出したいが、subprocess内のため静かにNoneを返すか、例外を投げる
            # 実運用ではエラーの種類を返すとデバッグしやすい
            return None