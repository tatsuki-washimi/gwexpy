import warnings
import numpy as np
import bottleneck as bn
import pandas as pd
from html import escape
from collections import OrderedDict
from typing import Optional, Union, Mapping, Any
from datetime import datetime
from astropy import units as u
from gwpy.time import LIGOTimeGPS, to_gps
from gwpy.types.array import Array
from gwpy.types.index import Index
from gwpy.types.series import Series

from gwexpy.types.metadata import MetaData, MetaDataDict, MetaDataMatrix


class SeriesMatrix:
    series_class = Series
    def __init__(
        self,
        value: Optional[np.ndarray] = None,                          # ndarray: shape=(N_rows, N_cols, N_samples)
        x0: float = 0.0,                                             # series軸(サンプリング)の始点
        dx: float = 1.0,                                             # series軸(サンプリング)の分解能
        xunit: Union[str, u.Unit] = 1,                               # series軸(サンプリング)の物理単位（例: Hz, s）
        epoch: Union[int, float, str, datetime, LIGOTimeGPS] = 0.0,  # データの基準時刻 (TimeSeriesならt0)
        name: str = "",                                              # この行列の人間可読ラベル
        names: Optional[Union[str, np.ndarray]] = None,              # 成分ごとのラベル（N×Mの文字列配列）
        units: Optional[Union[u.Unit, str, np.ndarray]] = None,      # 成分ごとの単位（N×Mの astropy.units）
        dtype: np.dtype = np.complex128,                             # 内部データ型
        rows: Optional[Mapping[str, Mapping[str, Union[str, u.Unit]]]] = None,  # 行チャンネルのメタデータ（dict形式）
        cols: Optional[Mapping[str, Mapping[str, Union[str, u.Unit]]]] = None,  # 列チャンネルのメタデータ（dict形式）
        copy: bool = True,               # 値のコピーオプション
        subok: bool = True,              # サブクラス許可フラグ
        attrs: Optional[Mapping[str, Any]] = None  # メタ情報
    ):
        # ---- 値の整形（Quantity対応、次元チェック含む） ----
        value_units = u.dimensionless_unscaled
        if value is None:
            value = np.zeros((0, 0, 0), dtype=dtype)
        if isinstance(value, u.Quantity):
            value_units = value.unit
            value = value.value             

        if value.ndim == 1:            # → (1, 1, S) として扱う
            value = value[None, None, :]
        elif value.ndim == 2:            # → (N, 1, S) として扱う
            value = value[:, None, :]
        elif value.ndim > 3:
            raise ValueError(f"value must be a 1D, 2D, or 3D array (got shape {value.shape})")
            
        self.value = np.array(value, dtype=dtype)
        N_rows, N_cols, N_samples = self.value.shape
        
        # ----  Series軸（例: frequency）の情報 ----
        self.x0 = x0
        self.dx = dx
        self.xunit = u.Unit(xunit)
        
        # ----  メタ情報 ----
        self.name = name
        self.attrs = dict(attrs) if attrs is not None else {}
        if epoch is not None:
            try:
                self.epoch = to_gps(epoch)
            except Exception:
                self.epoch = None
        else:
            self.epoch = None


        # 成分ラベル, チャンネル, 単位    
        self.names = np.full((N_rows, N_cols), '', dtype=str)
        self.units = np.full((N_rows, N_cols), u.Unit(value_units), dtype=object)

        initial_names = np.full((N_rows, N_cols), '', dtype=str)
        if isinstance(names, str):
            initial_names[:] = names
        elif isinstance(names, (list, np.ndarray)):
            if np.array(names).shape == (N_rows, N_cols):
                initial_names = np.array(names, dtype=str)
            else:
                warnings.warn(f"Invalid shape for names: expected {(N_rows, N_cols)}, got {np.array(names).shape}. Ignoring provided names.")
        elif names is not None:
            warnings.warn(f"Invalid type for names: expected str, list, or ndarray, got {type(names)}. Ignoring provided names.")
        self.names = initial_names
        
        
        # If units not provided, use the unit from input Quantity (value_units)
        self.units = self._init_unit_array(units if units is not None else value_units, N_rows, N_cols)

        # 行と列のメタデータ
        self.rows = self._init_axis_metadata(rows, N_rows, axis='row')
        self.cols = self._init_axis_metadata(cols, N_cols, axis='col')


    def _init_unit_array(self, units, N_rows, N_cols):
        shape = (N_rows, N_cols)
        def try_unit(val):
            try:
                return u.Unit(val)
            except Exception:
                return u.dimensionless_unscaled
    
        if isinstance(units, (u.Unit, str)): # スカラー入力の処理
            units = np.full(shape, try_unit(units), dtype=object)
        elif np.isscalar(units) or units is None:
            units = np.full(shape, u.dimensionless_unscaled, dtype=object)
        else:  # 配列入力の処理
            units = np.array(units, dtype=object)
            if units.shape == shape:
                units = np.array([[try_unit(units[i,j]) for j in range(shape[1])] for i in range(shape[0])], dtype=object)
            else:
                raise ValueError(f"Shape mismatch: expected {shape}, got {units.shape}")
            
        return units


    def _init_axis_metadata(self, metadata_dict, N, axis='axis'): # 補完付きで OrderedDict として行・列情報を初期化
            initial_keys = [f"{axis}{i}" for i in range(N)]
            keys = list(initial_keys) # デフォルトのキーリストをコピー

            if isinstance(metadata_dict, (dict, OrderedDict)):
                input_keys = list(metadata_dict.keys())
                
                # 入力されたキーでinitial_keysの先頭を更新
                num_keys_to_copy = min(len(initial_keys), len(input_keys))
                keys[:num_keys_to_copy] = input_keys[:num_keys_to_copy]

            # OrderedDictをデフォルトメタデータで初期化
            # デフォルト単位を u.dimensionless_unscaled に変更
            od = OrderedDict({key: {'unit': u.dimensionless_unscaled, 'name': '', 'channel': ''} for key in keys}) 
            
            # 入力されたメタデータで更新
            if isinstance(metadata_dict, (dict, OrderedDict)):
                # 入力キーリストを再取得 (OrderedDictのキー順にアクセスするため)
                input_keys_ordered = list(metadata_dict.keys())
                
                for key in input_keys_ordered:
                    if key in od: # 既存のキーのみを更新
                        # getメソッドで存在しないキーでもエラーにならないようにする
                        input_meta = metadata_dict.get(key, {})
                        od[key]['unit']    = u.Unit(input_meta.get('unit', u.dimensionless_unscaled)) # デフォルト単位の変更
                        od[key]['name']    = input_meta.get('name', '')
                        od[key]['channel'] = input_meta.get('channel', '')
                    # else: # odに存在しないキーは無視される（最初のキー補完ロジックで処理済み）
                    #    warnings.warn(f"Input metadata key '{key}' not found in generated keys. Ignoring.")

            return od

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def N_rows(self):
        return self.value.shape[0]

    @property
    def N_cols(self):
        return self.value.shape[1]

    @property
    def N_channels(self):
        return self.N_rows * self.N_cols

    @property
    def N_samples(self):
        return self.value.shape[2]

    @property
    def xindex(self):
        if self.N_samples == 0:
            return np.array([], dtype=float) * self.xunit
        return (self.x0 + self.dx * np.arange(self.N_samples)) * self.xunit

        
    @property
    def xspan(self):
        if self.N_samples == 0:
            return (None, None)
        return (self.xindex[0], self.xindex[-1])
        
    @property
    def xarray(self):
        """Return sample axis values without double-unit multiplication."""
        return self.xindex

    @property
    def duration(self):
        if self.N_samples == 0:
            return 0 * self.xunit
        return self.xindex[-1] - self.xindex[0]
    
    @property
    def xstep(self):
        return self.dx * self.xunit
        
    @property
    def row_keys(self):
        return tuple(self.rows.keys())

    @property
    def col_keys(self):
        return tuple(self.cols.keys())
        
    @property
    def keys(self):
        return (self.row_keys, self.col_keys)
        
    @property
    def row_units(self):
        return tuple(self.rows[k]['unit'] for k in self.row_keys)
    
    @property
    def col_units(self):
        return tuple(self.cols[k]['unit'] for k in self.col_keys)
    
    @property
    def row_names(self):
        return tuple(self.rows[k]['name'] for k in self.row_keys)
    
    @property
    def col_names(self):
        return tuple(self.cols[k]['name'] for k in self.col_keys)

    @property
    def row_channels(self):
        return tuple(self.rows[k]['channel'] for k in self.row_keys)
    
    @property
    def col_channels(self):
        return tuple(self.cols[k]['channel'] for k in self.col_keys)
    

    def __getitem__(self, keys):
        key_row, key_col = keys

        # keyをindexにする
        if isinstance(key_row, str):
            if key_row not in self.row_keys:
                raise KeyError(f"Invalid row key: '{key_row}' not in {self.row_keys}")
            i_row = self.row_keys.index(key_row)
        elif isinstance(key_row, int):
            if not (0 <= key_row < self.N_rows):
                raise IndexError(f"Row index out of range: {key_row}")
            i_row = key_row
        else:
            raise TypeError("Row index must be str or int")
    
        if isinstance(key_col, str):
            if key_col not in self.col_keys:
                raise KeyError(f"Invalid column key: '{key_col}' not in {self.col_keys}")
            i_col = self.col_keys.index(key_col)
        elif isinstance(key_col, int):
            if not (0 <= key_col < self.N_cols):
                raise IndexError(f"Column index out of range: {key_col}")
            i_col = key_col
        else:
            raise TypeError("Column index must be str or int")

        # unitの設定
        unit_value = self.units[i_row, i_col] if self.units is not None else None
        unit = unit_value if isinstance(unit_value, u.Unit) else u.dimensionless_unscaled

        # nameの設定 or 生成
        name_value = self.names[i_row, i_col] if self.names is not None else ''
        if isinstance(name_value, str) and name_value.strip():
            name = name_value
        else:
            name = f"{self.row_names[i_row]}, {self.col_names[i_col]}"
        
        # channelの生成
        row_ch = str(self.row_channels[i_row])
        col_ch = str(self.col_channels[i_col])
        channel = row_ch + ', ' + col_ch
            
        return self.series_class(self.value[i_row, i_col, :],
                                 xindex = self.xindex,
                                 xunit  = self.xunit,
                                 epoch  = self.epoch,
                                 unit = unit,
                                 name = name,
                                 channel = channel
                                 )

    def __setitem__(self, keys, value):
            key_row_input, key_col_input = keys

            # --- 初回代入時の自動初期化 ---
            # 現在の行列サイズが (0, 0, 0) の場合に初期化を試みる
            if self.N_samples == 0 and self.N_rows == 0 and self.N_cols == 0:
                if not isinstance(value, Series):
                    # Series 以外では行列サイズが決まらないため初期化できない
                    raise ValueError("Cannot initialize SeriesMatrix with non-Series value on first assignment. Provide a Series object.")

                # Series オブジェクトの場合、その情報を使って初期化
                sample_length = len(value)
                # デフォルトdtypeを使用
                self.value = np.zeros((1, 1, sample_length), dtype=self.value.dtype)
                self.units = np.full((1, 1), value.unit, dtype=object)
                self.names = np.full((1, 1), value.name or '', dtype=str) # value.name が None の場合を考慮

                # x軸情報の初期化
                # xindex の最初の要素が Quantity かどうかでアクセス方法を変える
                if hasattr(value, 'xindex') and len(value.xindex) > 0:
                    if isinstance(value.xindex[0], u.Quantity): # xindex全体がQuantityの場合を想定
                        self.x0 = value.xindex[0].value
                        self.xunit = value.xunit
                        self.dx = value.dx.value if hasattr(value.dx, 'value') else 1.0 # dxもQuantityの場合を考慮
                    else: # xindexがQuantityでない場合 (ndarrayなど)
                        self.x0 = value.xindex[0]
                        self.xunit = u.dimensionless_unscaled # 単位がない場合は単位なし
                        self.dx = value.dx if hasattr(value, 'dx') else 1.0
                else: # xindex が空または存在しない場合
                    self.x0 = 0.0
                    self.dx = 1.0
                    self.xunit = u.dimensionless_unscaled


                self.epoch = value.epoch if hasattr(value, 'epoch') else None # epochもSeriesから取得

                # 行と列のキーを設定 (文字列キーを優先、整数ならデフォルト文字列)
                key_row = key_row_input if isinstance(key_row_input, str) else 'row0'
                channel_row = self.row_channels[i_row]
                channel_col = self.col_channels[i_col]
                channel_parts = []
                if channel_row and str(channel_row).strip():
                    channel_parts.append(str(channel_row).strip())
                if channel_col and str(channel_col).strip():
                    channel_parts.append(str(channel_col).strip())
                channel = ', '.join(channel_parts) if channel_parts else None

                # デフォルトメタデータで初期化
                self.rows = OrderedDict({key_row: {
                    'unit': u.dimensionless_unscaled,
                    'name': key_row,
                    'channel': value.channel if hasattr(value, 'channel') and value.channel is not None else ''
                }})
                self.cols = OrderedDict({key_col: {
                    'unit': u.dimensionless_unscaled,
                    'name': key_col,
                    'channel': ''
                }})

                # 値を設定
                self.value[0, 0, :] = value.value
                return # 初期化と値設定完了

            # --- 既存行列への代入 ---

            # キーを入力インデックスに変換、必要なら行列を拡張

            # 現在の次元を確認
            current_row_keys = list(self.rows.keys())
            current_col_keys = list(self.cols.keys())

            # 必要な次元拡張を確認
            need_expand = False

            # 行キーの処理
            if isinstance(key_row_input, str):
                key_row = key_row_input
                if key_row not in current_row_keys:
                    # 新しい行が必要 - メタデータをデフォルトで初期化し追加
                    need_expand = True
                    self.rows[key_row] = {
                        'unit': u.dimensionless_unscaled,
                        'name': key_row, # デフォルト名はキーを使用
                        'channel': ''
                    }
                    # 新しいキーリストを更新
                    current_row_keys = list(self.rows.keys())
                i_row = current_row_keys.index(key_row)
            elif isinstance(key_row_input, int):
                i_row = key_row_input
                if not (0 <= i_row < self.N_rows):
                    if i_row == self.N_rows: # 末尾への追加として拡張を許可
                        need_expand = True
                        # デフォルトのキーで新しい行メタデータを追加
                        key_row = f'row{i_row}' # 整数インデックスに対応するデフォルト文字列キーを生成
                        self.rows[key_row] = {
                            'unit': u.dimensionless_unscaled,
                            'name': key_row,
                            'channel': ''
                        }
                        current_row_keys = list(self.rows.keys()) # OrderedDictのキー順序を維持
                    else:
                        raise IndexError(f"Row index out of range: {i_row} (current max index is {self.N_rows - 1})")
                else: # 既存の整数インデックスの場合
                    key_row = current_row_keys[i_row] # 対応する既存のキーを取得
            else:
                raise TypeError("Row key must be str or int")

            # 列キーの処理 (行と同様のロジック)
            if isinstance(key_col_input, str):
                key_col = key_col_input
                if key_col not in current_col_keys:
                    # 新しい列が必要 - メタデータをデフォルトで初期化し追加
                    need_expand = True
                    self.cols[key_col] = {
                        'unit': u.dimensionless_unscaled,
                        'name': key_col, # デフォルト名はキーを使用
                        'channel': ''
                    }
                    # 新しいキーリストを更新
                    current_col_keys = list(self.cols.keys())
                i_col = current_col_keys.index(key_col)
            elif isinstance(key_col_input, int):
                i_col = key_col_input
                if not (0 <= i_col < self.N_cols):
                    if i_col == self.N_cols: # 末尾への追加として拡張を許可
                        need_expand = True
                        # デフォルトのキーで新しい列メタデータを追加
                        key_col = f'col{i_col}' # 整数インデックスに対応するデフォルト文字列キーを生成
                        self.cols[key_col] = {
                            'unit': u.dimensionless_unscaled,
                            'name': key_col,
                            'channel': ''
                        }
                        current_col_keys = list(self.cols.keys()) # OrderedDictのキー順序を維持
                    else:
                        raise IndexError(f"Column index out of range: {i_col} (current max index is {self.N_cols - 1})")
                else: # 既存の整数インデックスの場合
                    key_col = current_col_keys[i_col] # 対応する既存のキーを取得
            else:
                raise TypeError("Column key must be str or int")

            # サンプル数を確認 (初回代入時以外はN_samplesは固定)
            expected_samples = self.N_samples
            if isinstance(value, Series):
                value_samples = len(value)
                if expected_samples == 0: # 行列サイズはあるがサンプルが0の場合
                    expected_samples = value_samples # Seriesのサンプル数に合わせる
                    need_expand = True # サンプル軸の拡張が必要
                elif value_samples != expected_samples:
                    raise ValueError(f"Length of Series ({value_samples}) does not match expected N_samples ({expected_samples})")
            elif isinstance(value, u.Quantity):
                # Quantityがスカラまたはサンプル数と一致する長さの1D配列であることを確認
                if not np.isscalar(value.value) and value.value.ndim != 1:
                    raise ValueError(f"Quantity value must be scalar or 1D array, got {value.value.shape}")
                if not np.isscalar(value.value) and len(value.value) != expected_samples and expected_samples != 0:
                    raise ValueError(f"Length of Quantity ({len(value.value)}) does not match expected N_samples ({expected_samples})")
                if expected_samples == 0 and not np.isscalar(value.value):
                    expected_samples = len(value.value)
                    need_expand = True
                elif expected_samples == 0 and np.isscalar(value.value):
                    # スカラQuantityでサンプル数0の行列に代入する場合は、サンプル数1とするか検討。
                    # ここではエラーとせず、次の代入部分で np.full(0, val) となり空配列が代入されるようにする（NumPyの挙動に依存）
                    # より明示的にサンプル数1の行列として扱うならここで expected_samples = 1; need_expand = True とする
                    pass # 現在のロジックを維持

            elif isinstance(value, np.ndarray):
                if value.ndim != 1 or (value.shape[0] != expected_samples and expected_samples != 0):
                    raise ValueError(f"Numpy array must be 1D with shape ({expected_samples},), got {value.shape}")
                if expected_samples == 0:
                    expected_samples = value.shape[0]
                    need_expand = True
            elif np.isscalar(value):
                # スカラ値でサンプル数0の行列に代入する場合も同様に np.full(0, value)
                pass # 現在のロジックを維持
            else:
                raise TypeError(f"Unsupported type for assignment: {type(value)}")


            # --- 配列拡張が必要な場合、実行する ---
            if need_expand:
                # 新しい行列のサイズを決定
                new_rows = max(len(self.rows), i_row + 1)
                new_cols = max(len(self.cols), i_col + 1)
                new_samples = expected_samples # ここで確定したサンプル数を使用

                # 既存の値配列を確保
                old_value = self.value
                old_names = self.names
                old_units = self.units
                # 既存の次元を取得 (拡張前の shape)
                old_rows = old_value.shape[0]
                old_cols = old_value.shape[1]
                old_samples = old_value.shape[2]

                # 新しい配列を作成 (既存のdtypeを使用)
                self.value = np.zeros((new_rows, new_cols, new_samples), dtype=old_value.dtype)
                self.names = np.full((new_rows, new_cols), '', dtype=str)
                self.units = np.full((new_rows, new_cols), u.dimensionless_unscaled, dtype=object)

                # 既存データをコピー
                # コピー元の次元とコピー先の次元の小さい方を使用
                copy_rows = min(old_rows, new_rows)
                copy_cols = min(old_cols, new_cols)
                copy_samples = min(old_samples, new_samples) # サンプル軸のコピーサイズも考慮

                if copy_samples > 0 and copy_rows > 0 and copy_cols > 0: # コピー元のサイズが0でない場合のみコピー
                    self.value[:copy_rows, :copy_cols, :copy_samples] = old_value[:copy_rows, :copy_cols, :copy_samples]

                self.names[:copy_rows, :copy_cols] = old_names[:copy_rows, :copy_cols]
                self.units[:copy_rows, :copy_cols] = old_units[:copy_rows, :copy_cols]

                # サンプル数が変更された場合、x軸情報も更新 (初回設定と同様のロジック)
                if old_samples == 0 and new_samples > 0:
                    if isinstance(value, Series) and hasattr(value, 'xindex') and len(value.xindex) > 0:
                        if isinstance(value.xindex[0], u.Quantity):
                            self.x0 = value.xindex[0].value
                            self.xunit = value.xunit
                            self.dx = value.dx.value if hasattr(value.dx, 'value') else 1.0
                        else:
                            self.x0 = value.xindex[0]
                            self.xunit = u.dimensionless_unscaled
                            self.dx = value.dx if hasattr(value, 'dx') else 1.0
                    else: # Quantityやndarray、スカラの場合はデフォルトにリセット（または初期値のまま）
                        # ここでどのように扱うかは設計による。ここではデフォルトに戻す例
                        self.x0 = 0.0
                        self.dx = 1.0
                        self.xunit = u.dimensionless_unscaled


            # --- 値とメタデータの設定 ---

            if isinstance(value, Series):
                # Series値の設定とメタデータ更新
                self.value[i_row, i_col, :] = value.value
                self.units[i_row, i_col] = value.unit
                # Seriesの名前を要素のnameに設定
                self.names[i_row, i_col] = value.name or ''

                # Seriesのchannel情報を要素に紐づくメタデータとして格納することを検討
                # 例: self.rows[key_row].update({'channel': value.channel if hasattr(value, 'channel') and value.channel is not None else ''})
                # ただし、これは行全体ではなく特定の要素に紐づく情報なので、要素ごとのメタデータとして持つのが適切か、
                # またはrows/colsのメタデータは行/列全体の代表情報に限定するか、設計思想による。
                # 現状のコードでは要素ごとのnames/units/valueと、行/列ごとのrows/colsメタデータが分かれている。
                # Series.channelを行の'channel'メタデータに反映させる場合は、key_rowに紐づく情報としてrows辞書を更新する必要がある。
                if hasattr(value, 'channel') and value.channel is not None:
                    # 例として、要素のnameにチャンネル情報を追記
                    # if self.names[i_row, i_col]:
                    #     self.names[i_row, i_col] += f" ({value.channel})"
                    # else:
                    #     self.names[i_row, i_col] = str(value.channel)
                    pass # ここでどのように扱うか実装を決める

            elif isinstance(value, u.Quantity):
                val = value.value
                unit_to_assign = value.unit

                # 値の設定
                if np.isscalar(val):
                    if self.N_samples > 0: # サンプル数が1以上の場合はnp.fullで展開
                        self.value[i_row, i_col, :] = np.full(self.N_samples, val, dtype=self.value.dtype)
                    # サンプル数が0の場合は代入されない (NumPyの挙動)
                elif len(val) == self.N_samples:
                    self.value[i_row, i_col, :] = val
                else:
                    # ここには到達しないはずだが念のため
                    raise ValueError(f"Length of Quantity ({len(val)}) does not match N_samples ({self.N_samples})")
                # 単位の設定
                self.units[i_row, i_col] = unit_to_assign

            elif np.isscalar(value):
                if self.N_samples > 0: # サンプル数が1以上の場合はnp.fullで展開
                    self.value[i_row, i_col, :] = np.full(self.N_samples, value, dtype=self.value.dtype)
                # サンプル数が0の場合は代入されない (NumPyの挙動)
                self.units[i_row, i_col] = u.dimensionless_unscaled # スカラ代入時は単位なしにリセット

            elif isinstance(value, np.ndarray):
                if value.ndim == 1 and value.shape[0] == self.N_samples:
                    self.value[i_row, i_col, :] = value
                    self.units[i_row, i_col] = u.dimensionless_unscaled # ndarray代入時は単位なしにリセット
                else:
                    # ここには到達しないはずだが念のため
                    raise ValueError(f"Numpy array must be 1D with shape ({self.N_samples},), got {value.shape}")
            else:
                # 上記の型チェックでエラーにならなかったが、ここで処理できない型
                raise TypeError(f"Unsupported type for assignment: {type(value)}")

            # 行と列のメタデータを更新（キーが存在する場合のみ）
            # 注意: 整数インデックスで代入した場合、key_row/key_colは自動生成されたrowN/colMになります
            # ここで rows[key_row] や cols[key_col] のメタデータを更新するロジックを必要に応じて追加
            # 例: self.rows[key_row]['name'] = 'Updated Row Name'
            pass
        
    @staticmethod
    def _safe_xindex_summary(xindex, precision=3) -> str:
        if len(xindex) == 0:
            return "[empty]"
        elif len(xindex) == 1:
            return f"[{xindex[0]:.{precision}g}]"
        else:
            return f"[{xindex[0]:.{precision}g}, ..., {xindex[-1]:.{precision}g}]"



    def __delitem__(self, keys):
        """
        Clear the data at the specified (row_key, col_key).
        
        Note: This does not remove the row or column from the matrix structure,
        but rather sets its values to zero and resets metadata.
        """
        key_row, key_col = keys

        i_row = self.row_keys.index(key_row)
        i_col = self.col_keys.index(key_col)

        self.value[i_row, i_col, :] = 0  # Clear data
        self.units[i_row, i_col] = u.dimensionless_unscaled
        self.names[i_row, i_col] = ''

    def transpose(self):
        """Return a new SeriesMatrix with rows and columns transposed."""
        from copy import deepcopy
        transposed = deepcopy(self)
        transposed.value = transposed.value.transpose((1, 0, 2))
        transposed.names = transposed.names.T
        transposed.units = transposed.units.T
        transposed.rows, transposed.cols = deepcopy(self.cols), deepcopy(self.rows)
        return transposed

    @property
    def T(self):
        return self.transpose()

    def __iter__(self):
        """Iterate over all (row_key, col_key): Series pairs."""
        for i, rkey in enumerate(self.row_keys):
            for j, ckey in enumerate(self.col_keys):
                yield (rkey, ckey), self[rkey, ckey]


    def crop(self, start=None, end=None):
        """Return a cropped SeriesMatrix between `start` and `end` on x-axis."""
        if self.N_samples == 0:
            return self

        xvals = self.xindex.to_value(self.xunit)         
        mask = np.ones_like(xvals, dtype=bool)
        if start is not None:
            mask &= (xvals >= start)
        if end is not None:
            mask &= (xvals <= end)

        idx = np.where(mask)[0]
        if len(idx) == 0:
            raise ValueError("No samples in specified range")

        new_value = self.value[:, :, idx]
        new_x0 = xvals[idx[0]]
        return SeriesMatrix(new_value, x0=new_x0, dx=self.dx, xunit=self.xunit,
                             epoch=self.epoch, name=self.name, names=self.names,
                             units=self.units, rows=self.rows, cols=self.cols, 
                            attrs=self.attrs.copy())

    def shift(self, shift_amount):
        x0_new = self.x0 + shift_amount
        return SeriesMatrix(
            value=self.value,
            x0=x0_new,
            dx=self.dx,
            xunit=self.xunit,
            epoch=self.epoch,
            name=self.name,
            names=self.names,
            units=self.units,
            rows=self.rows,
            cols=self.cols,
            attrs=self.attrs.copy()
        )
    
    
    def append(self, other):
        """Append another SeriesMatrix with same shape in N, M and same dx/x0/xunit."""
        if not isinstance(other, SeriesMatrix):
            raise TypeError("Only SeriesMatrix can be appended")
        if (self.shape[:2] != other.shape[:2] or self.dx != other.dx or
            self.xunit != other.xunit or self.epoch != other.epoch):
            raise ValueError("Incompatible SeriesMatrix shapes or sampling")        

        self.value = np.concatenate((self.value, other.value), axis=2)
        # xindex はプロパティなので自動更新される
        
        # ユニット比較の改善
        for i in range(self.N_rows):
            for j in range(self.N_cols):
                if str(self.units[i, j]) == str(u.dimensionless_unscaled):
                    self.units[i, j] = other.units[i, j]
                if not self.names[i, j].strip():
                    self.names[i, j] = other.names[i, j]
                    
        return self

    def prepend(self, other):
        """Prepend another SeriesMatrix."""
        if not isinstance(other, SeriesMatrix):
            raise TypeError("Only SeriesMatrix can be prepended")
        if (self.shape[:2] != other.shape[:2] or self.dx != other.dx or
            self.xunit != other.xunit or self.epoch != other.epoch):
            raise ValueError("Incompatible SeriesMatrix shapes or sampling")

        self.value = np.concatenate((other.value, self.value), axis=2)
        self.x0 = other.x0
        # xindex はプロパティなので自動更新される
        
        # ユニット比較の改善
        for i in range(self.N_rows):
            for j in range(self.N_cols):
                if str(other.units[i, j]) != str(u.dimensionless_unscaled):
                    self.units[i, j] = other.units[i, j]
                if other.names[i, j].strip():
                    self.names[i, j] = other.names[i, j]
        
        return self

    def row_index(self, key):
        """Return the index of the given row key."""
        try:
            return list(self.rows.keys()).index(key)
        except ValueError:
            raise KeyError(f"Invalid row key: {key}")

    def col_index(self, key):
        """Return the index of the given column key."""
        try:
            return list(self.cols.keys()).index(key)
        except ValueError:
            raise KeyError(f"Invalid column key: {key}")

    def get_index(self, key_row, key_col):
        """Return the (i,j) index for given row and column keys."""
        return self.row_index(key_row), self.col_index(key_col)

    def sort_rows(self, by='key', reverse=False, key_order=None):
        """
        Sort row keys.
        - by='key': sort alphabetically by key
        - by='value': sort by row['name']
        - by='custom': use key_order as a list of keys in desired order
        """
        if by == 'key':
            new_keys = sorted(self.row_keys, reverse=reverse)
        elif by == 'value':
            new_keys = sorted(self.row_keys, key=lambda k: self.rows[k].get('name', ''), reverse=reverse)
        elif by == 'custom':
            if not key_order or set(key_order) != set(self.row_keys):
                raise ValueError("custom key_order must match existing row_keys exactly")
            new_keys = key_order
        else:
            raise ValueError(f"Unknown sort mode: {by}")
        self._reorder_axis('row', new_keys)

    def sort_cols(self, by='key', reverse=False, key_order=None):
        """
        Sort column keys.
        - by='key': sort alphabetically by key
        - by='value': sort by col['name']
        - by='custom': use key_order as a list of keys in desired order
        """
        if by == 'key':
            new_keys = sorted(self.col_keys, reverse=reverse)
        elif by == 'value':
            new_keys = sorted(self.col_keys, key=lambda k: self.cols[k].get('name', ''), reverse=reverse)
        elif by == 'custom':
            if not key_order or set(key_order) != set(self.col_keys):
                raise ValueError("custom key_order must match existing col_keys exactly")
            new_keys = key_order
        else:
            raise ValueError(f"Unknown sort mode: {by}")
        self._reorder_axis('col', new_keys)

    '''
    def value_at(self, x):
        x = u.Quantity(x, self.xunit).to_value(self.xunit)
        xvals = self.xindex.to_value(self.xunit)
        
        if np.isscalar(x):
            x = np.array([x])
        
        results = []
        for xi in x:
            if xi < xvals[0] or xi > xvals[-1]:
                raise ValueError("x is out of bounds")
            i = np.searchsorted(xvals, xi) - 1
            i = np.clip(i, 0, self.N_samples - 2)
            frac = (xi - xvals[i]) / (xvals[i+1] - xvals[i])
            v0 = self.value[:, :, i]
            v1 = self.value[:, :, i+1]
            results.append((1-frac)*v0 + frac*v1)
        
        return np.stack(results, axis=-1)  # shape: (N, M, len(x))
    '''


    def copy_structure(self, rows=None, cols=None):
        rows = rows or self.row_keys
        cols = cols or self.col_keys
        new = SeriesMatrix(
            value=np.zeros((len(rows), len(cols), self.N_samples), dtype=self.value.dtype),
            names=[["" for _ in cols] for _ in rows],
            units=[[u.dimensionless_unscaled for _ in cols] for _ in rows],
            x0=self.x0, dx=self.dx, xunit=self.xunit, epoch=self.epoch,
            rows={k: {} for k in rows}, cols={k: {} for k in cols}, attrs=self.attrs.copy()
        )
        return new

    def get(self, row, col):
        try:
            return self[row, col]
        except KeyError:
            return None

    def zero_series_like(self, ref):
        """Create a Series-like object with zero values, preserving metadata."""
        return type(ref)(
            np.zeros_like(ref.value),
            xindex=ref.xindex,
            unit=ref.unit,
            name=ref.name,
            epoch=getattr(ref, "epoch", None),
            channel=getattr(ref, "channel", None),
        )

    def _reorder_axis(self, axis: str, new_keys: list):
        """
        Reorder internal arrays and metadata along a given axis ('row' or 'col') using new_keys.
        """
        if axis == 'row':
            idx = [self.row_keys.index(k) for k in new_keys]
            self.value = self.value[idx, :, :]
            self.names = self.names[idx, :]
            self.units = self.units[idx, :]
            self.rows = OrderedDict((k, self.rows[k]) for k in new_keys)
        elif axis == 'col':
            idx = [self.col_keys.index(k) for k in new_keys]
            self.value = self.value[:, idx, :]
            self.names = self.names[:, idx]
            self.units = self.units[:, idx]
            self.cols = OrderedDict((k, self.cols[k]) for k in new_keys)
        else:
            raise ValueError(f"Invalid axis: {axis}")
    
    
    
 

    ##### Elementary arithmetic #####
    def __add__(self, other):
        return self._add_sub(other, op='+')

    def __sub__(self, other):
        return self._add_sub(other, op='-')

    def _add_sub(self, other, op='+'):
        if not isinstance(other, SeriesMatrix):
            raise TypeError("Addition/Subtraction only supported between SeriesMatrix instances")

        # rows, colsの和集合を取る
        row_keys = sorted(set(self.row_keys).union(other.row_keys))
        col_keys = sorted(set(self.col_keys).union(other.col_keys))

        new = self.copy_structure(rows=row_keys, cols=col_keys)

        for i, rkey in enumerate(row_keys):
            for j, ckey in enumerate(col_keys):
                sval = self.get(rkey, ckey)
                oval = other.get(rkey, ckey)

                if sval is None:
                    sval = self.zero_series_like(oval)
                if oval is None:
                    oval = self.zero_series_like(sval)

                try:
                    oval = oval.to(sval.unit)
                except Exception:
                    try:
                        sval = sval.to(oval.unit)
                    except Exception:
                        raise ValueError(f"Unit mismatch for ({rkey}, {ckey})")

                result_value = sval.value + oval.value if op == '+' else sval.value - oval.value
                new_series = type(sval)(result_value, xindex=sval.xindex, unit=sval.unit, name=sval.name)
                if hasattr(sval, 'channel') and sval.channel is not None:
                    channel_str = str(sval.channel) if hasattr(sval.channel, '__str__') else ''
                    # channel が初期化時のパラメータとして使えるか確認が必要
                    # 使えない場合は後から設定
                    try:
                        new_series.channel = channel_str
                    except:
                        pass
                
                new[rkey, ckey] = new_series
        return new

    def __mul__(self, other):
        return self._mul_div(other, op='*')

    def __truediv__(self, other):
        return self._mul_div(other, op='/')

    def _mul_div(self, other, op='*'):
        from copy import deepcopy

        if np.isscalar(other) or isinstance(other, u.Quantity):
            factor = other.value if isinstance(other, u.Quantity) else other
            factor_unit = other.unit if isinstance(other, u.Quantity) else 1
            new = deepcopy(self)
            for i in range(self.N_rows):
                for j in range(self.N_cols):
                    new.value[i, j, :] = (self.value[i, j, :] * factor) if op == '*' else (self.value[i, j, :] / factor)
                    if isinstance(factor_unit, u.UnitBase):
                        new.units[i, j] = (self.units[i, j] * factor_unit) if op == '*' else (self.units[i, j] / factor_unit)
            return new

        if isinstance(other, SeriesMatrix):
            if self.shape[:2] != other.shape[:2]:
                raise ValueError("Shape mismatch for multiplication/division")
            new = self.copy_structure()
            for i in range(self.N_rows):
                for j in range(self.N_cols):
                    sval = self.value[i, j, :]
                    oval = other.value[i, j, :]
                    result = sval * oval if op == '*' else sval / oval
                    sun = self.units[i, j]
                    oun = other.units[i, j]
                    result_unit = (sun * oun) if op == '*' else (sun / oun)
                    new.value[i, j, :] = result
                    new.units[i, j] = result_unit
            return new

        raise TypeError("Unsupported type for multiplication/division")   
    
    def __pow__(self, other):
        """Calculate the element-wise power of the SeriesMatrix."""
        exponent = other

        # べき乗の指数はスカラ数値または無次元のQuantityである必要がある
        if isinstance(exponent, u.Quantity):
            if not exponent.isscalar:
                raise TypeError("Exponent Quantity must be a scalar.")
            if not exponent.unit.is_equivalent(u.dimensionless_unscaled):
                raise u.UnitsError("Exponent Quantity must be dimensionless.")
            # 無次元Quantityの場合は値を取り出す
            exponent_value = exponent.value
        elif np.isscalar(exponent) or isinstance(exponent, (int, float, complex)):
            # スカラ数値の場合はそのまま使用
            exponent_value = exponent
        else:
            raise TypeError("Exponent must be a scalar number or a dimensionless Quantity.")

        # 値配列の各要素をべき乗する
        # np.power は要素ごとのべき乗を計算する
        try:
            result_value = np.power(self.value, exponent_value)
        except Exception as e:
            # べき乗計算中に発生する可能性のあるエラーを捕捉 (例: 負の数の分数べき乗)
            raise ValueError(f"Error during element-wise power calculation: {e}")

        # 単位の各要素をべき乗する
        result_units = np.empty_like(self.units, dtype=object)
        for i in range(self.N_rows):
            for j in range(self.N_cols):
                original_unit = self.units[i, j]
                if isinstance(original_unit, u.Unit):
                    try:
                        # 単位のべき乗は Astropy Units が処理する
                        result_units[i, j] = original_unit**exponent_value
                    except Exception as e:
                         # 単位のべき乗計算中に発生する可能性のあるエラーを捕捉
                         raise ValueError(f"Error calculating unit power for element ({i}, {j}): {e}")
                else:
                    # 元の単位が Unit でない場合は、結果は無次元とする
                    result_units[i, j] = u.dimensionless_unscaled

        # 結果を新しい SeriesMatrix インスタンスとして作成
        new_matrix = self.__class__(
            value=result_value,
            x0=self.x0, # x軸情報は引き継ぐ
            dx=self.dx,
            xunit=self.xunit,
            epoch=self.epoch,
            name=f"({self.name})**{exponent}" if self.name else f"SeriesMatrix**{exponent}", # 名前に指数を含める (Quantityの場合はstrに変換)
            names=self.names.copy(), # 名前は通常べき乗で変わらないためコピー
            units=result_units, # 計算結果の単位を使用
            rows=self.rows.copy(), # 行メタデータをコピー
            cols=self.cols.copy(), # 列メタデータをコピー
            attrs=self.attrs.copy() if self.attrs is not None else None, # attrsをコピー
            dtype=result_value.dtype # 結果の値の dtype を使用
        )
        return new_matrix
    
    
    ##### Ststistics (scalar) #####
    def _create_parameter_matrix(self, value, units, operation_name="Result"):
        """
        Helper method to create a ParameterMatrix from calculation results,
        copying relevant metadata from this SeriesMatrix.
        """
        # Ensure the input value is 2D (N_rows, N_cols)
        if value.shape != (self.N_rows, self.N_cols):
            raise ValueError(f"Input value shape {value.shape} is not compatible with SeriesMatrix shape {(self.N_rows, self.N_cols)} for ParameterMatrix creation.")

        # Ensure the input units shape matches the value shape
        if units.shape != (self.N_rows, self.N_cols):
             raise ValueError(f"Input units shape {units.shape} is not compatible with SeriesMatrix shape {(self.N_rows, self.N_cols)} for ParameterMatrix creation.")

        # Create a new ParameterMatrix instance
        new_param_matrix = ParameterMatrix(
            value=value,
            rows=self.rows.copy(),          # 行メタデータをコピー
            cols=self.cols.copy(),          # 列メタデータをコピー
            epoch=self.epoch,               # epochをコピー
            names=self.names.copy(),        # 要素の名前をコピー
            units=units,                    # 計算結果の単位を使用
            attrs=self.attrs.copy() if self.attrs is not None else None, # attrsをコピー
            name=f"{operation_name} of {self.name}" if self.name else f"{operation_name} SeriesMatrix" # 名前を設定
        )
        return new_param_matrix
    
    def mean(self):
        mean_value = np.mean(self.value, axis=2)
        return self._create_parameter_matrix(value=mean_value,  units=self.units.copy(), operation_name="Mean")
    
    def nanmean(self):
        mean_value = np.nanmean(self.value, axis=2)
        return self._create_parameter_matrix(value=mean_value,  units=self.units.copy(), operation_name="Mean")
    
    def std(self):
        std_value = np.std(self.value, axis=2)
        return self._create_parameter_matrix(value=std_value, units=self.units.copy(), operation_name="Std Dev")
    
    def nanstd(self):
        std_value = np.nanstd(self.value, axis=2)
        return self._create_parameter_matrix(value=std_value, units=self.units.copy(), operation_name="Std Dev")

    def var(self):
        variance_matrix =  self.std()**2
        variance_matrix.name = f"Variance of {self.name}" if self.name else "Variance SeriesMatrix"
        return variance_matrix
    
    def nanvar(self):
        variance_matrix =  self.nanstd()**2
        variance_matrix.name = f"Variance of {self.name}" if self.name else "Variance SeriesMatrix"
        return variance_matrix

    def min(self):
        min_value = np.min(self.value, axis=2)
        return self._create_parameter_matrix(value=min_value, units=self.units.copy(), operation_name="Min")
    
    def nanmin(self):
        min_value = np.nanmin(self.value, axis=2)
        return self._create_parameter_matrix(value=min_value, units=self.units.copy(), operation_name="Min")

    def max(self):
        max_value = np.max(self.value, axis=2)
        return self._create_parameter_matrix(value=max_value, units=self.units.copy(), operation_name="Max")
    
    def nanmax(self):
        max_value = np.nanmax(self.value, axis=2)
        return self._create_parameter_matrix(value=max_value, units=self.units.copy(), operation_name="Max")

    def sum(self):
        sum_value = np.sum(self.value, axis=2)
        return self._create_parameter_matrix(value=sum_value, units=self.units.copy(), operation_name="Sum")
    
    def nansum(self):
        sum_value = np.nansum(self.value, axis=2)
        return self._create_parameter_matrix(value=sum_value, units=self.units.copy(), operation_name="Sum")

    def median(self):
        median_value = np.median(self.value, axis=2)
        return self._create_parameter_matrix(value=median_value, units=self.units.copy(), operation_name="Median")
    
    def nanmedian(self):
        median_value = np.nanmedian(self.value, axis=2)
        return self._create_parameter_matrix(value=median_value, units=self.units.copy(), operation_name="Median")

    def rms(self):
        rms_value = np.sqrt(np.mean(np.square(self.value), axis=2))
        return self._create_parameter_matrix(value=rms_value, units=self.units.copy(), operation_name="RMS")       
    
    def nanrms(self):
        rms_value = np.sqrt(np.nanmean(np.square(self.value), axis=2))
        return self._create_parameter_matrix(value=rms_value, units=self.units.copy(), operation_name="RMS")       
    
 
    ##### some calc #####   
    def diff(self, n: int = 1, axis: int = 2):
            """
            Calculate the n-th order discrete difference along the specified axis.
            Units are kept the same, following Gwpy's convention.
            """
            if axis != 2:
                # 現在はサンプル軸方向の差分に特化
                raise NotImplementedError("Differentiation is currently only supported along the sample axis (axis=2).")

            if not isinstance(n, int) or n < 1:
                raise ValueError("n must be a positive integer.")

            if self.N_samples <= n:
                # 差分を取るのに十分なサンプルがない場合
                warnings.warn(f"Not enough samples ({self.N_samples}) to compute {n}-th order difference along axis {axis}. Returning empty SeriesMatrix.")
                # 空の SeriesMatrix を返すか、エラーとするかは設計による
                return self.__class__(
                    value=np.empty((self.N_rows, self.N_cols, 0), dtype=self.value.dtype),
                    x0=self.x0, dx=self.dx, xunit=self.xunit, epoch=self.epoch,
                    name=f"Diff({self.name}, n={n})" if self.name else f"Diff(SeriesMatrix, n={n})",
                    names=self.names.copy(),
                    units=self.units.copy(), # 単位はそのままコピー
                    rows=self.rows.copy(), cols=self.cols.copy(),
                    attrs=self.attrs.copy() if self.attrs is not None else None
                )


            # 値の差分計算
            diff_value = np.diff(self.value, n=n, axis=axis)

            # 単位の計算: Gwpyの慣習に従い、元の単位を維持する
            resulting_units = self.units.copy()


            # x軸情報の更新
            # 新しいサンプル数
            new_N_samples = self.N_samples - n
            # 新しい開始点 (最初のn個のサンプルをスキップするため)
            # x0 は Quantity であることを想定し、dx も Quantity として加算
            # ただし、diff後のデータのx座標の定義は文脈によるため、Gwpy TimeSeriesのdiffの挙動を確認推奨
            # ここではシンプルにn * dxだけシフトすると仮定
            new_x0 = self.x0 + n * self.dx


            # 結果を新しい SeriesMatrix インスタンスとして作成
            new_matrix = self.__class__(
                value=diff_value,
                x0=new_x0,
                dx=self.dx, # サンプリング間隔は同じ
                xunit=self.xunit, # x軸の単位は同じ
                epoch=self.epoch, # epochは同じ
                name=f"Diff({self.name}, n={n})" if self.name else f"Diff(SeriesMatrix, n={n})",
                names=self.names.copy(),
                units=resulting_units, # 元の単位と同じ
                rows=self.rows.copy(),
                cols=self.cols.copy(),
                attrs=self.attrs.copy() if self.attrs is not None else None,
                dtype=diff_value.dtype # 結果の値の dtype を使用
            )
            return new_matrix

    def cumsum(self, axis: int = 2, dtype=None, out=None):
            """
            Calculate the cumulative sum along the specified axis.
            For axis=2 (sample axis), this approximates the discrete integration.
            Units are kept the same, following Gwpy's convention.
            """
            if axis != 2:
                # 現在はサンプル軸方向の累積和に特化
                raise NotImplementedError("Cumulative sum is currently only supported along the sample axis (axis=2).")

            # 値の累積和計算
            cumsum_value = np.cumsum(self.value, axis=axis, dtype=dtype, out=out)

            # 単位の計算: 元の単位と同じ
            resulting_units = self.units.copy() # 単位は変わらないためコピー

            # 結果を新しい SeriesMatrix インスタンスとして作成
            new_matrix = self.__class__(
                value=cumsum_value,
                x0=self.x0, # x軸情報は同じ
                dx=self.dx,
                xunit=self.xunit,
                epoch=self.epoch,
                name=f"CumSum({self.name})" if self.name else "CumSum SeriesMatrix",
                names=self.names.copy(),
                units=resulting_units,
                rows=self.rows.copy(),
                cols=self.cols.copy(),
                attrs=self.attrs.copy() if self.attrs is not None else None,
                dtype=cumsum_value.dtype # 結果の値の dtype を使用
            )
            return new_matrix 
    
    def rolling_mean(self, window: int, min_periods: Optional[int] = None):
        """
        Calculate the rolling mean along the sample axis (axis=2).
        Uses bottleneck.move_mean for efficiency.
        Units are kept the same.
        """
        if bn is None:
             raise ImportError("Bottleneck is required for rolling_mean. Please install it (`pip install bottleneck`).")

        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window size must be a positive integer.")

        if min_periods is None:
            min_periods = window # デフォルトはウィンドウサイズと同じ

        if not isinstance(min_periods, int) or min_periods < 0 or min_periods > window:
             raise ValueError("min_periods must be a non-negative integer less than or equal to window.")

        if self.N_samples < window:
            warnings.warn(f"Window size ({window}) is larger than the number of samples ({self.N_samples}). Returning SeriesMatrix filled with NaN.")
            # ウィンドウサイズよりサンプルが少ない場合は NaN で埋める
            nan_value = np.full_like(self.value, np.nan)
            return self.__class__(
                value=nan_value,
                x0=self.x0, dx=self.dx, xunit=self.xunit, epoch=self.epoch,
                name=f"RollingMean({self.name}, window={window})" if self.name else f"RollingMean(SeriesMatrix, window={window})",
                names=self.names.copy(),
                units=self.units.copy(),
                rows=self.rows.copy(), cols=self.cols.copy(),
                attrs=self.attrs.copy() if self.attrs is not None else None
            )


        # 値の移動平均計算
        # bottleneck.move_mean を使用
        # unitsは自動的にnumpyarrayの操作で伝播されるが、ここでは明示的に指定しない（bottleneckの仕様による）
        # 結果の単位は元の単位と同じになる
        rolling_mean_value = bn.move_mean(
            self.value,
            window=window,
            axis=2,
            min_count=min_periods
        )

        # 単位の計算: 元の単位と同じ
        resulting_units = self.units.copy()

        # 結果を新しい SeriesMatrix インスタンスとして作成
        new_matrix = self.__class__(
            value=rolling_mean_value,
            x0=self.x0, # x軸情報は同じ
            dx=self.dx,
            xunit=self.xunit,
            epoch=self.epoch,
            name=f"RollingMean({self.name}, window={window})" if self.name else f"RollingMean(SeriesMatrix, window={window})",
            names=self.names.copy(),
            units=resulting_units,
            rows=self.rows.copy(),
            cols=self.cols.copy(),
            attrs=self.attrs.copy() if self.attrs is not None else None,
            dtype=rolling_mean_value.dtype # 結果の値の dtype を使用
        )
        return new_matrix
 
    
    
    ##### Visualizations #####
    def __repr__(self):
        xindex_summary = self._safe_xindex_summary(self.xindex)
        info = (
            f"SeriesMatrix(shape={self.shape}, xunit={self.xunit}, name='{self.name}')\n"
            f"  epoch   : {self.epoch}\n"
            f"  x0      : {self.x0}\n"
            f"  dx      : {self.dx}\n"
            f"  xindex  : {xindex_summary} ({self.N_samples} samples)\n"
            f"  xunit   : {self.xunit}\n"
        )

        # rows metadata
        df_rows = pd.DataFrame.from_dict(self.rows, orient='index')
        df_cols = pd.DataFrame.from_dict(self.cols, orient='index')

        info += "\n[ Row metadata ]\n" + df_rows.to_string()
        info += "\n\n[ Column metadata ]\n" + df_cols.to_string()

        if self.names is not None:
            df_names = pd.DataFrame(self.names, index=self.row_keys, columns=self.col_keys)
            info += "\n\n[ Channel names ]\n" + df_names.to_string()

        if self.units is not None:
            df_units = pd.DataFrame(self.units, index=self.row_keys, columns=self.col_keys)
            info += "\n\n[ Channel units ]\n" + df_units.astype(str).to_string()
            
        if self.attrs:
            info += "\n[ Attributes ]\n" + str(self.attrs)

        return info

    def _repr_html_(self):
        xindex_summary = self._safe_xindex_summary(self.xindex)
        html = f"<h3>SeriesMatrix: shape={self.shape}, xunit={escape(str(self.xunit))}, name='{escape(self.name)}'</h3>"
        html += f"<ul>"
        html += f"<li><b>epoch:</b> {escape(str(self.epoch))}</li>"
        html += f"<li><b>x0:</b> {self.x0}, <b>dx:</b> {self.dx}, <b>N_samples:</b> {self.N_samples}</li>"
        html += f"<li><b>xindex:</b> {xindex_summary}</li>"
        html += f"</ul>"

        df_rows = pd.DataFrame.from_dict(self.rows, orient='index')
        df_cols = pd.DataFrame.from_dict(self.cols, orient='index')
        df_names = pd.DataFrame(self.names, index=self.row_keys, columns=self.col_keys) if self.names is not None else None
        df_units = pd.DataFrame(self.units, index=self.row_keys, columns=self.col_keys) if self.units is not None else None

        html += "<h4>Row Metadata</h4>" + df_rows.to_html()
        html += "<h4>Column Metadata</h4>" + df_cols.to_html()
        if df_names is not None:
            html += "<h4>Channel Names</h4>" + df_names.to_html()
        if df_units is not None:
            html += "<h4>Channel Units</h4>" + df_units.astype(str).to_html()
        if self.attrs:
            html += "<h4>Attributes</h4><pre>" + escape(str(self.attrs)) + "</pre>"

        return html

    

    def plot(self,
             row=None, col=None,
             subplots=True, sharex=True, grid=True,
             legend='channels',
             color=None, linestyle='-', alpha=1.0, linewidth=1.5, marker=None
             ):
        """Plot selected rows and columns of the SeriesMatrix."""
        import matplotlib.pyplot as plt

        rows = [row] if isinstance(row, str) else (row if row is not None else self.row_keys)
        cols = [col] if isinstance(col, str) else (col if col is not None else self.col_keys)

        nrow = len(rows)
        ncol = len(cols)

        if subplots:
            fig, axes = plt.subplots(nrow, ncol, sharex=sharex, figsize=(4*ncol, 2.5*nrow))
            if nrow == 1 and ncol == 1:
                axes = np.array([[axes]])
            elif nrow == 1:
                axes = axes[np.newaxis, :]
            elif ncol == 1:
                axes = axes[:, np.newaxis]
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            axes = np.full((nrow, ncol), ax)

        x = self.xindex.to_value(self.xunit)
        xunit = self.xunit.to_string()

        for i, rkey in enumerate(rows):
            for j, ckey in enumerate(cols):
                ax = axes[i, j]
                series = self[rkey, ckey]
                y = series.value.real

                kwargs = dict(
                    linestyle=linestyle, alpha=alpha, linewidth=linewidth
                )
                if isinstance(color, list):
                    kwargs['color'] = color[(i * ncol + j) % len(color)]
                elif color is not None:
                    kwargs['color'] = color
                if marker is not None:
                    kwargs['marker'] = marker

                ax.plot(x, y, **kwargs)

                if i == nrow - 1:
                    ax.set_xlabel(f"x [{xunit}]")
                if j == 0:
                    ax.set_ylabel(f"[{series.unit}]")
                if i == 0:
                    ax.set_title(str(ckey))
                if j == 0:
                    ax.annotate(str(rkey), xy=(-0.15, 0.5), xycoords='axes fraction', ha='right', va='center', rotation=90)

                if legend == 'names':
                    label = series.name
                else:  # Channel オブジェクトを文字列に変換
                    label = str(series.channel) if hasattr(series.channel, '__str__') else repr(series.channel)
                
                ax.legend([label], loc='upper right')

                if grid:
                    ax.grid(True)

        plt.tight_layout()
        return fig, axes

    ##### I/O finctions #####
    def write(self, filename, format='hdf5', **kwargs):
        """Write the SeriesMatrix to a file."""
        if format.lower() != 'hdf5':
            raise NotImplementedError(f"Only 'hdf5' format is currently supported, not {format}")

        import h5py
        with h5py.File(filename, 'w', **kwargs) as f:
            f.create_dataset('value', data=self.value)
            f.create_dataset('names', data=self.names.astype('S'))
            f.create_dataset('units', data=np.array([[str(u) for u in row] for row in self.units], dtype='S'))
            f.attrs['x0'] = self.x0
            f.attrs['dx'] = self.dx
            f.attrs['xunit'] = str(self.xunit)
            f.attrs['epoch'] = float(self.epoch)
            f.create_dataset('rows', data=np.array(list(self.rows.keys()), dtype='S'))
            f.create_dataset('cols', data=np.array(list(self.cols.keys()), dtype='S'))
            attrs_grp = f.create_group('attrs')
            for k, v in self.attrs.items():
                attrs_grp.attrs[k] = str(v)
        return filename

    @classmethod
    def read(cls, filename, format='hdf5', **kwargs):
        """Read a SeriesMatrix from a file."""
        if format.lower() != 'hdf5':
            raise NotImplementedError(f"Only 'hdf5' format is currently supported, not {format}")

        import h5py
        with h5py.File(filename, 'r', **kwargs) as f:
            value = f['value'][()]
            names = f['names'][()].astype(str)
            raw_units = f['units'][()].astype(str)
            units = np.array([[u.Unit(u_) for u_ in row] for row in raw_units], dtype=object)
            x0 = f.attrs['x0']
            dx = f.attrs['dx']
            xunit = u.Unit(f.attrs['xunit'])
            epoch = f.attrs['epoch']
            rows = {k if isinstance(k, str) else k.decode(): {} for k in f['rows'][()]}
            cols = {k if isinstance(k, str) else k.decode(): {} for k in f['cols'][()]}
            attrs = {k: v for k, v in f['attrs'].attrs.items()}
        return cls(value=value, names=names, units=units,
                   x0=x0, dx=dx, xunit=xunit, epoch=epoch,
                   rows=rows, cols=cols, attrs=attrs)
        
        
class ParameterMatrix(SeriesMatrix):
    """Matrix of scalar parameters for a set of channels.
    Internal data: 3D array (N_rows, N_cols, 1), but behaves as 2D.
    """
    def __init__(
        self,
        value,
        rows=None,
        cols=None,
        epoch=None,
        name="",         # 追加: ParameterMatrix自身の名前を受け取る
        attrs=None       # 追加: その他のメタ情報を受け取る
    ):
        # Ensure value is 2D array
        value = np.atleast_2d(value)
        # Internally store as (N, M, 1)
        value = value[:, :, np.newaxis]

        # ParameterMatrixはサンプル軸サイズが1固定なので、x軸情報は固定値を渡す
        # nameとattrsを基底クラスに渡すように修正
        super().__init__(
            value,
            rows=rows,
            cols=cols,
            epoch=epoch,
            name=name,   # 渡す
            attrs=attrs, # 渡す
            x0=0.0,      # 固定
            dx=1.0,      # 固定
            xunit=u.dimensionless_unscaled, # 固定
            dtype=value.dtype # 入力valueのdtypeを使用
        )

    @property
    def value(self):
        """2D array view (N_rows, N_cols)"""
        return self._value[:, :, 0]

    @value.setter
    def value(self, val):
        val = np.atleast_2d(val)
        # setterでもサンプル軸サイズを1に固定することを保証
        if val.ndim == 2:
             self._value = val[:, :, np.newaxis]
        elif val.ndim == 3 and val.shape[2] == 1:
             self._value = val
        else:
             raise ValueError(f"Cannot set value with shape {val.shape} for ParameterMatrix. Expected 2D array or 3D array with shape (N, M, 1).")
        
        

    def __getitem__(self, keys):
            key_row_input, key_col_input = keys

            # --- キーを行列インデックスに変換 ---
            # SeriesMatrixのget_indexメソッドを利用してインデックスを取得
            try:
                i_row, i_col = self.get_index(key_row_input, key_col_input)
            except KeyError as e:
                # 指定されたキーが行列に存在しない場合はKeyErrorを発生させる
                raise KeyError(f"Access failed: {e}. Keys must correspond to existing rows/cols.")
            except TypeError:
                # get_indexがサポートしないキーの型の場合
                raise TypeError(f"Invalid keys for access: {keys}. Keys must be strings or integers corresponding to existing rows/cols.")

            # --- 値と単位を取得して返す ---
            # 内部の値配列から、指定された行と列の要素（サンプル軸の最初の要素）を取り出す
            val = self.value[i_row, i_col] # ParameterMatrixの@property valueは(N, M)ビューを返す
            
            # その要素に対応する単位を取得
            unit = self.units[i_row, i_col]
            
            # 単位があればQuantityとして、なければスカラとして返す
            if isinstance(unit, u.Unit):
                return val * unit
            else:
                return val # 単位なしの場合はスカラとして返す


    def __setitem__(self, keys, value):
        key_row_input, key_col_input = keys

        # ParameterMatrixはサンプル軸サイズが1であることを確認
        if self.N_samples != 1:
            # 通常 ParameterMatrix は __init__ で N_samples=1 に初期化されるが、念のためチェック
            raise RuntimeError("ParameterMatrix must have N_samples == 1")

        # キーを行列インデックスに変換
        # SeriesMatrixのget_indexメソッドを利用
        try:
            i_row, i_col = self.get_index(key_row_input, key_col_input)
        except KeyError as e:
            # 指定されたキーが行列に存在しない場合、KeyErrorを発生させる
            # ParameterMatrixではSeriesMatrixのような自動拡張は行わない設計とする
            raise KeyError(f"Assignment failed: {e}. ParameterMatrix does not support automatic expansion.")
        except TypeError:
            # get_indexがサポートしないキーの型の場合
            raise TypeError(f"Invalid keys for assignment: {keys}. Keys must be strings or integers corresponding to existing rows/cols.")


        # 代入される値の処理
        if isinstance(value, Series):
            # Seriesの場合は、サンプル数が1であることと、対応するx軸情報を持っているかチェック
            if len(value) != 1:
                raise ValueError(f"Cannot assign Series with {len(value)} samples to ParameterMatrix element (expected 1 sample).")
            # Option: Seriesのx軸情報 (x0, dx, xunit, epoch) がParameterMatrixと互換性があるかチェックしても良い
            # 例えば value.xunit == self.xunit など

            val_to_assign = value.value[0] # 最初のサンプルのみ取得
            unit_to_assign = value.unit
            name_to_assign = value.name or ''
            channel_to_assign = value.channel if hasattr(value, 'channel') and value.channel is not None else ''

        elif isinstance(value, u.Quantity):
            if np.isscalar(value.value):
                val_to_assign = value.value
                unit_to_assign = value.unit
                name_to_assign = '' # Quantity自体に名前やチャンネル情報はないと仮定
                channel_to_assign = ''
            elif len(value.value) == 1: # 長さ1のQuantity配列も許容
                val_to_assign = value.value[0]
                unit_to_assign = value.unit
                name_to_assign = ''
                channel_to_assign = ''
            else:
                raise ValueError(f"Cannot assign Quantity with {len(value.value)} elements to ParameterMatrix element (expected 1 element).")

        elif np.isscalar(value):
            val_to_assign = value
            unit_to_assign = u.dimensionless_unscaled # スカラには単位なしと仮定
            name_to_assign = ''
            channel_to_assign = ''

        elif isinstance(value, np.ndarray) and value.ndim == 0: # NumPy scalar
             val_to_assign = value.item() # Python scalar に変換
             unit_to_assign = u.dimensionless_unscaled
             name_to_assign = ''
             channel_to_assign = ''

        else:
            raise TypeError(f"Unsupported type for assignment to ParameterMatrix element: {type(value)}. Expected scalar, Quantity (scalar or length 1), or Series (length 1).")


    @property
    def T(self):
        """Transpose the matrix."""
        new = self.copy()
        new._value = np.transpose(self._value, axes=(1, 0, 2))
        new.rows, new.cols = self.cols, self.rows
        return new

    def copy_structure(self):
        """Create an empty ParameterMatrix with same structure."""
        new = self.__class__(
            np.zeros((self.N_rows, self.N_cols)),
            rows=self.rows.copy(),
            cols=self.cols.copy(),
            epoch=self.epoch
        )
        return new


    def sort_rows(self, by='key', reverse=False, key_order=None):
            """
            Sort row keys.
            - by='key': sort alphabetically by key
            - by='name': sort by row['name']
            - by='custom': use key_order as a list of keys in desired order
            """
            current_keys = list(self.rows.keys())

            if by == 'key':
                new_keys = sorted(current_keys, reverse=reverse)
            elif by == 'name':
                # 'name'メタデータに基づいてソートするためのキーリストを作成
                # 該当するnameがない場合は空文字列として扱う
                new_keys = sorted(current_keys, key=lambda k: self.rows[k].get('name', ''), reverse=reverse)
            elif by == 'custom':
                if not key_order:
                    raise ValueError("key_order must be provided when by='custom'")
                if set(key_order) != set(current_keys):
                    raise ValueError("custom key_order must match existing row_keys exactly")
                new_keys = key_order
            else:
                raise ValueError(f"Unknown sort mode: {by}. Supported modes are 'key', 'name', 'custom'.")

            # 新しいキーの順序に対応する元のインデックスを取得
            idx = [current_keys.index(k) for k in new_keys]

            # 新しいインスタンスを作成し、データを並べ替える
            new = self.__class__(
                value=self._value[idx, :, :],  # 行方向に並べ替え
                rows=OrderedDict((k, self.rows[k]) for k in new_keys), # OrderedDictを新しい順序で再構築
                cols=self.cols.copy(), # colsはコピー
                epoch=self.epoch,
                # その他のメタデータも適切にコピーまたは並べ替えが必要
                # names, unitsはvalueと同時に並べ替える
                names=self.names[idx, :],
                units=self.units[idx, :],
                # attrsはコピー
                attrs=self.attrs.copy() if self.attrs is not None else None
            )
            return new


    def sort_cols(self, by='key', reverse=False, key_order=None):
        """
        Sort column keys.
        - by='key': sort alphabetically by key
        - by='name': sort by col['name']
        - by='custom': use key_order as a list of keys in desired order
        """
        current_keys = list(self.cols.keys())

        if by == 'key':
            new_keys = sorted(current_keys, reverse=reverse)
        elif by == 'name':
            # 'name'メタデータに基づいてソートするためのキーリストを作成
            # 該当するnameがない場合は空文字列として扱う
            new_keys = sorted(current_keys, key=lambda k: self.cols[k].get('name', ''), reverse=reverse)
        elif by == 'custom':
            if not key_order:
                raise ValueError("key_order must be provided when by='custom'")
            if set(key_order) != set(current_keys):
                raise ValueError("custom key_order must match existing col_keys exactly")
            new_keys = key_order
        else:
            raise ValueError(f"Unknown sort mode: {by}. Supported modes are 'key', 'name', 'custom'.")

        # 新しいキーの順序に対応する元のインデックスを取得
        idx = [current_keys.index(k) for k in new_keys]

        # 新しいインスタンスを作成し、データを並べ替える
        new = self.__class__(
            value=self._value[:, idx, :],  # 列方向に並べ替え
            rows=self.rows.copy(), # rowsはコピー
            cols=OrderedDict((k, self.cols[k]) for k in new_keys), # OrderedDictを新しい順序で再構築
            epoch=self.epoch,
            # その他のメタデータも適切にコピーまたは並べ替えが必要
            # names, unitsはvalueと同時に並べ替える
            names=self.names[:, idx],
            units=self.units[:, idx],
            # attrsはコピー
            attrs=self.attrs.copy() if self.attrs is not None else None
        )
        return new

    def __pow__(self, other):
        """Calculate the element-wise power of the ParameterMatrix."""
        exponent = other

        # べき乗の指数はスカラ数値または無次元のQuantityである必要がある
        if isinstance(exponent, u.Quantity):
            if not exponent.isscalar:
                raise TypeError("Exponent Quantity must be a scalar.")
            if not exponent.unit.is_equivalent(u.dimensionless_unscaled):
                raise u.UnitsError("Exponent Quantity must be dimensionless.")
            # 無次元Quantityの場合は値を取り出す
            exponent_value = exponent.value
        elif np.isscalar(exponent) or isinstance(exponent, (int, float, complex)):
            # スカラ数値の場合はそのまま使用
            exponent_value = exponent
        else:
            raise TypeError("Exponent must be a scalar number or a dimensionless Quantity.")


        # 値配列の各要素をべき乗する (ParameterMatrix の @property value は 2D ビューを返すので、それに対して計算)
        try:
            # self.value は (N, M) のビューなので、powerの結果も (N, M) となる
            result_value_2d = np.power(self.value, exponent_value)
        except Exception as e:
            # べき乗計算中に発生する可能性のあるエラーを捕捉
            raise ValueError(f"Error during element-wise power calculation: {e}")

        # 単位の各要素をべき乗する
        result_units = np.empty_like(self.units, dtype=object)
        for i in range(self.N_rows):
            for j in range(self.N_cols):
                original_unit = self.units[i, j]
                if isinstance(original_unit, u.Unit):
                    try:
                        # 単位のべき乗は Astropy Units が処理する
                        result_units[i, j] = original_unit**exponent_value
                    except Exception as e:
                         # 単位のべき乗計算中に発生する可能性のあるエラーを捕捉
                         raise ValueError(f"Error calculating unit power for element ({i}, {j}): {e}")
                else:
                    # 元の単位が Unit でない場合は、結果は無次元とする
                    result_units[i, j] = u.dimensionless_unscaled


        # 結果を新しい ParameterMatrix インスタンスとして作成
        # ParameterMatrix のコンストラクタに 2D の値配列と計算結果の単位配列を渡す
        new_param_matrix = self.__class__(
            value=result_value_2d, # 2D の値配列を渡す
            rows=self.rows.copy(), # 行メタデータをコピー
            cols=self.cols.copy(), # 列メタデータをコピー
            epoch=self.epoch, # epochを引き継ぐ
            names=self.names.copy(), # 名前を引き継ぐ
            units=result_units, # 計算結果の単位を使用
            attrs=self.attrs.copy() if self.attrs is not None else None, # attrsをコピー
            # ParameterMatrix コンストラクタは x0, dx, xunit, dtype を内部で設定
            name=f"({self.name})**{exponent}" if self.name else f"ParameterMatrix**{exponent}" # 名前に指数を含める (Quantityの場合はstrに変換)
        )
        return new_param_matrix

    def plot(self, figsize=None, cmap="viridis", show_labels=True):
        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.imshow(self.value, aspect='auto', origin='lower', cmap=cmap)
        plt.colorbar(cax, ax=ax)

        if show_labels:
            if self.cols:
                ax.set_xticks(np.arange(len(self.cols)))
                ax.set_xticklabels(self.cols, rotation=45, ha='right')
            if self.rows:
                ax.set_yticks(np.arange(len(self.rows)))
                ax.set_yticklabels(self.rows)

        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')
        plt.tight_layout()
        plt.show()

    # 禁止する機能
    @property
    def xindex(self):
        raise AttributeError("ParameterMatrix has no xindex.")

    @property
    def dx(self):
        raise AttributeError("ParameterMatrix has no dx.")

    @property
    def x0(self):
        raise AttributeError("ParameterMatrix has no x0.")

    def shift(self, dx):
        raise AttributeError("ParameterMatrix does not support shift().")

    def value_at(self, x):
        raise AttributeError("ParameterMatrix does not support value_at().")

    def append(self, *args, **kwargs):
        raise AttributeError("ParameterMatrix does not support append().")

    def prepend(self, *args, **kwargs):
        raise AttributeError("ParameterMatrix does not support prepend().")

    def crop(self, *args, **kwargs):
        raise AttributeError("ParameterMatrix does not support crop().")
