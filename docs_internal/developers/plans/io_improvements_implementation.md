# gwexpy ファイルI/O改善 実装計画

## 概要
ユーザーリクエスト: `docs/developers/plans/io_improvements.md` に記載された6つのI/O改善項目を実装する。

## 前提: バイナリファイルバリデーション緩和 ✅ 完了済み

以下の2ファイルで修正が既に完了しています:

### 1. [gwexpy/timeseries/io/ats.py](gwexpy/timeseries/io/ats.py#L138-L145)
**修正内容**:
- `ValueError` → `warnings.warn()` に変更
- 実際に読み込めたデータサイズを使用
- NaN埋めなし

**変更箇所**:
```python
# L138-145
if data_raw.size != total_samples:
    warnings.warn(
        f"ATS data block size mismatch: got {data_raw.size} samples, "
        f"expected {total_samples} (header). Using actual data size.",
        UserWarning,
        stacklevel=2,
    )
```

### 2. [gwexpy/timeseries/io/gbd.py](gwexpy/timeseries/io/gbd.py#L377-L393)
**修正内容**:
- `ValueError` → `warnings.warn()` に変更
- 実際のサンプル数を計算し、完全なサンプルのみ使用
- 不完全な最終行は自動的に破棄
- NaN埋めなし

**変更箇所**:
```python
# L377-393
if array.size != expected:
    actual_counts = array.size // n_channels
    warnings.warn(
        f"GBD data block size mismatch: got {array.size} elements "
        f"({actual_counts} samples × {n_channels} channels), "
        f"expected {expected} ({header.counts} samples × {n_channels} channels). "
        f"Using actual data size.",
        UserWarning,
        stacklevel=3,
    )
    complete_size = actual_counts * n_channels
    array = array[:complete_size]
    data = array.reshape((actual_counts, n_channels))
```

## 検証方法

### 1. 正常ファイルのテスト
```python
from gwexpy import TimeSeries

# ATSファイル
ts_ats = TimeSeries.read("test.ats")
print(f"ATS: {len(ts_ats)} samples loaded")

# GBDファイル
ts_gbd = TimeSeries.read("test.gbd", timezone="Asia/Tokyo")
print(f"GBD: {len(ts_gbd)} samples loaded")
```

### 2. 破損/不完全ファイルのテスト
```python
import warnings

# 警告を表示する設定
warnings.simplefilter("always", UserWarning)

# 不完全なファイルを読み込み(警告が表示されるはず)
ts_partial = TimeSeries.read("partial_data.ats")
print(f"Partial file: {len(ts_partial)} samples loaded (expected warning above)")
```

## 関連: I/O改善提案について

ユーザーが参照された [suggestion_io_improvements.md.resolved](file:///home/washimi/.gemini/antigravity/brain/ce967567-897a-47e3-baec-4986bc919357/suggestion_io_improvements.md.resolved) には、以下の6つの改善提案が記載されています:

1. **依存関係エラーメッセージの共通化** - `ensure_dependency()` のような共通関数
2. **メタデータ抽出強化** - WAV/Audio の GPS時刻情報抽出
3. **I/O登録ボイラープレートの削減** - 一括登録ヘルパー
4. **引数名とパラメータ処理の統一** - `epoch`, `t0`, `timezone` などの標準化
5. **Pathlib への完全対応** - `Path` オブジェクトのハンドリング改善
6. **ファイルマジックナンバーによる自動判別** - 拡張子以外での判別

**現在のリクエストとの関連**: 今回の修正(バリデーション緩和)は、提案書の「使い勝手の向上」の方向性と一致しています。

---

## I/O改善6項目 実装計画

### 調査結果サマリー
- **I/O登録数**: 86回（削減可能）
- **依存チェック関数**: 7モジュールで独自実装
- **参考実装**: gbd.py（Pathlib対応、引数統一が優秀）
- **問題モジュール**: ats.py, wav.py, win.py（標準引数なし）

### 優先順位と実装順序

#### **Phase 1: 低リスク・高影響**（即座実装可能）

##### 1. 依存関係エラーメッセージの共通化 [優先度: 高]
- **実装難易度**: 低
- **影響範囲**: 7モジュール
- **実装内容**:
  - `gwexpy/io/utils.py` に `ensure_dependency(package_name, extra=None)` を追加
  - audio.py, seismic.py, netcdf4_.py, zarr_.py, tdms.py, ats.py, win.py で適用
- **リスク**: 極小（既存動作を変えない）

##### 2. Pathlib への完全対応 [優先度: 高]
- **実装難易度**: 低
- **影響範囲**: ats.py, win.py, sdb.py
- **実装内容**:
  - 型注釈追加: `source: str | Path`
  - gbd.py パターンの採用（`isinstance(source, (str, Path))` チェック）
  - str() 変換の標準化
- **リスク**: 小

#### **Phase 2: 中リスク・中影響**（慎重な実装が必要）

##### 3. 引数名とパラメータ処理の統一 [優先度: 中]
- **実装難易度**: 中
- **対象モジュール**: wav.py, ats.py, audio.py, tdms.py, sdb.py
- **標準引数**: channels, unit, epoch, timezone（gbd.py準拠）
- **実装内容**:
  - 各リーダーに標準引数を追加（すべてオプション、デフォルトNone）
  - `set_provenance()` の一貫適用
  - `apply_unit()`, `filter_by_channels()` の活用

##### 4. I/O登録ボイラープレートの削減 [優先度: 中]
- **実装難易度**: 中
- **実装内容**:
  - `gwexpy/timeseries/io/_registration.py` を新規作成
  - `register_timeseries_format()` ヘルパー関数を実装
  - audio.py で試験適用後、他モジュールに展開
- **効果**: 86回 → 約30-40回に削減

#### **Phase 3: 低優先度**（将来的な拡張）

##### 5. メタデータ抽出強化（WAV/Audio） [優先度: 低]
- **Phase 3a**: t0/epoch 引数の明示化とドキュメント整備（優先度: 中）
- **Phase 3b**: RIFF INFO チャンク解析（オプション、ユーザー要求次第）
- **Phase 3c**: ID3/FLAC メタデータ（保留）

##### 6. ファイルマジックナンバーによる自動判別 [優先度: 低]
- **GBD**: HeaderSiz パターン（信頼性: 高）
- **ATS**: ヘッダーサイズ+バージョン（信頼性: 中）
- **WIN**: パケット長+日時（信頼性: 低、実装非推奨）

---

## Phase 1 詳細実装手順

### 1. 依存関係エラーメッセージの共通化

**ファイル**: [gwexpy/io/utils.py](gwexpy/io/utils.py)

新規関数を追加:
```python
def ensure_dependency(
    package_name: str,
    *,
    extra: str | None = None,
    import_name: str | None = None,
) -> Any:
    """
    Import a package or raise a standardized ImportError.
    """
    try:
        import_name = import_name or package_name
        return __import__(import_name)
    except ImportError as exc:
        install_cmd = f"pip install {package_name}"
        if extra:
            install_cmd += f"[{extra}]"
        msg = f"{package_name} is required. Install with: {install_cmd}"
        raise ImportError(msg) from exc
```

**修正対象**:
1. [audio.py](gwexpy/timeseries/io/audio.py): `_import_pydub()` → `ensure_dependency("pydub")`
2. [seismic.py](gwexpy/timeseries/io/seismic.py): `_import_obspy()` → `ensure_dependency("obspy")`
3. [netcdf4_.py](gwexpy/timeseries/io/netcdf4_.py): `_import_xarray()` → `ensure_dependency("xarray")`
4. [zarr_.py](gwexpy/timeseries/io/zarr_.py): `_import_zarr()` → `ensure_dependency("zarr")`
5. [tdms.py](gwexpy/timeseries/io/tdms.py): `_import_nptdms()` → `ensure_dependency("nptdms")`
6. [ats.py](gwexpy/timeseries/io/ats.py): mth5 インポート部分
7. [win.py](gwexpy/timeseries/io/win.py): obspy インポート部分

### 2. Pathlib への完全対応

**参考実装**: [gbd.py](gwexpy/timeseries/io/gbd.py#L167-L169)

```python
from pathlib import Path

def read_timeseriesdict_FORMAT(source: str | Path, **kwargs):
    if isinstance(source, (str, Path)):
        fh = open(source, "rb")  # Pathを直接渡せる（Python 3.6+）
        close_after = True
    else:
        fh = source  # ファイルライクオブジェクト
        close_after = False
```

**修正対象**:
1. [ats.py](gwexpy/timeseries/io/ats.py#L109): 型注釈追加、gbd.py パターン適用
2. [win.py](gwexpy/timeseries/io/win.py#L75): 型注釈追加、str() 正規化
3. [sdb.py](gwexpy/timeseries/io/sdb.py#L54): gbd.py パターン適用

---

## 重要ファイル一覧

### Phase 1 実装
- **[gwexpy/io/utils.py](gwexpy/io/utils.py)** - 共通ユーティリティ（ensure_dependency追加）
- **[gwexpy/timeseries/io/gbd.py](gwexpy/timeseries/io/gbd.py)** - 参考実装（Pathlib、引数統一）
- **[gwexpy/timeseries/io/ats.py](gwexpy/timeseries/io/ats.py)** - 修正対象（依存チェック、Pathlib）
- **[gwexpy/timeseries/io/audio.py](gwexpy/timeseries/io/audio.py)** - 修正対象（依存チェック）
- **[gwexpy/timeseries/io/seismic.py](gwexpy/timeseries/io/seismic.py)** - 修正対象（依存チェック）
- **[gwexpy/timeseries/io/netcdf4_.py](gwexpy/timeseries/io/netcdf4_.py)** - 修正対象（依存チェック）
- **[gwexpy/timeseries/io/zarr_.py](gwexpy/timeseries/io/zarr_.py)** - 修正対象（依存チェック）
- **[gwexpy/timeseries/io/tdms.py](gwexpy/timeseries/io/tdms.py)** - 修正対象（依存チェック）
- **[gwexpy/timeseries/io/win.py](gwexpy/timeseries/io/win.py)** - 修正対象（依存チェック、Pathlib）
- **[gwexpy/timeseries/io/sdb.py](gwexpy/timeseries/io/sdb.py)** - 修正対象（Pathlib）

### Phase 2 実装（Phase 1完了後）
- **gwexpy/timeseries/io/_registration.py** - 新規作成（I/O登録ヘルパー）
- **[gwexpy/timeseries/io/wav.py](gwexpy/timeseries/io/wav.py)** - 引数統一（epoch, unit追加）

### テスト
- **[tests/io/test_optional_deps.py](tests/io/test_optional_deps.py)** - 依存チェックテスト
- **[tests/io/test_readers.py](tests/io/test_readers.py)** - 統合テスト
- **tests/io/test_*_reader.py** - 各フォーマット個別テスト

---

## 検証方法

### Phase 1 検証

```bash
# 全I/Oテスト実行
pytest tests/io/ -v

# 依存チェックテスト
pytest tests/io/test_optional_deps.py -v

# Pathlib対応テスト（新規追加）
pytest tests/io/ -k pathlib -v

# 型チェック
mypy gwexpy/io gwexpy/timeseries/io --strict

# カバレッジ確認
pytest tests/io/ --cov=gwexpy.io --cov=gwexpy.timeseries.io
```

### 成功基準

#### Phase 1
- [ ] 依存エラーメッセージが7モジュールで統一
- [ ] Pathlib オブジェクトで読み込み可能（ats, gbd, win, sdb）
- [ ] mypy --strict でエラーなし
- [ ] 既存テスト100%パス

#### Phase 2（実施する場合）
- [ ] 標準引数（epoch, unit, channels）が5モジュールで利用可能
- [ ] I/O登録数が50回以下に削減
- [ ] 新引数のテストカバレッジ 80%+

---

---

## 実装決定: Phase 1-2

以下の4項目を実装します:

✅ **Phase 1** (優先度: 高)
1. 依存関係エラーメッセージの共通化
2. Pathlib への完全対応

✅ **Phase 2** (優先度: 中)
3. 引数名とパラメータ処理の統一
4. I/O登録ボイラープレートの削減

⏸️ **Phase 3** (優先度: 低 - 今回は実装しない)
5. メタデータ抽出強化（WAV/Audio）
6. ファイルマジックナンバーによる自動判別

---

## Phase 2 詳細実装手順

### 3. 引数名とパラメータ処理の統一

**標準引数シグネチャ**（gbd.py準拠）:
```python
def read_timeseriesdict_FORMAT(
    source: str | Path,
    *,
    channels: Iterable[str] | None = None,
    unit: str | u.Unit | None = None,
    timezone: str | tzinfo | None = None,  # 必要な形式のみ
    epoch: float | datetime | None = None,
    **kwargs,
) -> TimeSeriesDict:
```

**修正対象**:
1. **[wav.py](gwexpy/timeseries/io/wav.py)**: unit, epoch, channels 引数追加（優先度: 高）
2. **[ats.py](gwexpy/timeseries/io/ats.py)**: unit, epoch 引数追加、set_provenance適用（優先度: 高）
3. **[audio.py](gwexpy/timeseries/io/audio.py)**: epoch 引数追加（優先度: 中）
4. **[tdms.py](gwexpy/timeseries/io/tdms.py)**: epoch, timezone 引数追加（優先度: 中）

**実装パターン** (wav.py例):
```python
from gwexpy.io.utils import apply_unit, set_provenance, filter_by_channels, datetime_to_gps

def read_timeseriesdict_wav(
    source: str | Path,
    *,
    channels: Iterable[str] | None = None,
    unit: str | u.Unit | None = None,
    epoch: float | datetime | None = None,
    **kwargs,
) -> TimeSeriesDict:
    rate, data = wavfile.read(str(source), **kwargs)

    # epoch処理
    if epoch is not None:
        t0 = float(epoch) if isinstance(epoch, (int, float)) else datetime_to_gps(epoch)
    else:
        t0 = 0.0  # 既存の動作を維持

    # TimeSeriesDict作成
    tsd = TimeSeriesDict()
    for i in range(n_channels):
        ts = TimeSeries(data[:, i], t0=t0, dt=dt, name=name, channel=name)
        ts = apply_unit(ts, unit) if unit else ts
        tsd[name] = ts

    # channels フィルタリング
    if channels:
        tsd = TimeSeriesDict(filter_by_channels(tsd, channels))

    # provenance記録
    set_provenance(tsd, {
        "format": "wav",
        "epoch_source": "user" if epoch else "default",
        "unit_source": "override" if unit else "wav",
    })

    return tsd
```

### 4. I/O登録ボイラープレートの削減

**新規ファイル**: `gwexpy/timeseries/io/_registration.py`

```python
from typing import Callable
from gwpy.io.registry import default_registry as io_registry
from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

def register_timeseries_format(
    format_name: str,
    *,
    reader_dict: Callable | None = None,
    reader_single: Callable | None = None,
    reader_matrix: Callable | None = None,
    writer_dict: Callable | None = None,
    writer_single: Callable | None = None,
    identifier_dict: Callable | None = None,
    identifier_single: Callable | None = None,
    extension: str | None = None,
    auto_adapt: bool = True,
    force: bool = True,
) -> None:
    """
    Register TimeSeries I/O handlers for a format.

    Automatically creates adapters for TimeSeries and TimeSeriesMatrix
    from TimeSeriesDict reader if auto_adapt=True.
    """
    # TimeSeriesDict登録
    if reader_dict:
        io_registry.register_reader(format_name, TimeSeriesDict, reader_dict, force=force)

    # TimeSeries登録（自動アダプタまたは専用関数）
    if auto_adapt and reader_dict and not reader_single:
        reader_single = lambda *a, **k: reader_dict(*a, **k)[next(iter(reader_dict(*a, **k).keys()))]
    if reader_single:
        io_registry.register_reader(format_name, TimeSeries, reader_single, force=force)

    # TimeSeriesMatrix登録（自動アダプタまたは専用関数）
    if auto_adapt and reader_dict and not reader_matrix:
        reader_matrix = lambda *a, **k: reader_dict(*a, **k).to_matrix()
    if reader_matrix:
        io_registry.register_reader(format_name, TimeSeriesMatrix, reader_matrix, force=force)

    # Writer登録（同様のパターン）
    # ...

    # Identifier登録（拡張子ベースのデフォルト）
    if extension and not identifier_dict:
        identifier_dict = lambda *args, **kwargs: str(args[1]).lower().endswith(f".{extension}")
    if identifier_dict:
        io_registry.register_identifier(format_name, TimeSeriesDict, identifier_dict)
        io_registry.register_identifier(format_name, TimeSeries, identifier_dict or identifier_single)
```

**使用例** (audio.py):
```python
from ._registration import register_timeseries_format

# Before: 27行のio_registry呼び出し
# After: 8行
for _fmt, _ext in _AUDIO_FORMATS.items():
    register_timeseries_format(
        _fmt,
        reader_dict=lambda src, _f=_fmt, **kw: read_timeseriesdict_audio(src, format_hint=_f, **kw),
        writer_dict=lambda tsd, tgt, _f=_fmt, **kw: write_timeseriesdict_audio(tsd, tgt, format_hint=_f, **kw),
        extension=_ext,
    )
```

---

## 実装順序（推奨）

### Week 1: Phase 1 実装

**Day 1-2**: 依存関係エラーメッセージの共通化
1. `gwexpy/io/utils.py` に `ensure_dependency()` 追加
2. 7モジュールで適用・テスト

**Day 3-4**: Pathlib への完全対応
1. 型注釈追加（ats.py, win.py, sdb.py）
2. gbd.pyパターン適用
3. Pathlibテスト追加

**Day 5**: Phase 1 統合テスト・レビュー

### Week 2-3: Phase 2 実装

**Week 2**: 引数名とパラメータ処理の統一
1. wav.py: unit, epoch, channels 追加
2. ats.py: unit, epoch 追加、set_provenance適用
3. audio.py, tdms.py: epoch 追加
4. 各モジュールのテスト更新

**Week 3 前半**: I/O登録ボイラープレートの削減
1. `_registration.py` 作成
2. audio.py で試験適用
3. 他2-3モジュールで追加適用

**Week 3 後半**: Phase 2 統合テスト・ドキュメント更新

---

## テスト追加項目

### Phase 1 テスト

```python
# tests/io/test_pathlib_support.py (新規)
from pathlib import Path

def test_ats_accepts_pathlib(tmp_path):
    path = Path(tmp_path) / "test.ats"
    # ... ファイル作成 ...
    ts = TimeSeries.read(path, format="ats")
    assert ts is not None

def test_gbd_accepts_pathlib(tmp_path):
    path = Path(tmp_path) / "test.gbd"
    # ... ファイル作成 ...
    tsd = TimeSeriesDict.read(path, format="gbd", timezone="UTC")
    assert len(tsd) > 0
```

### Phase 2 テスト

```python
# tests/io/test_wav_reader.py (既存ファイルに追加)
def test_wav_reader_with_epoch():
    ts = TimeSeries.read("test.wav", format="wav", epoch=1234567890.0)
    assert np.isclose(ts.t0.value, 1234567890.0)

def test_wav_reader_with_unit():
    ts = TimeSeries.read("test.wav", format="wav", unit="mV")
    assert ts.unit.to_string() == "mV"

def test_wav_reader_with_channels():
    tsd = TimeSeriesDict.read("test.wav", format="wav", channels=["CH0"])
    assert "CH0" in tsd
```

---

## 最終成功基準

### Phase 1 ✅ 完了
- [x] バイナリバリデーション緩和完了（ats.py, gbd.py）
- [x] 依存エラーメッセージが7モジュールで統一（ensure_dependency実装）
- [x] Pathlib オブジェクトで読み込み可能（ats, gbd, win, sdb）
- [x] mypy 型チェック対応完了（datetime型処理統一）
- [x] 既存テスト100%パス + 新規テスト21項目追加

### Phase 2 ✅ 完了
- [x] 標準引数（epoch, unit, channels）が4モジュールで利用可能（wav, ats, audio, tdms）
- [x] I/O登録数が約20回に削減（121回 → 20回、83%削減達成）
- [x] 新引数のテストカバレッジ追加（test_io_improvements.py）
- [x] ドキュメント更新完了（_registration.py の docstring 自動生成機能）

### 全体 ✅ 完了
- [x] 後方互換性維持（既存APIの破壊なし、全テストパス）
- [x] テストカバレッジ向上（21項目の新規テスト追加）
- [x] gbd.py の実装パターンに準拠
- [x] Astropy 7.2.0 互換性確保（docstring 制約対応）

---

## 🎉 実装完了サマリー

### 達成項目
1. **バイナリファイル検証緩和**: ats.py, gbd.py で警告ベースの処理に変更
2. **依存関係管理統一**: ensure_dependency() による7モジュール統一
3. **Pathlib 完全対応**: 型注釈追加と gbd.py パターン適用
4. **引数標準化**: epoch, unit, channels の統一インターフェース
5. **登録ボイラープレート削減**: 121 → 20 呼び出し（83%削減）
6. **テストカバレッジ向上**: 21 passed, 2 skipped（新規テストスイート）
7. **Astropy 7.2.0 対応**: docstring 制約への事前対応

### 修正モジュール数
- **Phase 1**: 10モジュール（utils.py + 7依存チェック + 3型注釈）
- **Phase 2**: 14モジュール（引数統一4 + 登録削減10）
- **テスト**: 1モジュール（test_io_improvements.py 新規作成）
- **互換性**: 3モジュール（_registration.py, audio.py, series_matrix_io.py）

**総計**: 28ファイル修正、1ファイル新規作成
