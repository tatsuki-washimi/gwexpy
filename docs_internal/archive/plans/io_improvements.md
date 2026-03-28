# gwexpy ファイルI/O 改善提案

現在のコードベースをレビューした結果、ファイルI/O周りで改善の余地があると思われる箇所を以下の6点にまとめました。

## 1. 依存関係エラーメッセージの共通化
複数のモジュール（`audio.py`, `seismic.py`, `netcdf4_.py` 等）で、以下のような独自のインポートチェックが行われています。
```python
def _import_xarray():
    try: import xarray as xr
    except ImportError: raise ImportError("Install with `pip install xarray`...")
```
これを `gwexpy.io.utils.ensure_dependency(name, extra=None)` のような共通関数に置き換えることで、インストール指示のトーンを統一できます。

## 2. メタデータの抽出強化（WAV/Audio）
現在、WAVやMP3/FLACの読み込み時に $t_0$（GPS時刻）が $0.0$ に固定されています。
- RIFF INFOタグやID3, FLACメタデータから時刻情報を抽出する試みを行う。
- 読み込み時に `t0` 引数を通じて明示的に指定しやすくする（現状 `TimeSeries` コンストラクタに正しく渡っていない箇所がありました）。

## 3. I/O登録ボイラープレートの削減
各I/Oモジュール末尾で、`TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix` のそれぞれに対してリーダー/ライターを個別に登録しています。
これを `gwexpy.io.registry.register_timeseries_io(fmt, reader, ...)` のような一括登録ヘルパーで簡略化できます。

## 4. 引数名とパラメータ処理の統一
リーダーによって `epoch`, `t0`, `timezone` などの引数の扱いが異なっています。
- すべての自作リーダーで `gwexpy.io.utils.apply_unit` や `set_provenance` を一貫して適用する。
- `timezone` や `unit` を共通のキーワード引数として標準化する。

## 5. Pathlib への完全対応
一部のリーダー（特にバイナリ形式）で、`source` が `pathlib.Path` オブジェクトである場合に不具合（`f.seek()` などのエラー）が発生しうる箇所があります。すべてのエントリポイントで `str()` への正規化を徹底するか、ファイルライクオブジェクトとしての扱いを強化する必要があります。

## 6. ファイルマジックナンバーによる自動判別
ATSやGBDなどのバイナリ形式は、現状「拡張子」のみで判別されています。
`io_registry.register_identifier` に、ファイルの先頭数バイトを読み取って判別するロジックを追加することで、拡張子が正しくないファイルでも自動判別が可能になります。

---

これらの改善は、機能の追加というよりも「使い勝力の向上（不便さの解消）」に寄与するものと考えています。もし優先順位が高いものがあれば、具体的に検討・実装を進めることが可能です。
