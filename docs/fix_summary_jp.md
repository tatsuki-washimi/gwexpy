# 修正案まとめ（日本語）

- `pyproject.toml` の `[project.optional-dependencies.all]` に統計系の `minepy` と `dcor` を含め、`[stats]` 相当の依存を `pip install ".[all]"` で漏れなく導入できるようにする。
- エントリポイント `gwexpy = gwexpy.cli:main` に対応する `main` 関数を `gwexpy/cli/__init__.py` に用意し、インストール後の `gwexpy` コマンドが `AttributeError` で落ちないようにする。
- PyTorch 互換層の `from_torch` で `resolve_conj` / `resolve_neg` の有無を確認し、未実装の古い PyTorch バージョンでも例外を避けられるようフォールバックやバージョン要件の明示を行う。
- `bootstrap_spectrogram` でリサンプル結果をすべて保持するメモリ設計を見直し、チャンク処理や逐次集計で大規模スペクトログラムでもスケーラブルに動作させる。
- Polars I/O でのインデックス列指定におけるパラメータ名を `time_column` へ統一するなど、TimeSeries と FrequencySeries 間で同じ操作に同じ名前を用いる。
- `TimeSeries` と `TimeSeriesCore` 間で重複する `is_regular` などの実装を共通の基底クラス／ミックスインに集約し、メンテナンス性を高める。
- 時間・周波数両系列での正則性チェックロジックを共有ミックスインとして切り出し、重複コードの分散と振る舞いの乖離を防ぐ。
