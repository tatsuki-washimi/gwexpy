# GUIアプリケーション（別リポジトリ）

GWexpy は解析ライブラリであり、GUIアプリケーション本体は提供しません。
以前の開発用途の `gwexpy.gui` モジュールはリポジトリから分離・削除されました。
PyPI配布物にも含まれていなかった機能です。

- [pyaggui](https://github.com/tatsuki-washimi/pyaggui) は独立したGUIアプリケーションです。導入方法・CLI・テストはそのリポジトリで管理します。
- **GWexpy Studio** は別リポジトリで開発されるGUIプロジェクトです。導入方法と互換性については当該プロジェクトの説明を参照してください。

`gwexpy.plot` などの解析ライブラリ側のプロット機能は引き続き利用できます。
今回削除するのは旧GUIアプリケーション実装であり、描画ライブラリの機能ではありません。

分離の経緯: [GWexpy Issue #645](https://github.com/tatsuki-washimi/gwexpy/issues/645)
