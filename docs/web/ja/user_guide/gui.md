# GUI アプリケーションは別リポジトリで管理しています

GWexpy 本体は解析ライブラリです。旧 `gwexpy.gui` アプリケーションは
ソースツリーと配布物には含まれません。

従来の実験的な GUI は独立した
[pyaggui リポジトリ](https://github.com/tatsuki-washimi/pyaggui)
へ分離されました。GWexpy Studio は、別リポジトリで開発される
独立した GUI アプリケーションであり、pyaggui と同一のものではありません。

時系列解析・I/O・統計・描画には GWexpy の公開 Python API を使用してください。
このページは旧 GUI ガイドへの既存リンクを維持するための移行案内です。
旧 GUI 実装の履歴は Git history に残されています。
