# Full Mode: ドキュメント更新

ドキュメントの同期とビルド手順。

## Docstring の同期

コード変更後、docstring とドキュメントの整合性を確認:

1. 関数シグネチャの変更 → docstring を更新
2. 新しいパラメータの追加 → `Parameters` セクションに追記
3. 戻り値の変更 → `Returns` セクションを更新

## Sphinx ビルド

```bash
cd docs
make html

# または
sphinx-build -b html web/en/ _build/html/en/
sphinx-build -b html web/ja/ _build/html/ja/
```

## 警告の確認

ビルド時の警告を確認:

```bash
sphinx-build -W -b html web/en/ _build/html/en/
```

- `-W`: 警告をエラーとして扱う
- nitpick 警告は `conf.py` の `nitpick_ignore` で管理

## バイリンガル対応

gwexpy は英語・日本語の両方でドキュメントを提供:

1. `docs/web/en/` - 英語版
2. `docs/web/ja/` - 日本語版

コード変更時は両方を更新。

## API リファレンスの更新

新しいモジュールを追加した場合:

1. `docs/web/en/reference/` に対応する `.rst` ファイルを作成
2. `index.rst` の toctree に追加
3. 日本語版も同様に更新
