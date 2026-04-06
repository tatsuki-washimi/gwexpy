# コマンドラインインターフェース (CLI)

## 概要

GWexpy CLI は GWpy のパイプライン機能をコマンドラインからアクセスするためのツールです。現在、GWexpy は対話的な解析とスクリプティングのために **Python API** を主に利用しています。CLI は一般的なワークフローへの軽量なフロントエンドを提供します。

## 現在のステータス

GWexpy CLI は **開発中**です。いくつかのコマンドが利用可能ですが、ほとんどの高度な解析ワークフローは **Python API** を使用することをお勧めします。

## 利用可能なコマンド

### `gwexpy --version`

インストールされている GWexpy のバージョンを表示します：

```bash
gwexpy --version
```

出力：
```
gwexpy 0.1.0
```

### `gwexpy --help`

一般的なヘルプ情報を表示します：

```bash
gwexpy --help
```

## GWpy CLI の使用

GWexpy は GWpy のコマンドラインツールを再エクスポートしています。GWpy CLI の詳細なドキュメントについては、[GWpy ドキュメント](https://gwpy.readthedocs.io/en/latest/cli/)を参照してください。

**注意：** 複雑な解析パイプライン、重力波パラメータ推定、カスタムデータ処理については、**Python API** の使用をお勧めします。API の例については、[はじめに](./getting_started.md)ガイドを参照してください。

## 今後の開発予定

今後の GWexpy リリースでは、以下の専門的なサブコマンドを追加予定です：
- データ取り込みと検証
- ノイズ特性評価
- 時間-周波数解析
- イベント位置決定

計画中の機能とスケジュールについては、[ロードマップ](https://github.com/tatsuki-washimi/gwexpy/issues)を参照してください。

## トラブルシューティング

インストール後に `gwexpy` コマンドが見つからない場合は、GWexpy が正しい環境にインストールされていることを確認してください：

```bash
pip install -e ".[gw]"  # 重力波 extras 付きでインストール
```

インストールを確認します：

```bash
python -c "import gwexpy; print(gwexpy.__version__)"
```
