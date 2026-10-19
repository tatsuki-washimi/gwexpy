# グラフィカルユーザーインターフェース (独立プロジェクト pyaggui)

## 概要

**ステータス:** 独立リポジトリへ移行完了 ([pyaggui](https://github.com/tatsuki-washimi/pyaggui))

旧GWexpyツリー内GUI実装（`gwexpy.gui`）はGWexpyから撤去され、独立リポジトリ **pyaggui** へ移行しました。

GWexpy は科学解析、物理データモデル、デジタル信号処理に特化したコアライブラリとして保守されます。対話的GUI機能は独立したコンパニオンプロジェクトで提供されます。

## 関連GUIプロジェクト

- **pyaggui**: 重力波データ探索および可視化のための独立した診断GUIアプリケーション。
  - リポジトリ: [https://github.com/tatsuki-washimi/pyaggui](https://github.com/tatsuki-washimi/pyaggui)
  - 主な機能: マルチチャンネル時系列探索、スペクトル蓄積・平均化、スペクトログラム表示、NDSストリーミング。
- **GWexpy Studio**: 次世代デスクトップ解析スタジオ（独立アプリケーションプロジェクト）。

## インストールと利用方法

`pyaggui` のGitHubリポジトリは現在privateで、公開PyPI配布もありません。インストールにはリポジトリへのアクセス権が必要です。
アクセス権がある場合は、READMEに記載された次のソースインストール方法を利用してください。`pip install pyaggui` は案内されている方法ではありません。

```bash
git clone https://github.com/tatsuki-washimi/pyaggui.git
cd pyaggui
python -m pip install .
pyaggui
```

動作要件、対応環境、Issue報告については [pyaggui リポジトリ](https://github.com/tatsuki-washimi/pyaggui) を参照してください。

## 関連項目

- {doc}`Python API ドキュメント <../index>` - プログラマティックな科学データ解析
- {doc}`チュートリアル <tutorials/index>` - GWexpy の学習用インタラクティブ例
