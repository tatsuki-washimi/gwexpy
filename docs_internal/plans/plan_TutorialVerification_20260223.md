# 実施計画: チュートリアル・ノートブックの検証計画

**日付**: 2026年2月23日
**担当**: Antigravity (AI Assistant)
**ステータス**: 未着手 (Proposed)

## 概要

GWpy 4.0.0 への移行および Python 3.11+ へのアップグレードに伴い、ユーザー向けのチュートリアルおよびドキュメント内のノートブックが正しく動作することを検証します。GWpy 4.0 では I/O レジストリやインデックスの扱い（Array2D 軸の慣習）に変更があり、これらは頻繁にチュートリアルで使用される機能であるため、整合性の確認が不可欠です。

## 検証戦略

### 1. ノートブックの分類

`docs/web/*/user_guide/tutorials/` および `examples/` 以下のノートブックを、重要度と依存関係に基づき以下の順で検証します。

- **フェーズ 1 (Core Tutorials - English)**: `TimeSeries`, `FrequencySeries`, `Spectrogram`, `Plot` の基本操作を扱うチュートリアル（最優先）。
- **フェーズ 2 (Advanced Analysis)**: `Bruco`, `STLT`, `Correlation` など、今回の移行でロジックを修正した高度な解析手法。
- **フェーズ 3 (Case Studies / Japanese)**: 物理モデリングの統合フロー（Noise Budget 等）および日本語版チュートリアル。

### 2. 実行環境とツール

- **環境**: Python 3.11 (gwexpy-migration conda 環境)。
- **ツール**: `pytest --nbmake` を使用。これにより並列実行と失敗箇所の詳細レポートが可能になります。
- **依存関係**: 現在インストールされているオプション依存関係（NDS2, FrameL, Control 等）の範囲で実行し、環境不足によるスキップは許容します。

### 3. 具体的な検証ステップ

#### フェーズ 1: 基本機能の健全性確認 (Sanity Check)

以下の主要な英語チュートリアルを検証します。

- `intro_timeseries.ipynb`
- `intro_frequencyseries.ipynb`
- `intro_spectrogram.ipynb`
- `intro_plotting.ipynb`

#### フェーズ 2: 高度な解析機能の検証

Migration の影響を直接受けた以下のノートブックを重点的に確認します。

- `advanced_bruco.ipynb`
- `matrix_spectrogram.ipynb` (`SpectrogramMatrix` の軸検証を含む)
- `advanced_correlation.ipynb`

#### フェーズ 3: 包括的ドキュメントテスト

`docs/web/en/` 以下の全ノートブックを実行し、結果を最終レポートとしてまとめます。

## 検証コマンド

```bash
# 特定のノートブックの実行
pytest --nbmake docs/web/en/user_guide/tutorials/intro_timeseries.ipynb

# ディレクトリ一括実行
pytest --nbmake docs/web/en/user_guide/tutorials/
```

## 成果物

- 各ノートブックの実行可否レポート。
- 不具合が発見された場合、修正および `walkthrough.md` への追記。

---

_本計画書は Antigravity によって自動生成されました。_
