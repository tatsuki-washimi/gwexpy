---
name: audit_api_elements
description: リポジトリ内の全クラス・関数を、ソースコードの定義から自動的に抽出し、公開・内部実装などの利用区分で分類・整理する。
---

# Audit API Elements

このスキルは、リポジトリ内で定義されているすべてのクラスとトップレベルの関数を抽出し、それらが「ユーザー向けの公開 API」か「内部実装用のプライベート要素」かを系統的に分類・整理するためのものです。

## Instructions

### 1. 全要素の抽出 (Extraction)

以下のコマンド（または同梱の `scripts/audit_api.sh`）を使用して、リポジトリ内の定義をスキャンします。

```bash
# クラスの抽出
find gwexpy/ -name "*.py" ! -name "*test*" -exec grep -H "^[[:space:]]*class " {} +

# トップレベル関数の抽出 (Public のみ)
find gwexpy/ -name "*.py" ! -name "*test*" -exec grep -H "^def " {} + | grep -vE ":def[[:space:]]+_"
```

> [!IMPORTANT]
> インデントされた `class` や `def` を見逃さないよう、正規表現 `^[[:space:]]*` を使用してください（条件付き定義やネストされたクラスを捕捉するため）。

### 2. 利用区分による分類 (Categorization)

抽出されたリストを以下の基準で分類します。

#### A. 公開クラス・関数 (Public)
- ユーザーが直接 `import` して使用することを想定したもの。
- アンダースコア（`_`）で始まらない名前。
- `TimeSeries`, `FrequencySeries`, `tconvert`, `to_gps` など、主要な解析エンリポイント。

#### B. 内部実装・基底クラス (Internal / Base)
- 名前が `_` で始まるもの（例: `_TimeSeriesDictLike`）。
- クラス名に `Mixin`, `Base`, `Core`, `Protocol`, `Interface` を含むもの。
- `gui/` 配下の全要素（GUI パッケージは内部実装扱いとする）。
- `MetaData` およびその関連クラス（内部的な状態管理用）。

#### C. I/O バックエンド (IO Backends)
- `read_*`, `write_*`, `identify_*` 形式の関数。
- これらは `TimeSeries.read()` 等の統一 API を通じて内部的に呼び出されるため、直接の使用は想定せず「内部実装」に分類する。

#### D. 結果・コンテキストオブジェクト (Context / Results)
- 解析メソッドの戻り値としてユーザーに返されるオブジェクト（例: `PCAResult`, `CouplingResult`）。
- ユーザーが直接インスタンス化することは稀だが、公開 API の一部として扱う。

### 3. 特殊なケースの確認

- **`__init__.py` のチェック**: PEP 562（`__getattr__`）による動的インポートや、他ライブラリ（`gwpy` 等）からの透過的な再エクスポートがないか確認し、リストに備考として記載する。

## Cautions (このチャットでの知見)

- **I/O の判断**: 単独の `read_timeseries_gbd` 等は内部用とし、クラスメソッド経由での利用を標準とする。
- **特定のサブパッケージ**: `gwexpy/gui/` は一律で内部実装（プライベート）として扱う。
- **メタデータの判断**: `MetaData` は一方向的な（内部での）属性管理のため、ユーザー向けの公開データ構造からは除外する。

## Scripts

同梱のスクリプトを使用すると、一括でリストを出力できます：
```bash
./scripts/audit_api.sh  # 引数なしで gwexpy/ をスキャン
```
