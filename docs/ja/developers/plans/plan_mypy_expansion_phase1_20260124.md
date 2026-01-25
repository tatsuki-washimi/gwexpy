# 作業計画書: 型チェックカバレッジの拡大 (Phase 1: FrequencySeries)

**作成日時**: 2026-01-24 17:55
**タスク**: `gwexpy.frequencyseries` パッケージの MyPy 除外を解除し、型エラーを修正する

---

## 1. 目的とゴール

リポジトリ全体の型安全性を向上させるため、現在 `pyproject.toml` で `ignore_errors = true` に設定されているモジュールの除外を段階的に解除します。最初のステップとして、コアコンポーネントである `gwexpy.frequencyseries` を対象にします。

**ゴール**:

- `gwexpy.frequencyseries` パッケージ全体の MyPy チェックをパスさせる。
- Mixin クラスにおける属性未定義エラーを適切に解決する。
- 静的な型定義の重複を排除する。

---

## 2. 詳細ロードマップ

### フェーズ 1: `gwexpy/frequencyseries/frequencyseries.py` の修正 (5分)

- `SeriesType` の再定義エラーを `TYPE_CHECKING` ガードを使用して回避。

### フェーズ 2: Mixin クラスの型定義改善 (15分)

- `FrequencySeriesMatrixAnalysisMixin` 等において、`self` が期待する属性（`value`, `meta`, `shape`等）を定義。
- 必要に応じて `typing.Protocol` または `self: TargetClass` 注釈を使用。

### フェーズ 3: コレクション型の修正 (10分)

- `collections.py` におけるジェネリクス (`TypeVar`) と `super()` 呼び出しの整合性を確認し修正。

### フェーズ 4: 検証と定着 (5分)

- `mypy -p gwexpy.frequencyseries` を実行し、エラーがゼロであることを確認。
- `pyproject.toml` からの除外設定が完全に削除されていることを最終確認。

---

## 3. テスト・検証計画

- **MyPy**: `mypy -p gwexpy.frequencyseries --ignore-missing-imports`
- **Pytest**: 既存の `frequencyseries` 関連テストを実行し、リグレッションがないことを確認。

---

## 4. 推奨モデル、スキル、工数見積もり

### 推奨モデル

- **Claude Sonnet 4.5**
  - 複雑な Mixin の型推論とリファクタリングに最適。

### 推定時間

- **合計**: 35 〜 45 分

| 項目       | 時間 |
| :--------- | :--- |
| 分析と準備 | 5分  |
| 実装と修正 | 25分 |
| 検証       | 10分 |

---

## 5. 承認確認

この計画で進めてよろしいでしょうか？承認いただければ、フェーズ1から順次着手します。
