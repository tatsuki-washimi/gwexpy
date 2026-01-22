# 作業報告: ScalarField への統一リファクタ (2026-01-22)

## 概要
- Field4D 系 API を `gwexpy.fields` の `ScalarField`/`FieldList`/`FieldDict` に統一し、ドメインメタデータのバリデーションを追加。
- ドキュメント、チュートリアル、テストを新 API に更新。
- 全テスト・lint・型チェックを実行し合格。

## 変更点
- `gwexpy/fields/` に FieldBase/ScalarField/collections を追加し、FFT 後のドメイン/単位整合を検証。
- `gwexpy/__init__.py` から新フィールド API を公開、`types/field4d.py` はレガシー shim に。
- テスト: 新規 `tests/fields/test_scalarfield_domain.py` でドメイン/単位伝播を検証。既存 Field4D* テストを新 API import に更新。
- ドキュメント/例: ScalarField 参照ページを追加、Field4D* 参照ページを削除。ノートブック・デモ・開発ドキュメントを新名称に置換。

## 検証
- `ruff check` : pass
- `mypy .` : pass
- `pytest` : pass (2284 passed, 330 skipped, 3 xfailed; GUI/外部環境マーク等の skip は既存)

## 所感・次のアクション
- ベクトル/テンソルフィールド実装は未着手（NotImplemented のプレースホルダーあり）。
- docstring の新名称反映やさらなるドメイン伝播テスト（部分 FFT/切り出しの網羅）を追加可能。
- リリース前に `wrap_up_gwexpy` で最終整備・コミット推奨。
