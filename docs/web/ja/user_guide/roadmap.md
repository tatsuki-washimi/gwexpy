# ロードマップ

このページは、GWexpy のドキュメント整備と機能計画の**公開ロードマップ入口**です。
厳密なリリース契約ではなく、どの方向へ改善を進めているかを共有するための概要として扱ってください。

## このページの読み方

- **直近**: 近いイテレーションで改善優先度が高い領域
- **中期**: 現在の docs / API 整理の後で広げていきたい領域
- **長期**: 有用だが、まだ具体的な時期を固定していない方向性

ロードマップは公開していますが、研究上の要請や保守コスト、上流依存の変化によって優先度は変わりえます。

## 現在のリリース基準

- [v0.2.3](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.3) は、現在公開されているメンテナンスリリースです。公開 API や依存パッケージを追加せず、GWpy 4.0.1 / 4.0.2 の監査済み既定挙動との互換性を復元しました。
- v0.2.0 で、予測可能な単位、metadata の伝播、暗黙の型 downgrade ではなく明示的に失敗する container arithmetic の基準を確立しました。
- exact timing、相互運用可能な persistence、決定論的な provenance、公開 GWpy compatibility は v0.2.0 で確立され、v0.2.3 でも維持されています。

v0.2.3 は [PyPI](https://pypi.org/project/gwexpy/0.2.3/)、[conda-forge](https://anaconda.org/conda-forge/gwexpy)、[Zenodo（DOI 10.5281/zenodo.22344992）](https://doi.org/10.5281/zenodo.22344992) で公開されています。次の minor release のテーマは committed ではなく、今後の作業は方向性として扱います。

## 直近の重点項目

- ナビゲーション、アクセシビリティ、検索性を含むドキュメント品質改善
- GWpy ユーザー向け移行ガイドの明確化
- notebook / tutorial の CI 上での安定性向上
- チュートリアル、ガイド、API リファレンス間の参照強化

## 中期の重点項目

- 現在は試作段階に留まる CLI の拡張
- ノイズ特性評価や時間周波数解析ワークフローの拡充
- 外部科学技術 Python ライブラリとの相互運用ガイド拡張
- 数値アルゴリズムや物理前提に関する公開検証ノートの充実

## 長期的な方向性

- 事例とリファレンスを横断して探せるビジュアル・ナビゲーションの強化
- ドキュメント検証とサンプルコード検証の自動化拡張
- テーマ、検索品質、インタラクティブ表示の改善

## 公開トラッキング先

- [不具合報告・機能追加リクエスト用の簡易フォーム](https://forms.gle/c8jJaf9UCs5tb5cC8)
- [GitHub Issues](https://github.com/tatsuki-washimi/gwexpy/issues)
- [セキュリティポリシー](https://github.com/tatsuki-washimi/gwexpy/blob/main/SECURITY.md)。脆弱性情報はフォームや public issue には含めないでください。
- [GitHub Releases](https://github.com/tatsuki-washimi/gwexpy/releases)
- [更新履歴](changelog.md)

## 補足

このロードマップは開発方針の共有用であり、実装順序やリリース時期を保証するものではありません。
GWexpy は研究用途のソフトウェアであり、今後の状況に応じて優先度が見直される可能性があります。
