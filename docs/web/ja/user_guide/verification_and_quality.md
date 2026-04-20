---
myst:
  html_meta:
    description: "GWexpy で notebook、direct I/O、アルゴリズム監査、coverage がどのように公開可視化されているかと、その読み方を整理します。"
---

# 検証と品質の見方

> **ページ種別:** 透明性ガイド

このページでは、`gwexpy` が現在どの種類の検証シグナルを公開しているか、その根拠がどこにあるか、そしてその限界をどう読むべきかを整理します。

ここで示すのは「すべての機能が一様に検証済み」という主張ではありません。Notebook、direct I/O、アルゴリズム監査、リポジトリ全体の coverage について、それぞれ別の根拠に辿れるようにするための案内ページです。

## このページでわかること

| 項目 | 内容 |
| --- | --- |
| **対象読者** | チュートリアル、I/O 形式、アルゴリズムがどの程度公開根拠に支えられているかを確認したい方 |
| **前提** | user guide を読める程度の基本知識があれば十分です |
| **こんなときに読む** | notebook がどう検証されるか知りたい、I/O 対応表とテストの関係を見たい、監査ノートの入口を知りたい、coverage 表示の意味と限界を確認したい |
| **検索ヒント** | verification, quality, coverage, notebook policy, SUPPORTED_IO_MATRIX, codecov, audit trail |

**検索ヒント:** verification, quality, coverage, notebook policy, SUPPORTED_IO_MATRIX, codecov, audit trail

:::{important}
**このページは「透明性の地図」であって、一括保証ではありません**

検証の方法は対象ごとに異なります。Notebook の中には CI で全実行されるものもあれば、重い notebook のように構造確認中心のものもあります。optional dependency を持つテストは、環境によって skip される場合もあります。
:::

## 公開されている根拠の入口

| 対象 | 公開ソース | 何がわかるか |
| --- | --- | --- |
| Notebook チュートリアル | [Notebook Policy](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/NOTEBOOK_POLICY.md) | `Light` / `Heavy` / `Display-only` の分類と、CI がそれぞれをどう扱うか |
| direct I/O 形式 | [SUPPORTED_IO_MATRIX](https://github.com/tatsuki-washimi/gwexpy/blob/main/SUPPORTED_IO_MATRIX.md) | どの公開 format 群に、どのテストファイルが対応づけられているかと、どこに optional backend があるか |
| アルゴリズム監査 | [検証済みアルゴリズム](validated_algorithms.md) | 数値許容誤差、前提条件、監査証跡への導線 |
| リポジトリ全体の coverage | [README の codecov バッジ](https://github.com/tatsuki-washimi/gwexpy) と、そのリンク先である [Codecov ダッシュボード](https://codecov.io/gh/tatsuki-washimi/gwexpy) | リポジトリ全体の line coverage がどこで公開されているかを示す入口。feature 単位の証明ではなく、全体傾向のシグナルとして使います |

## Notebook の検証方針

公開 notebook の扱いは、リポジトリ内の [Notebook Policy](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/NOTEBOOK_POLICY.md) に基づきます。

現時点の公開方針は次のとおりです。

- **Light** notebook は、Notebook Policy 上では CI で `papermill` により全実行される想定です。
- **Heavy** notebook は CI 対象ではありますが、全セル実行を常に保証するものではなく、構造確認や収集確認に寄せて扱われます。
- **Display-only** notebook は整えた出力の保持を重視しており、「CI で毎回全実行される notebook」と同義ではありません。
- 公開 notebook の正本は `docs/web/{en,ja}/user_guide/tutorials/` に置かれます。

したがって、「公開 docs に notebook がある」こと自体は有益なシグナルですが、それだけで「あらゆる CI 経路で全セル実行される」とまでは読まない方が安全です。

## direct I/O の検証可視化

public direct I/O の検証可視化では、[SUPPORTED_IO_MATRIX](https://github.com/tatsuki-washimi/gwexpy/blob/main/SUPPORTED_IO_MATRIX.md) が主要な入口です。

この表は、たとえば次のような疑問に答えるときに使います。

- 「この format は公開対応としてどこまで見てよいか」
- 「この format claim はどのテストファイルに紐づいているか」
- 「この経路は optional backend に依存するのか」

[ファイル I/O 対応フォーマットガイド](io_formats.md) と合わせて読むと役割分担が明確です。

- user guide 側は「どう選ぶか」「どう呼ぶか」を説明し、
- matrix 側は「どのテストが根拠か」を示し、
- 備考が optional dependency や skip 条件を補います。

## coverage 表示の読み方と限界

`gwexpy` は [Codecov](https://codecov.io/gh/tatsuki-washimi/gwexpy) によって、リポジトリ全体の coverage シグナルを公開しています。`README.md` にも codecov の状態バッジとリンクがあり、公開された確認入口として辿れます。

ただし、この値は慎重に読む必要があります。

- 自動テスト全体の健康度をざっくり把握するには有用です。
- 一方で、すべてのアルゴリズム分岐、すべての notebook、すべての optional-backend 経路が同じ強さで実行されていることまでは保証しません。
- 実際の判断では、Notebook Policy、I/O matrix、アルゴリズム監査ノートと併読してください。

## このページが主張しないこと

- すべての公開 notebook が、すべての CI 実行で全セル再実行されるとは主張しません。
- すべての optional dependency が、すべての test 環境に入っているとは主張しません。
- [検証済みアルゴリズム](validated_algorithms.md) に書かれた個別の前提条件や許容誤差を、このページが置き換えるものではありません。
- repository-wide の line coverage を、そのまま feature 単位の科学的妥当性の証明に読み替えるべきだとは主張しません。

## 関連ページ

- [検証済みアルゴリズム](validated_algorithms.md)
- [ファイル I/O 対応フォーマットガイド](io_formats.md)
- [トラブルシューティング](troubleshooting.md)

## 次に読む

- [検証済みアルゴリズム](validated_algorithms.md) でアルゴリズム単位の前提条件、許容誤差、監査リンクを確認する
- [ファイル I/O 対応フォーマットガイド](io_formats.md) で利用者向けの format 選択と backend 条件を確認する
- [トラブルシューティング](troubleshooting.md) で実行時エラーから逆引きする
