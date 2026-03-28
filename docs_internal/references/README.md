# docs_internal/references/

参考文献・仕様書・内部資料のアーカイブ。Git 管理されているのは `.md` ファイルのみで、PDF や大容量バイナリは `.gitignore` で除外されている。

## ディレクトリ構成

| ディレクトリ / ファイル | 内容 |
| --- | --- |
| `data-analysis/` | 信号処理・データ解析に関する学術文献・発表資料 |
| `data-analysis/HHT/` | Hilbert-Huang Transform（HHT）関連の論文・発表スライド |
| `data-analysis/STLT/` | 短時間線形変換（STLT）関連文献 |
| `data-analysis/noise-budget/` | LIGO/Virgo/KAGRA のノイズバジェット報告書 |
| `data-analysis/noise-subtraction/` | DeepClean 等のノイズ減算手法に関する論文・発表資料 |
| `data-analysis/non-Gaussian/` | 非ガウスノイズ解析に関する文献 |
| `data-analysis/stat-info/` | 統計的情報論に関する参考資料 |
| `file-io/` | 計測器・ファイル形式仕様書（ADU、SEED、GWF 等） |
| `gui-app/` | GUI 関連の診断ソフトウェア仕様書 |
| `main_E.pdf` | 主要参考文献（概要書） |

## .gitignore の設定

```gitignore
docs_internal/references/*        # PDF・バイナリはすべて除外
!docs_internal/references/**/*.md # .md ファイルのみ追跡
```

`deepclean_extracted/` 等の大容量展開ディレクトリ（889 MB 超）も上記ルールで除外される。手元で再現するには元の PDF から展開すること。

## 追加・更新のルール

- **PDF や大容量ファイルはコミットしない** — `.gitignore` により除外済み。
- 新しいサブトピックを追加する場合は、このファイルの表も更新する。
- 出所・ライセンスが明確な文献のみを格納する。
