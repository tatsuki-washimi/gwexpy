# 17-19 Responsive Table Probe Result

- Date: 2026-04-20
- Scope: local built HTML under `docs/_build/html` only
- Allowed runtime:
  - `node`
  - `/usr/bin/google-chrome`
- Probe script: `/tmp/gwexpy_17_19_probe.mjs`
- Raw evidence bundle: `/tmp/gwexpy-17-19-evidence/`

## Conclusion

`17-19` の再検証では、`320px` の狭幅表示と `1920px+` 相当の広幅表示 (`2560x1080`, `3440x1440`) の両方で、右側の gray gutter は再現しなかった。

ページ本文レベルでは `bodyScrollWidth == bodyClientWidth` が維持され、page-level horizontal overflow は確認されなかった。テーブルが viewport より広いページでも、`.wy-table-responsive` または notebook 出力コンテナ `.output_area.rendered_html` が `overflow-x: auto` を持ち、body 側へ overflow を伝播させていない。

## Verification Method

1. `conda run -n gwexpy sphinx-build -E -b html -D nbsphinx_execute=never docs docs/_build/html`
2. `node /tmp/gwexpy_17_19_probe.mjs`

Probe は `file://` で以下のローカル生成ページを直接開いて計測した。

- `file:///home/washimi/work/gwexpy/docs/_build/html/web/ja/user_guide/io_formats.html`
- `file:///home/washimi/work/gwexpy/docs/_build/html/web/ja/user_guide/tutorials/intro_table.html`
- `file:///home/washimi/work/gwexpy/docs/_build/html/web/ja/user_guide/tutorials/matrix_frequencyseries.html`

## Viewports Used

- Narrow mobile:
  - `320x900`
- `1920px+` class wide displays:
  - `2560x1080`
  - `3440x1440`

## Recorded Evidence

### 320px class checks

#### `io_formats`

- Page: `file:///home/washimi/work/gwexpy/docs/_build/html/web/ja/user_guide/io_formats.html`
- Screenshot: `/tmp/gwexpy-17-19-evidence/mobile_320x900/io_formats.png`
- Body:
  - `bodyClientWidth = 320`
  - `bodyScrollWidth = 320`
- Article:
  - `articleClientWidth = 268`
  - `articleScrollWidth = 268`
- Table status:
  - max table width `576`
  - `.wy-table-responsive` が `overflow-x: auto`
  - body-level overflow なし
- Gray gutter:
  - `grayGutterHeuristic = false`
  - スクリーンショット上でも右側 gray gutter は見えない

#### `intro_table`

- Page: `file:///home/washimi/work/gwexpy/docs/_build/html/web/ja/user_guide/tutorials/intro_table.html`
- Screenshot: `/tmp/gwexpy-17-19-evidence/mobile_320x900/intro_table.png`
- Body:
  - `bodyClientWidth = 320`
  - `bodyScrollWidth = 320`
- Article:
  - `articleClientWidth = 268`
  - `articleScrollWidth = 276`
- Table status:
  - max dataframe width `258.25`
  - notebook container `.output_area.rendered_html` が `overflow-x: auto`
  - dataframe 自体の viewport overflow は再現せず
- Interpretation:
  - article 内部には scrollable な notebook block が残るが、body へは伝播していない
- Gray gutter:
  - `grayGutterHeuristic = false`
  - スクリーンショット上でも右側 gray gutter は見えない

#### `matrix_frequencyseries`

- Page: `file:///home/washimi/work/gwexpy/docs/_build/html/web/ja/user_guide/tutorials/matrix_frequencyseries.html`
- Screenshot: `/tmp/gwexpy-17-19-evidence/mobile_320x900/matrix_frequencyseries.png`
- Body:
  - `bodyClientWidth = 320`
  - `bodyScrollWidth = 320`
- Article:
  - `articleClientWidth = 268`
  - `articleScrollWidth = 317`
- Table status:
  - max dataframe width `258.25`
  - notebook container `.output_area.rendered_html` が `overflow-x: auto`
  - dataframe 自体の viewport overflow は再現せず
- Interpretation:
  - article 内部の notebook content はやや広いが、body-level overflow にはなっていない
- Gray gutter:
  - `grayGutterHeuristic = false`
  - スクリーンショット上でも右側 gray gutter は見えない

### 1920px+ class checks

#### `2560x1080` representative checks

##### `io_formats`

- Screenshot: `/tmp/gwexpy-17-19-evidence/ultrawide_2560x1080/io_formats.png`
- Body:
  - `bodyClientWidth = 2545`
  - `bodyScrollWidth = 2545`
- Article:
  - `articleClientWidth = 696`
  - `articleScrollWidth = 696`
- Gray gutter:
  - `grayGutterHeuristic = false`

##### `intro_table`

- Screenshot: `/tmp/gwexpy-17-19-evidence/ultrawide_2560x1080/intro_table.png`
- Body:
  - `bodyClientWidth = 2545`
  - `bodyScrollWidth = 2545`
- Article:
  - `articleClientWidth = 696`
  - `articleScrollWidth = 696`
- Gray gutter:
  - `grayGutterHeuristic = false`

##### `matrix_frequencyseries`

- Screenshot: `/tmp/gwexpy-17-19-evidence/ultrawide_2560x1080/matrix_frequencyseries.png`
- Body:
  - `bodyClientWidth = 2545`
  - `bodyScrollWidth = 2545`
- Article:
  - `articleClientWidth = 696`
  - `articleScrollWidth = 696`
- Gray gutter:
  - `grayGutterHeuristic = false`

#### `3440x1440` confirmation check

##### `io_formats`

- Screenshot: `/tmp/gwexpy-17-19-evidence/ultrawide_3440x1440/io_formats.png`
- Body:
  - `bodyClientWidth = 3425`
  - `bodyScrollWidth = 3425`
- Article:
  - `articleClientWidth = 696`
  - `articleScrollWidth = 696`
- Gray gutter:
  - `grayGutterHeuristic = false`
- Note:
  - 右側に広い背景余白は残るが、fixed-width content column と theme 背景によるもの
  - `bodyScrollWidth == bodyClientWidth` なので overflow 起因の gray gutter ではない

## Implementation Note

この再検証に先立ち、notebook 由来の pandas dataframe を通常テーブルと同様に body overflow から隔離するため、`docs/_static/custom.css` に `.output_area.rendered_html` 向けの `overflow-x: auto` と `table.dataframe` のモバイル調整を追加した。

## Closure Statement

少なくとも `320px` と `1920px+` 相当の harsh condition において、`17-19` の対象だった「表起因の body-level overflow / right gray gutter」は再現しない。内部 scroll container は残るが、これは意図された containment であり、gray gutter 再発の証拠ではない。
