# GWexpy SoftwareX論文投稿準備 - 統合作業計画

> **目的**: SoftwareX誌への論文投稿に向けて、リポジトリ整備・再現性保証・アーカイブ・投稿までを一貫して実行する。
> **使い方**: 上から順に作業を進め、各チェックボックスを完了したらチェックする。

---

## 現状サマリー

| 項目 | 現状 | 目標 |
|------|------|------|
| バージョン | `0.1.0b2` (ベータ) | `0.1.0` |
| 依存関係 | core に 80+ パッケージ | core を最小化、extras に分離 |
| GUI | `gwexpy/gui/` に完全実装 | core から除外、experimental |
| examples | 37個のノートブック | CI統合、論文用に厳選 |
| DOI | 未発行 | Zenodo経由で取得 |
| 連絡先 | README: `example.com` | GitHub Issues + Contact Form |
| 論文 | `docs/gwexpy-paper/main.tex` | Code metadata を実装と同期 |

---

## Step 1: リポジトリ整理

**目的**: Release に不要なファイル（開発ログ、AI作業ログ、個人用ツール等）を含めない。

| 推奨モデル | 工数 |
|-----------|------|
| **Haiku** | 0.5-1日 |

### 作業内容

- [ ] `docs/` 内の開発用作業プラン・レポート・AIログを `docs_internal/` または別ブランチへ移動
  ```bash
  git mv docs/dev_notes docs_internal/dev_notes
  git commit -m "Move internal dev notes out of docs for release"
  ```
- [ ] 不要ファイル（.vscode/.idea、巨大データ、個人用スクリプト）を `.gitignore` へ追加または `git rm`
- [ ] `tests/`, `examples/`, `notebooks/` は残す（再現性のため）
- [ ] examples は論文で使う 3-5 本に厳選

### 検証

- [ ] `git status` が clean
- [ ] README が参照可能

---

## Step 2: バージョン・メタデータの統一

**目的**: `pyproject.toml`、`CHANGELOG.md`、論文の Code metadata を一致させる。

| 推奨モデル | 工数 |
|-----------|------|
| **Haiku** (version変更) / **Sonnet** (CHANGELOG) | 0.5-1日 |

### 作業内容

- [ ] `pyproject.toml`: `version = "0.1.0"` に更新
- [ ] `CHANGELOG.md`: `[Unreleased]` → `[0.1.0] - YYYY-MM-DD` に変換、リリースノート追加
- [ ] `main.tex` の Code metadata C1 を `0.1.0` に修正

### 検証

```bash
pip install -e . && python -c "import gwexpy; print(gwexpy.__version__)"
# 出力: 0.1.0
```

---

## Step 3: GUI除外（コア依存からの分離）

**目的**: 論文で GUI を除外済みとするため、core から GUI 依存を削除し、extras に分離する。

| 推奨モデル | 工数 |
|-----------|------|
| **Sonnet** | 1-2日 |

### 作業内容

- [ ] `pyproject.toml`: GUI依存（PyQt5, pyqtgraph等）を `[gui]` extras に完全分離
- [ ] `.github/workflows/test.yml`: GUI テストを条件付き実行に変更
  ```yaml
  if: contains(github.event.head_commit.message, '[gui-test]')
  ```
- [ ] `README.md`: GUI は experimental と明記

### 検証

```bash
pip install -e .       # GUI依存なしでインストール成功
pip install -e .[gui]  # GUI依存ありでインストール成功
```

---

## Step 4: 依存関係整理（core/extras分離）

**目的**: core を最小限にして `pip install gwexpy` が軽量かつ失敗しにくくする。

| 推奨モデル | 工数 |
|-----------|------|
| **Opus** (分析) / **Sonnet** (実装) | 1-2日 |

### 現状分析

現在の `pyproject.toml` には **46パッケージ** が core dependencies に含まれており、多くがオプショナル機能向けです。コード内では既に `require_optional()`、`ensure_dependency()`、try-except パターンでオプショナルインポートが実装されています。

### 依存関係の分類

#### Core（必須 - 削除不可）
| パッケージ | 理由 |
|-----------|------|
| `numpy>=1.21.0,<2.0.0` | 数値計算の基盤 |
| `scipy>=1.7.0` | 信号処理・統計の基盤 |
| `astropy>=5.0` | 単位系・座標系・時刻処理 |
| `gwpy>=4.0.0,<5.0.0` | GWpy拡張の基盤 |
| `pandas>=1.3.0` | DataFrame操作（広く使用） |
| `matplotlib>=3.5.0` | 可視化の基盤 |
| `typing_extensions` | 型ヒント互換性 |
| `bottleneck` | 高速NaN処理 |
| `h5py` | HDF5読み書き |
| `igwn-segments` | 軽量、基本的なセグメント管理 |
| `ligotimegps`, `gpstime` | GPS時刻処理（基本機能） |

#### 既にオプショナル設計されている（コード内で `require_optional()` or try-except ガード済み）
| パッケージ | 使用箇所 | 推奨extras |
|-----------|----------|-----------|
| `obspy` | noise/obspy_.py | `[seismic]` |
| `mth5`, `mtpy`, `mt_metadata` | interop/mt_.py | `[seismic]` |
| `scikit-learn` | timeseries/decomposition.py（PCA/ICA） | `[analysis]` |
| `statsmodels`, `pmdarima` | timeseries/arima.py | `[analysis]` |
| `iminuit`, `emcee`, `corner` | fitting/*.py | `[fitting]` |
| `PyEMD (EMD-signal)` | timeseries/_spectral_special.py | `[analysis]` |
| `PyWavelets` | timeseries/_spectral_special.py | `[analysis]` |
| `control` | interop/control_.py | `[control]` |
| `nptdms` | timeseries/io/tdms.py | `[io]` |
| `gwinc` | noise/gwinc_.py | `[gw]` |
| `pygmt` | plot/geomap.py | `[plotting]` |
| `ligo.skymap` | plot/skymap.py | `[gw]` |
| `dcor` | timeseries/_statistics.py | `[analysis]` |
| `hurst`, `hurst-exponent`, `exp-hurst` | timeseries/hurst.py | `[analysis]` |

#### extras へ分離すべき GW Data Access パッケージ
| パッケージ | 理由 | 推奨extras |
|-----------|------|-----------|
| `lalsuite` | GUI loader のみで使用 | `[gw]` |
| `gwdatafind` | GWpy依存（直接使用なし） | `[gw]` |
| `gwosc` | GWpy依存（直接使用なし） | `[gw]` |
| `dqsegdb2` | セグメント検索（オプション） | `[gw]` |
| `dttxml` | io/dttxml_common.py（既に optional設計） | `[gw]` |

#### 削除候補
| パッケージ | 状態 |
|-----------|------|
| `dateparser` | import箇所なし → **削除** |

### 推奨 extras 構造

```toml
[project.optional-dependencies]
analysis = [
    "scikit-learn",
    "statsmodels",
    "pmdarima",
    "dcor",
    "hurst",
    "EMD-signal",
    "PyWavelets",
]

fitting = [
    "iminuit",
    "emcee",
    "corner",
]

control = [
    "control",
]

seismic = [
    "obspy",
    "mth5",
    "mtpy",
    "mt_metadata",
]

gw = [
    "lalsuite",
    "gwdatafind",
    "gwosc",
    "dqsegdb2",
    "dttxml",
    "gwinc",
    "ligo.skymap",
]

io = [
    "nptdms",
]

plotting = [
    "pygmt",
]

# Existing
audio = [
    "pydub",
    "tinytag>=1.10",
]

gui = [
    "PyQt5",
    "pyqtgraph",
    "qtpy",
    "sounddevice",
]

# Convenience
all = [
    "gwexpy[analysis]",
    "gwexpy[fitting]",
    "gwexpy[control]",
    "gwexpy[seismic]",
    "gwexpy[gw]",
    "gwexpy[io]",
    "gwexpy[plotting]",
    "gwexpy[audio]",
]
```

### 最小 core dependencies（提案）

```toml
dependencies = [
    "numpy>=1.21.0,<2.0.0",
    "scipy>=1.7.0",
    "astropy>=5.0",
    "gwpy>=4.0.0,<5.0.0",
    "pandas>=1.3.0",
    "matplotlib>=3.5.0",
    "typing_extensions",
    "bottleneck",
    "h5py",
    "igwn-segments",
    "ligotimegps",
    "gpstime",
]
```

**削減効果**: 46 → 12 パッケージ（約74%削減）

### 作業内容

- [ ] 現在の `_EXTRA_MAP` in `gwexpy/interop/_optional.py` をテンプレートとして使用
- [ ] `pyproject.toml` の `dependencies` を上記の 12 パッケージに削減
- [ ] `optional-dependencies` を上記の構造で定義
- [ ] 不要なパッケージ（`dateparser`）を削除
- [ ] `interop/_optional.py` の `_EXTRA_MAP` と同期

### 検証

```bash
pip install -e .                  # core のみ（12パッケージ）
pip install -e .[analysis]        # + 分析ツール
pip install -e .[fitting]         # + fitting（iminuit等）
pip install -e .[seismic]         # + 地震学データ
pip install -e .[gw]              # + GW data access
pip install -e "[all]"            # 全て
```

---

## Step 5: CI整備（notebooks + tests）

**目的**: 論文の Listing が CI で常時動くことを保証する（再現性）。

| 推奨モデル | 工数 |
|-----------|------|
| **Sonnet** | 1-3日 |

### 作業内容

- [ ] `notebooks/` ディレクトリ作成（論文用の最小実行可能例、synthetic/anonymized data使用）
- [ ] `.github/workflows/test.yml` に `nbmake` ジョブ追加:
  ```yaml
  - name: Run notebooks
    run: |
      pip install nbmake
      pytest --nbmake notebooks/ --nbmake-timeout=600
  ```
- [ ] ユニットテスト追加:
  - メタデータ不変性テスト（`to_obs` → `from_obs` の round-trip）
  - DTT XML: `products=None` で `ValueError` 発生テスト
  - API一致性テスト（`inspect.signature` でシグネチャ確認）
- [ ] GUI の heavy CI ジョブを分離または条件実行に

### 検証

- [ ] CI が green
- [ ] notebooks ジョブが artifact に出力図を保存

---

## Step 6: 論文 Listing と API 実装の整合

**目的**: 論文の Listing がそのまま実行でき、読者が notebook/例を動かせるようにする。

| 推奨モデル | 工数 |
|-----------|------|
| **Opus** (分析) / **Sonnet** (修正) | 2-4日 |

### 論文中のAPI使用例（要検証）

1. **Listing 1** (DTT XML読み込み):
   - `TimeSeriesDict.read(source, format="dttxml", channels=[...], products=["tf"])`
   - ✓ 実装確認済み（`products` 必須化）

2. **Listing 2** (転送関数ワークフロー):
   - `TimeSeriesMatrix.from_timeseries(list(tsdict.values()))`
   - `matrix.transfer_function(target="...", reference="...", mode="steady")`
   - `tf.plot_bode()`
   - ✓ `transfer_function()` 実装確認済み（steady/transient対応）
   - ⚠ `plot_bode()` メソッドの存在を確認必要

3. **Listing 3** (異種フォーマット融合・コヒーレンス):
   - `TimeSeries.read(format="gwf"|"win"|"wav")`
   - `matrix.coherence_ranking(target="...", band=(10,100))`
   - `coh.topk(n=3)`
   - `coh.plot_ranked(top_k=3)`
   - ⚠ これらのメソッド存在確認必要

### 作業内容

- [ ] 論文のListingで使用されているメソッドを全て抽出
- [ ] 実装側で各メソッドの存在・シグネチャを確認
- [ ] 不一致があれば修正（実装追加 or 論文修正）
- [ ] 各Listingを実行可能notebookとして作成
- [ ] `dttxml` reader のエラーメッセージ・ドキュメント改善

### 検証

- [ ] 全Listingをnotebookに転記し、CIで実行成功を確認
- [ ] examples の簡易スクリプトが exit code 0 を返す

---

## Step 7: README整備

**目的**: 論文との整合性、ユーザーがすぐ動かせる状態にする。

| 推奨モデル | 工数 |
|-----------|------|
| **Sonnet** (インストール) / **Haiku** (連絡先) | 0.5-1日 |

### 作業内容

- [ ] インストールセクションを core / extras に分ける
- [ ] **連絡先**: メールアドレス直接記載を削除 → 以下に置き換え:
  ```markdown
  ## Support / Contact

  For questions, bug reports, or feature requests:
  - **GitHub Issues**: https://github.com/[username]/gwexpy/issues (recommended)
  - **Contact form**: [Google Form URL] (for private inquiries)

  For academic citations and correspondence, see the published paper.
  ```
- [ ] Examples / Notebooks へのリンク追加
- [ ] GUI は experimental と明記
- [ ] `CITATION.cff` と `LICENSE` をリポジトリに追加

### 検証

- [ ] README の手順で新しい環境にインストールし、notebooks が起動する

---

## Step 8: Release作成

**目的**: GitHub Release を作成し、Zenodo アーカイブの準備をする。

| 推奨モデル | 工数 |
|-----------|------|
| **Haiku** | 0.5日 |

### 作業内容

- [ ] `release/v0.1.0` ブランチを作成:
  ```bash
  git checkout -b release/v0.1.0
  ```
- [ ] 最終確認後、タグを作成・push:
  ```bash
  git tag v0.1.0
  git push origin release/v0.1.0
  git push origin v0.1.0
  ```
- [ ] GitHub の Releases で `v0.1.0` を Draft → Release（説明に CHANGELOG を引用）

### 推奨付加ファイル

- [ ] `.zenodo.json` を追加:
  ```json
  {
    "title": "GWexpy: Extending GWpy for gravitational-wave detector commissioning",
    "upload_type": "software",
    "description": "...",
    "creators": [{ "name": "Washimi, Tatsuki", "affiliation": "NAOJ" }],
    "license": "MIT",
    "keywords": ["gwpy", "gravitational-wave", "commissioning"]
  }
  ```

---

## Step 9: Zenodo連携・DOI取得

**目的**: 論文の Code metadata C3 に恒久識別子（DOI）を入れる。

| 推奨モデル | 工数 |
|-----------|------|
| 手動作業 | 0.5-1日 |

### 作業内容

- [ ] Zenodo アカウントで GitHub Integration を有効化
- [ ] 当該リポジトリを ON に設定
- [ ] GitHub Release を作成すると Zenodo が自動でアーカイブし DOI を発行
- [ ] 発行された DOI を以下に反映:
  - [ ] `main.tex` の Code metadata C3
  - [ ] `README.md` に DOI badge 追加

### 検証

- [ ] Zenodo で DOI が resolve され、アーカイブが確認できる
- [ ] Zenodo page にコードスナップショット（zip）と metadata がある

---

## Step 10: arXiv投稿

**目的**: preprint を公開し、関係者への共有準備をする。

| 推奨モデル | 工数 |
|-----------|------|
| 手動作業 | 0.5日 |

### 作業内容

- [ ] LaTeX の最終版（preprint用）を準備、図は `figures/` に入れる
- [ ] `make tex` または `pdflatex` で PDF を生成し確認
- [ ] arXiv のカテゴリ（`astro-ph.IM` / `physics.data-an` 等）を選択
- [ ] upload（source タイプ：LaTeX、PDF とソースを添付）→ arXiv ID を取得

### 注意点

- SoftwareX は preprint を許容している
- arXiv 掲載時点で repository と DOI が揃っていると好ましい

---

## Step 11: 関係者への連絡

**目的**: upstream / community に情報共有し、互換性や協力の機会を確保する。

| 推奨モデル | 工数 |
|-----------|------|
| 手動作業 | 0.5日 |

### 連絡先（優先順）

1. **Duncan M. Macleod（GWpy lead）** — 直接メール
2. **GWpy 開発チーム** — GitHub Issue または Discussion
3. **BruCo の主要関係者**（言及している場合）— FYI
4. **主要なコラボ機関（NAOJ パートナー等）** — 共有＆謝辞の確認
5. **（オプション）研究コミュニティ**: mailing list, Slack, Twitter

### タイミング

- **arXiv 投稿後**（preprint URL がある状態で送る）

### メールテンプレート

```
Subject: GWexpy — experimental extension for GWpy (preprint + repo)

Dear Duncan,

I wanted to let you know about GWexpy, an experimental extension for GWpy
focused on detector commissioning workflows.

Repository: https://github.com/tatsuki-washimi/gwexpy
Preprint: https://arxiv.org/abs/XXXX.XXXXX
Zenodo (snapshot): https://doi.org/10.5281/zenodo.xxxxx

If you have any feedback about API compatibility or documentation,
I'd greatly appreciate it.

Best regards,
Tatsuki Washimi
```

---

## Step 12: SoftwareX投稿

**目的**: journal 投稿（SoftwareX の OSP）を行う。

| 推奨モデル | 工数 |
|-----------|------|
| 手動作業 | 1-3日 |

### 作業内容

- [ ] Elsevier / SoftwareX の LaTeX テンプレート（`elsarticle`）に合わせる
- [ ] **Code metadata (C1-C9)** を必ず記載:
  - C1: `0.1.0`（pyproject.toml と一致）
  - C3: Zenodo DOI
  - C9: `tatsuki.washimi@nao.ac.jp`（SoftwareX要件）
- [ ] 実行可能な notebooks / examples を supplement として用意し、Zenodo アーカイブに含める
- [ ] Figures は本稿に表示、追加の詳細図は GitHub/Docs に置く
- [ ] Cover letter に「Zenodo DOI」「GitHub URL」「examples の実行手順」を記載
- [ ] Submit

---

## Step 13: （採択後）PyPI / conda-forge

**結論**: 投稿前は必須ではない。採択後かユーザー拡大を考える段階で行う。

| 推奨モデル | 工数 |
|-----------|------|
| **Sonnet** (PyPI) / 手動作業 (conda-forge) | PyPI: 1日、conda-forge: 数日〜数週間 |

### PyPI（比較的簡単）

- [ ] `pyproject.toml`（PEP 517/518 準拠）を整備
- [ ] `python -m build` → `twine upload dist/*`

### conda-forge（労力大）

- [ ] feedstock 作成、CI 設定、メンテが必要
- [ ] 研究パッケージとしては採択後に検討で可

---

## 最終チェックリスト（投稿直前）

- [ ] `pyproject.toml` と `main.tex`（C1）が同じバージョン（`0.1.0`）
- [ ] Zenodo DOI を取得し `main.tex`（C3）に記載
- [ ] `CITATION.cff` と `LICENSE` をリポジトリに追加
- [ ] `notebooks/` のすべてを CI で成功させる（nbmake）
- [ ] README に install / examples / citation を整備
- [ ] GUI を core から除外 or experimental 明記
- [ ] 論文の Listing がそのまま notebooks で再現可能
- [ ] README の連絡先が GitHub Issues + Contact Form
- [ ] arXiv 用 PDF とソースを準備しアップロード
- [ ] Duncan 氏および GWpy チームに arXiv URL + DOI を共有
- [ ] `pip install gwexpy` が成功（core のみ）
- [ ] `pip install gwexpy[gui]` が成功（extras付き）
- [ ] CI が全て green

---

## 付録A: モデル選択ガイドライン

| モデル | 推奨用途 | 特徴 |
|--------|----------|------|
| **Opus** | アーキテクチャ設計、API整合性分析、複雑な依存関係分析 | 深い推論、長文脈理解 |
| **Sonnet** | コード実装、CI修正、ドキュメント執筆、テスト作成 | バランス良好、コスト効率 |
| **Haiku** | 単純な修正（バージョン変更、連絡先修正）、フォーマット調整 | 高速、低コスト |
| **Gemini** | 長文ドキュメントの要約、大量ファイルの一括分析 | 長文脈ウィンドウ |
| **GPT-4** | 代替オプション（Anthropic系が使えない場合） | 汎用性 |

---

## 付録B: 推奨作業スケジュール

```
Week 1:
├── Day 1:   Step 1 (リポジトリ整理)
├── Day 2:   Step 2 (バージョン統一) + Step 3 (GUI除外) 開始
├── Day 3-4: Step 3 完了 + Step 4 (依存関係整理)
└── Day 5:   Step 5 (CI整備) 開始

Week 2:
├── Day 1-2: Step 5 完了 + Step 6 (API整合) 開始
├── Day 3-4: Step 6 完了
└── Day 5:   Step 7 (README整備)

Week 3:
├── Day 1:   Step 8 (Release作成) + Step 9 (Zenodo/DOI)
├── Day 2:   Step 10 (arXiv投稿)
├── Day 3:   Step 11 (関係者連絡)
└── Day 4-5: Step 12 (SoftwareX投稿) + 最終検証
```

---

## 付録C: 重要ファイル一覧

| ファイル | 役割 |
|----------|------|
| `pyproject.toml` | バージョン、依存関係 |
| `CHANGELOG.md` | リリースノート |
| `README.md` | ドキュメント |
| `.github/workflows/test.yml` | CI設定 |
| `.github/workflows/docs.yml` | ドキュメントCI |
| `gwexpy/gui/` | GUI実装（除外対象） |
| `examples/` | サンプルnotebook |
| `notebooks/` | 論文用例（要作成） |
| `gwexpy/timeseries/io/dttxml.py` | DTT XML reader |
| `docs/gwexpy-paper/main.tex` | 論文本体 |
| `.zenodo.json` | Zenodo metadata |
| `CITATION.cff` | 引用情報 |

---

## 付録D: 推奨 Issue タイトル（そのまま貼れる）

1. `pyproject.toml` と `CHANGELOG.md` の version 整合を取る（align with release v0.1.0）
2. Create GitHub Release v0.1.0 and archive to Zenodo (obtain DOI)
3. Remove GUI from core dependencies / mark GUI as experimental (move tests)
4. Make paper listings executable: align API names between `main.tex` and implementation
5. Add examples/notebooks + CI job (nbmake) to verify manuscript examples
6. Update README: core vs extras installation & support contact
7. Improve dttxml reader error messages and add unit tests for `products` requirement
8. Add metadata round-trip unit tests (ObsPy/PyTorch)
9. Add bench/benchmarks notebooks for coherence/CSD scaling
10. Make `scripts/run_all_notebooks.py` environment-agnostic and robust

---

## 付録E: 戦術的アドバイス

* **最重要**は「論文に書いてある API と実装を一致させること」と「再現可能性（notebook が CI で走る）」。査読者は Listing/Examples を実行しようとするため、ここが最初に壊れていると信頼性を失う。
* **次に重要**なのは「Code metadata の整合（version/DOI）」。SoftwareX が恒久識別子を強く求めるため。
* GUI は**ビルド／依存を壊しやすい要素**なので、初回リリースから外すことは投稿採択確率を上げる。
* 依存関係の整理（core/extras）を先に行うと CI の安定化、インストール容易化につながるため優先度は高め。
* **最初の Release は"最小かつ整った"形で**作ること（不要な internal 資料を入れない）。
* **コミュニケーションは短く・礼儀正しく**（arXiv URL + GitHub + DOI を添える）。
