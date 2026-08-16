# gwexpy v0.2.0 実装計画・現状報告

## 現状と完了条件

- 実装は未着手で、現在は `/mplan` による計画策定段階。
- リポジトリ、ROADMAP、既存テスト、依存関係を調査し、Drafter／Critic／Auditor／Scientist／Final Reviewer のレビューを完了。最終判定は **APPROVED**。
- #513 は「整数 GPS ns を保持する」方針で確定済み。
- Plan Mode のため、コード、計画ファイル、GitHub は変更していない。
- 現ブランチは `origin/main` より1マージ遅れているため、実装時は最新 `origin/main` から専用の統合 worktree を作る。既存の untracked `.codegraph/` は触らない。
- 実装期間は 2026-08-17〜2026-10-12。必須契約を満たし、追加候補は期限までに全ゲートを通過したものだけ採用する。

## 実装内容

### コンテナ契約：#612／#637／#676

- #637 は `NDArrayOperatorsMixin` と内部 `ndarray` による composition 方式で試作する。
- `shape`、`dtype`、値、スライス、代入、反復、copy、astype、real/imag、conj、転置、reshape、`np.asarray(matrix)`、`matrix.view(np.ndarray)` を互換契約とする。
- `ndarray` としての identity、typed/raw view、buffer、`.base/.data/.flags/.strides/__array_interface__` は非契約と明記する。
- ufunc は通常呼び出しだけをサポートし、`out=`、reduction系、複数出力、未対応の `where` は明示的に `TypeError`。破壊的更新は atomic な in-place dunder のみに限定する。
- 加減算の単位互換、乗除算、sqrt/power、dimensionless 必須の log/exp、比較・predicate の戻り値を Astropy 整合の規則として固定する。
- #612 は型付きの正準マニフェストを既存 matrix テストから参照し、各セルに期待クラス・単位・メタデータまたは正確な例外を記述する。xfail は使わず、セル数も固定する。
- 変更前を B0、#637 候補を B1 として保存する。#637 を延期した場合は B0 をリリース基準とし、台帳へ `adopted: false` を記録する。
- #676 は最新 `origin/main` の固定 SHA と、#637 の凍結済み実行時ファイルだけを適用した隔離候補を比較する。候補パッチの SHA-256、環境、全生サンプル、中央値、MAD を保存する。
- 3回 warm-up、独立プロセス7回、各測定250 ms以上。ばらつきが5%超なら最大3回再測定し、安定しなければ採用しない。
- 採用条件は、各操作の中央値差 `≤ max(基準値×20%, 10 µs)`、比率の幾何平均 `≤ 1.10`、RSS増分 `≤ max(基準値×10%, 8 MiB)`。

### GPS時刻、HDF5、provenance：#513／#402／#508

- TimeSeries系へ keyword-only `t0_ns` を追加する。総GPSナノ秒の符号付きint64として `0 <= t0_ns <= 2^63-1` を許可し、bool、範囲外、既存 `t0`/`epoch` と不一致の併用は拒否する。
- 内部値は `_gwex_t0_gps_ns`、公開参照は read-only `t0_gps_ns: int | None`。LIGOTimeGPSまたは `t0_ns` 入力は exact、float秒入力は ties-to-even で最近傍nsへ量子化して precision 状態を `quantized` とする。
- exact 値は copy、pickle、HDF5で保持する。連続スライスは `slice.indices()` 後の offset と `dt` を `Fraction` で評価し、offsetが整数nsなら exact を維持する。それ以外の通常スライスは量子化し、不規則軸、step≠1、fancy/boolean indexing は exact slot を破棄する。
- 1 ns隣接時刻の保証範囲は、exact入力からの copy／pickle／HDF5、および整数nsオフセットの連続スライスからMNEへの変換までとする。表面上の既存float epoch APIは維持する。
- HDF5のGWpyネイティブデータセットは変更せず、ファイルrootまたは包含groupに単一の `_gwexpy_sidecar_json_v1` を置く。dataset attributeや兄弟datasetは追加しない。
- sidecar形式は `{"schema":"gwexpy.hdf5.sidecar","version":1,"objects":{"relative/path":{"metadata":...,"provenance":...}}}`。POSIX相対パス、JSONプリミティブ、NumPy scalar正規化、Astropy単位の正準表現だけを許可する。
- 書き込みは検証後に置換、appendは無関係なobjectを維持して同一pathだけ置換する。欠落は空扱い、壊れたJSONや未知versionは `ValueError`。
- GWpy単独ではcore payloadを読め、GWexpyではsidecarからexact epoch、metadata、provenanceを復元する。
- 対象は TimeSeries、FrequencySeries、Spectrogram、StateVector。SegmentList/DataQualityFlagは既存writer経路がある場合だけ対応し、なければ capability テストをN/Aとして新規writerは作らない。
- #508 の `.provenance` はJSON mappingとし、schema、algorithm、全parameter、RNG方式・bit generator・seed、software versionを記録する。外部RNGは `caller_managed`。
- `GauChResult.metadata is GauChResult.provenance` を保証する。copy、slice、単項演算、pickle、HDF5復元では内容をdeep-copyした新しいmappingを作り、二項演算は左右入力をdeep-copyした決定的な operation treeを生成する。既存の戻り値型は変えない。

### 期限付き追加候補

- #409：GWpy 4.0.1以降の capability/version を監査し、既存 `median_mean` を重複登録しない。
- #410：独立なχ²₂／指数分布標本だけを対象に、奇数 `N` は `α_N = Σ((-1)^(k+1)/k, k=1..N)`、偶数は `α_N=α_(N-1)`、補正値は `median/α_N` とする。bool、非整数、`N<=0` は拒否する。黄金値は N=1/2: 1、N=3/4: 5/6、N=5/6: 47/60、極限は ln 2。重複・相関標本への非適用を文書化し、科学レビューの承認がなければ延期する。[FINDCHIRP Appendix B](https://arxiv.org/abs/gr-qc/0509116)
- #411／#412：長形式の必須列を `start_gps_ns`, `duration_ns`, `source_channel`, `response_channel`, `frequency_hz`, `coupling_factor`, `coupling_factor_unit` に固定する。有限性、非負値、int64 overflow、channel、単位を検証する。任意の `estimate_kind` は既定 `measurement` または `upper_limit`。upper limit時だけ `limit_method` と `0<confidence_level<1` を許可し、無効binはNaN/nullではなく行ごと除外する。
- #588：filesystem pathの並列処理は spawn型 `ProcessPoolExecutor`。`parallel` と `nproc` の併用は常に `TypeError`。None/False/1は直列、Trueは `min(cpu_count, span数, 8)`、整数は2〜8、0/負数は `ValueError`、float/string/objectは `TypeError`。未知spanの並列要求はsubmit前に拒否し、結果は `(start_ns,end_ns,input_index)` 順。失敗時は残作業をcancelし部分結果を返さない。
- #590：公開入口は `dataset_options={...}` のみ。許可キーは chunks、compression、compression_opts、shuffle、fletcher32。`chunks=False` はh5pyへ渡す前に `None` へ正規化し、filterとの併用を拒否する。compressionは None/gzip/lzf、gzip levelは0〜9、lzf optionsはNone。tuple chunkはrank、正整数、shape上限を検証し、zero-length次元は `chunks=True` のみ許可する。未知option、scaleoffset、利用不能codecは作成前に拒否する。

## 実行順序とモデル割振り

| 担当 | 用途 |
|---|---|
| Luna | 全ての実装、テスト作成・実行、benchmark、通常の不具合修正 |
| Terra | 各laneの個別設計・コード・テストレビュー |
| Sol | 複数laneを横断する統合レビュー、またはLunaとTerraで解決できない難所だけ |
| root | 進行管理、write scope管理、統合、最終検証の取りまとめ |

- Lunaは利用可能なモデルとして扱い、TerraやSolへ通常実装を代替させない。
- 同時サブエージェントは最大3。最新 `origin/main` の統合worktree内で担当ファイルを排他的に割り当てる。
- 8月17日から基盤契約、B0、HDF5互換baseline、#676計測基盤を凍結する。
- #637候補とB1を隔離して作り、9月7日に互換表・benchmark・fallbackを含む判断資料を完成、9月14日に採用／延期を正式決定する。
- 採用しない場合は#637のruntime変更だけを戻し、B0、契約テスト、fallback文書、benchmark証拠は残す。
- 8月31日に追加候補の中間判定、9月28日に新規追加を停止、10月4日に証拠を凍結、10月5日時点で未完・未承認の候補を自動延期し、10月12日をv0.2.0目標日とする。

## 検証・リリース判定

- 各laneのfocused test後、`python scripts/ci/run_gate.py pr-fast`、`io-contract`、`interop-mne`、`io-gwf` を実行する。
- 統合時に `pytest tests/`、`ruff check`、`ruff format --check`、`mypy`、ドキュメントbuildを通す。
- 代表環境は、最小構成 Python 3.11／NumPy 1.23.2／Astropy 5／GWpy 4.0、現行構成 Python 3.12／NumPy 2／最新Astropy・GWpy 4.x、最新NumPy 1.x。全組合せはCIで確認する。
- staleな `.[dev,test,docs]` extrasには依存せず、既存環境または必要パッケージを明示して検証する。
- core/shared gateの失敗はリリースを停止する。追加候補だけの失敗、性能不合格、科学承認不足は該当機能を延期する。
- #413のrelease evidence、ROADMAP結果、各issueの `complete`／`partial`／`blocked`、#637採否をcompletion ledgerへ記録し、Solの統合レビュー後に最終判定する。
- #581は#676に必要な最小benchmark基盤だけを対象とし、#403のnproc移行、version tag、release公開、GitHub issue操作は今回の自動実行範囲外とする。
- 計画承認後、同一内容を `docs/plans/2026-08-16-v020-implementation-plan.md` と `~/.claude/plans/2026-08-16-v020-implementation-plan.md` に保存する。コミット、push、tag、release、GitHub更新は個別の許可後に実施する。
