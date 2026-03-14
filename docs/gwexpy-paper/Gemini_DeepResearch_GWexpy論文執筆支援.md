# **SoftwareX 投稿に向けた GWexpy 原稿の包括的批判および改善要求レポート**

## **1\. 査読の全体展望と致命的欠陥の総括**

重力波データ解析用パッケージ「GWpy」を実験現場向けに拡張した「GWexpy」に関する原稿（GWexpy\_paper.pdf および main.tex）ならびに作業計画（原稿以外の作業ToDo.md）の包括的な査読および評価を行う 1。重力波干渉計（Advanced LIGO、Advanced Virgo、KAGRAなど）のコミッショニング現場において、多種多様なフォーマットやチャンネル間でメタデータ（単位、サンプリングレート、タイムスタンプなど）が喪失するという課題は、観測運用において極めて深刻な問題であり、本ソフトウェアの開発動機は時宜を得たものである 1。

しかしながら、『SoftwareX』誌への「Original Software Publication (OSP)」としての採択を目指すにあたり、現在の原稿にはジャーナルの厳格な規定に対する違反、ソフトウェア工学的な設計方針（アーキテクチャ）に関する学術的論証の欠如、および重力波データ解析エコシステムに既に存在する類似ツール群との差別化の甘さが散見される 1。SoftwareXは単なるソフトウェアのマニュアルや宣伝記事を掲載する場ではなく、ソフトウェアがどのように科学的発見を推進し、その設計がいかにして研究の再現性や他分野への再利用性（Potential for reuse）を担保しているかを厳しく審査する学術誌である 8。

現在の原稿は、全体的な文字数や構成の枠組みとしてはOSPの制限（最大6ページ、約3000語）に収まるよう設計されているものの 12、その内容はパッケージの表面的な機能紹介に留まっており、査読者が最も重視する「なぜ既存のGWpyデータ構造（例：TimeSeriesDict）では不可能なのか」「LIGOコミュニティで標準的に使用されている既存のパーサーライブラリ（dttxml 等）と何が違うのか」という技術的な問いに対する回答が完全に欠落している 1。本報告書では、対象ジャーナルの規定、ソフトウェア工学のベストプラクティス、および重力波実験コミュニティの現状という3つの観点から、本原稿を採択可能な水準へと引き上げるための極めて厳格かつ具体的な改稿要件を提示する。

## **2\. SoftwareX 誌の出版規定への適合性に関する厳密な評価**

SoftwareXの編集委員会は、規定のフォーマットに従っていない原稿を査読プロセスに回すことなく、即座にデスクリジェクト（Desk Rejection）とする方針をとっている 7。現在の提出物 main.tex には、組版テンプレートおよび必須メタデータ表の構成において致命的な違反が存在するため、直ちに以下の修正を行う必要がある。

### **2.1. 公式 LaTeX テンプレートの適用と書式要件**

現在の原稿は \\documentclass\[11pt\]{article} という一般的なLaTeXクラスを使用しているが 1、これは明確な規定違反である。SoftwareXへの投稿においては、Elsevierの公式テンプレートである elsarticle クラス（具体的には \\documentclass\[preprint,12pt, a4paper\]{elsarticle} または提供された専用テンプレート）の使用が義務付けられている 18。

elsarticle への移行は、単なるテキストの流し込みでは完了しない。特にフロントマター（著者名、所属、連絡先等の記述）において、\\author, \\address, \\corref などの専用コマンドを用いた厳密な構造化が要求される 22。著者は作業計画 において「公式テンプレートへの流し込み」を認識しているが、移行に際しては、SoftwareX特有の要件であるアブストラクトの記述方法や、キーワードの設定（最大6つ）についても、指定されたフォーマットを逸脱してはならない 12。

### **2.2. コードメタデータ（Code Metadata）表の厳密な再構築**

原稿内の「6 Code metadata」セクションに配置された表は 1、視覚的には整っているものの、SoftwareXが義務付けている「C1からC9」までの固定識別子を用いた標準フォーマットに従っていない。ジャーナルの規定では、表の左列は変更不可の定型文であり、右列にのみ情報を記入する形式が求められる 12。

現在の原稿内容をベースに、SoftwareXの規定に完全に準拠したメタデータ表の構成を以下に示す。原稿の表は、例外なくこの構造に置き換えなければならない。

| Nr. | Code metadata description | Metadata |
| :---- | :---- | :---- |
| C1 | Current code version | v0.4.0 1 |
| C2 | Permanent link to code/repository used for this code version | [https://github.com/tatsuki-washimi/gwexpy](https://github.com/tatsuki-washimi/gwexpy) 1 |
| C3 | Permanent link to reproducible capsule | (Zenodo DOIを付与後に正確なURLを記載) 1 |
| C4 | Legal code license | MIT License 1 |
| C5 | Code versioning system used | git 12 |
| C6 | Software code languages, tools and services used | Python 1 |
| C7 | Compilation requirements, operating environments and dependencies | Linux, macOS, Windows. Python \>= 3.11. Dependencies: gwpy \>= 4.0.0, numpy \>= 1.21.0, scipy \>= 1.7.0, astropy \>= 5.0, pandas \>= 1.3.0, matplotlib \>= 3.5.0 1 |
| C8 | If available, link to developer documentation/manual | [https://tatsuki-washimi.github.io/gwexpy/](https://tatsuki-washimi.github.io/gwexpy/) 1 |
| C9 | Support email for questions | tatsuki.washimi@nao.ac.jp 1 |

表内の「C3: Permanent link to reproducible capsule」は、査読者がソフトウェアの動作を担保するための最も重要な項目である。著者はGitHub ReleaseとZenodoを連携してDOIを取得する計画を立てているが 、このアーカイブにはソースコードだけでなく、論文内で提示されたIllustrative examples（Listing 1および2）をワンクリックで実行可能なコンテナ環境（Dockerfile、environment.yml、または Code Ocean カプセル）と、実行に必要なサンプルデータ（合成データまたは匿名化された小規模な実データ）が完全に同梱されていなければならない 12。査読環境が提供されていないソフトウェア論文は、それだけでリジェクトの対象となる 6。

## **3\. ソフトウェア設計と既存エコシステムとの学術的比較**

学術的なソフトウェア論文において、既存のツール群との比較検討を通じた「新規性（Novelty）」と「優位性（Superiority）」の論証は不可欠である。現在の原稿は、GWexpyの利点を主張する一方で、重力波データ解析エコシステムに既に存在する強力なライブラリ群への言及を避けているように見受けられ、これでは査読者からの厳しい追及を免れない 6。

### **3.1. GWpy の内部構造（TimeSeriesDict）と GWexpy.TimeSeriesMatrix の対比**

原稿の「2.2 Architecture」では、多チャンネルデータを扱うための新しい抽象化として TimeSeriesMatrix が紹介されている 1。しかし、GWpyの熟練したユーザーや開発者が査読者となった場合、「GWpyには既に複数のチャンネルを辞書形式で一括管理し、メタデータを保持したままデータを取得・操作できる gwpy.timeseries.TimeSeriesDict クラスが存在するのに、なぜわざわざ新しいMatrixクラスを作成したのか？」という疑念を抱くことは確実である 14。

TimeSeriesDict は、複数の gwpy.timeseries.TimeSeries オブジェクトをチャンネル名をキーとして保持し、read(), fetch(), crop(), resample() などの操作を一括で適用できる強力なコンテナである 14。GWexpyの TimeSeriesMatrix が、学術的に評価されるためには、単なる TimeSeriesDict のラッパーを超えた「行列表現（Matrix representation）に起因する本質的な数学的・プログラム的優位性」を原稿内で明示しなければならない。

以下の表は、論文内で明示すべき GWpy の既存データ構造と GWexpy の新規データ構造の機能的差異の例である。これを基にアーキテクチャの正当性を記述すべきである。

| 比較項目 | GWpy.TimeSeriesDict | GWexpy.TimeSeriesMatrix (要求される機能定義) |
| :---- | :---- | :---- |
| **データ保持形式** | チャンネル名をキーとする辞書 (Dictionary) | 2次元テンソル（行列表現）とメタデータレジストリ |
| **時間軸の扱い** | チャンネルごとに異なる時間軸（t0, dt）を許容 | 行列化の段階で全チャンネルの時間軸の完全な同期を強制 |
| **外部ライブラリへの変換** | チャンネルごとにNumPy配列を抽出し手動で結合 | PyTorchテンソルやObsPy Streamへメタデータを付随して一括変換 |
| **多変量解析の適用** | チャンネルペアごとの反復処理（Forループ）が必要 | クロススペクトル密度行列やBruCoスタイルの計算を内部で一括処理 |

原稿の「2.1 Design principles」において、「Workflow-first abstractions」として行列値の診断（matrix-valued diagnostics）を第一級オブジェクト（first-class objects）として扱うと述べている点は優れているが 1、それが「時間軸が完全にアライメントされた2次元配列に対するベクトル化演算」を可能にし、かつ「AstropyのQuantityに基づく単位（Units）や較正情報を消失させない」という具体的な実装上の利点に直結していることを、2.2節あるいは2.3節で深く論証する必要がある 28。

### **3.2. DTT XML パーサーの新規性と既存ツール群（dttxml, cdsutils, kontrol）との競合分析**

原稿は、LIGO Diaggui DTT XML ファイルをメタデータを保持したまま読み込む機能を大きな特色としている（Section 2.4, 3.1）1。しかし、LIGO/Virgo/KAGRAコミュニティには、DTT XMLを解析するためのツールが既に複数存在している。具体的には、CDS環境で標準的に用いられるPythonライブラリ dttxml 13 や、cdsutils 33、さらにはKAGRAのサスペンション制御用に開発され伝達関数のフィッティング機能等を有する kontrol パッケージ 35 である。

既存の dttxml パッケージは、dtt\_read('fname.xml') メソッドを通じて測定値をネストされた辞書（Bunch型）として返す低レベルなインターフェースを提供しているが 13、これらはGWpyの時系列・周波数系列オブジェクトとは直接統合されていない。著者のGWexpyが、内部のI/Oアダプターとしてこの既存の dttxml パッケージを利用しているのか（あるいはフォークしたのか、独自にゼロからXMLツリーをパースしているのか）についての言及が原稿内に全く存在しないことは、査読において「関連研究の調査不足」と見なされる 6。

原稿の「2.3 Lossless cross-domain interoperability」または「2.4 DTT XML reader note」において、既存のパーサーライブラリ群への適切な参照（引用）を追加した上で、GWexpyの優位性を以下のように論じるべきである。

* 既存の dttxml や ezca 35 は制御系との低レベルなインターフェースには優れているが、抽出されたデータは単なる数値の配列や辞書であり、データの単位（例：V/count）や測定時刻、チャンネルの方向性といった物理的メタデータがGWpyの解析パイプラインにシームレスに引き継がれない。  
* GWexpyは、TimeSeries.read(..., format="dttxml", products=\["tf"\]) というGWpyの標準的な高レベルAPIを踏襲することで、DTT XMLの複雑なネスト構造を隠蔽しつつ、伝達関数（TF）やコヒーレンスといった測定プロダクトを、ただちにプロットやモデリングが可能な gwpy.frequencyseries.FrequencySeries 等のオブジェクトとしてインスタンス化する 1。この「パースからオブジェクト化までのメタデータ保存パイプライン」こそが、コミッショニング現場でのアドホックなスクリプト記述を駆逐する革新性である。

### **3.4. BruCo および非定常ノイズ解析（HHT/EMD）ワークフローの実装詳細**

原稿のアーキテクチャ図（Figure 1）およびセクション2.2において、分析ツールとして「HHT/EMD 」や「BruCo に着想を得た自動ノイズカップリングワークフロー」が含まれていると主張している 1。

BruCo（Brute Force Coherence）は、重力波干渉計のひずみチャンネル ![][image1] と、無数の環境モニター・補助チャンネル（地震計、磁力計、マイクなど）との間の線形コヒーレンスを網羅的に計算し、特定の周波数帯域におけるノイズ結合（Noise coupling）を特定するための事実上の標準ツールである 40。しかし、原稿内でのBruCoへの言及は表面的な「inspired by」に留まっており、GWexpyが実際にどのようにこの機能を拡張・統合したのかが不明瞭である。

著者は、単に「コヒーレンスを計算できる」という事実を述べるのではなく、Listing 2で示された matrix.coherence(target="IFO\_frame\_channel", band=(10, 100)) メソッドが、内部的にどのような処理を行っているかを記述すべきである。例えば、「多様なサンプリングレートを持つローカルファイル（WIN, WAV）とフレームデータを TimeSeriesMatrix 内で同一のサンプリング周波数に揃え（アンチエイリアシングフィルタを伴うリサンプリング）、Welch法 45 を用いてクロススペクトル密度行列を構築し、全てのペアにおけるコヒーレンス ![][image2] を並列計算した上で、ターゲットチャンネルに対する上位結合チャンネルを抽出する。この全プロセスにおいて、各センサーの物理単位とGPS時刻のメタデータは完全に保護される」といった具体的な処理機構の記述が必要である。これにより、読者はGWexpyが単なる関数群ではなく、真の意味での「Workflow-level abstraction」であることを理解する。

同様に、HHT（Hilbert-Huang Transform）やEMD（Empirical Mode Decomposition）46 についても、それらが非定常なトランジェントノイズ（グリッチ）の解析においてFourier変換（FFT）やQ変換（Q-transform）47 をどのように補完するのか、そしてそれがメタデータ付きオブジェクトとしてどのように返されるのかを最低限1段落を用いて説明すべきである。

## **4\. 原稿セクション別の辛口評価と具体的な改稿指示**

提出された LaTeX 原稿 1 の各セクションに対し、さらなる改善点と構成の最適化を指摘する。

### **4.1. Title & Abstract (タイトルとアブストラクト)**

**批判:**

タイトルは適切であるが、アブストラクトの記述が定性的すぎる。「lossless handling of rich metadata」という表現は抽象的であり、ソフトウェアの具体的な入力と出力をイメージしにくい。SoftwareXの読者は、実装されている技術要素を即座に把握することを望む。

**改善指示:** GWexpyがサポートする具体的なファイルフォーマット（LIGO DTT XML, WIN, ATS, HDF5, WAV 等）と、出力される互換オブジェクト（PyTorch Tensor, ObsPy Stream 等）の名称をアブストラクトに明記する 1。また、「数行のコードで多変量コヒーレンスや伝達関数解析を実現し、再現可能なコミッショニングパイプラインを構築する」といった、運用的メリットを明確に打ち出すべきである。

### **4.2. 1 Motivation and significance (動機と意義)**

**批判:** コミッショニングにおける「メタデータ喪失の痛点（Pain point）」の記述は優れている。しかし、重力波干渉計の最前線の状況（LIGOのA+アップグレードやKAGRAのO4/O5に向けた複雑なノイズハンティング、低温サスペンションの制御など）2 への言及が不足しているため、なぜ今このソフトウェアが必要なのかという切迫感が薄い。また、既存のGWpyが天体物理学的な推論（Astrophysical inference）やGWOSCの公開データ処理に強く最適化されており 48、現場の未加工・非標準フォーマットの処理において生じるギャップを埋める存在としてGWexpyがある、という論理構成が明確でない。

**改善指示:** 序盤で重力波観測におけるディテクターキャラクターリゼーション（DetChar）とノイズ低減の重要性を強調し、既存のGWpyが持つ「単一チャンネル・標準フレームデータ中心」のアーキテクチャの限界を指摘する 48。その上で、実験室でのローカルな取得データ（WINフォーマットなど）と干渉計の制御データ（DTT XMLなど）を統合する際の「メタデータ剥離（CSVやNumPy配列へのキャストによる単位・時刻の消失）」の深刻な悪影響を論証する。

### **4.3. 2 Software description & Architecture (ソフトウェアの説明とアーキテクチャ)**

**批判:** Figure 1 のアーキテクチャ図（TikZで描画された簡素なブロック図）1 は、ソフトウェア論文の図としてはあまりにも情報量が少ない。内部のデータ構造、メソッドの依存関係、メタデータの流れが視覚的に伝わらない。また、「2.4 DTT XML reader note」は、APIリファレンスの注意書きのレベルであり、論文の本文（特にアーキテクチャを語る章）に独立したサブセクションとして存在するのは非常に不自然である 1。

**改善指示:**

* **図の高度化:** 単なるブロック図ではなく、UMLのクラス図または詳細なデータフロー図に差し替える。「ローカルファイル ![][image3] I/O Adapters ![][image3] TimeSeriesMatrix (メタデータレジストリとNumPyテンソルを内包) ![][image3] 分析メソッド群 ![][image3] Visualization」というデータの流れを、メタデータがどのように随伴していくかの注釈とともに視覚化する。  
* **セクション2.4の統合:** DTT XMLの読み込み時に products=\["tf"\] などを明示的に指定させる設計思想（Ambiguous defaultsの排除とプロビナンスの確保）は、セクション2.3の「メタデータ保存戦略」の一部として統合し、より上位の設計哲学として語るべきである。  
* **インターフェースの解説:** 外部ライブラリへの変換（obspy.Stream や torch.Tensor）1 において、具体的にどのようなメタデータマッピングが行われるのか（例：GWpyの channel.name が ObsPyの network.station.location.channel にどうマッピングされるか、PyTorchへの変換時にテンソルの attributes としてメタデータがどうシリアライズされるか）を最低限の技術的詳細をもって記述する 1。

### **4.4. 3 Illustrative examples (実例)**

**批判:** 提示されている Listing 1（DTT XMLからの伝達関数算出）と Listing 2（異種フォーマットの融合とコヒーレンスランク付け）1 は、ユースケースとしては非常に適切である。しかし、コードスニペットの末尾が単なる print() 関数で終わっており 1、読者（査読者）はメタデータが本当に保存されているのか、出力結果がどの程度リッチなオブジェクトであるのかを視覚的に確認できない。SoftwareXでは、コードの効果を示すために出力のプロットや生成される解析結果の図表を併載することが強く推奨されている 12。

**改善指示:** Listing 1 または 2 のいずれか（あるいは両方）の出力結果として、メタデータ（単位、軸ラベル、チャンネル名）が適切にフォーマットされた状態の「Bode線図（伝達関数）」または「コヒーレンススペクトル図」を、Figure 2等として論文内に追加する。これにより、「メタデータが保存されているため、プロット関数を呼ぶだけで完全な軸情報を持った図が生成される」というGWexpyの最大の利点を視覚的に証明できる。 また、著者は作業計画 で「コード例とAPIの完全な一致」を意図しているが、これは必須要件である。査読者は提供される再現性カプセル（Jupyter notebook）を実行し、論文のListingと一言一句違わず動作し、同じ図が出力されることを確認する 6。擬似コード（Pseudocode）は一切許容されない。

### **4.5. 4 Impact (影響)**

**批判:** 記述が抽象的であり、インパクトが定量化されていない 1。

**改善指示:** 運用的インパクト（Operational impact）について、従来のワークフローとの明確な比較を行う。例えば、「DTT XMLから伝達関数を抽出し、別の解析パイプラインに渡す作業は、従来数十行のカスタムパーススクリプトを必要としたが、GWexpyによりこれが3行のAPIコールに短縮され、単位変換エラーによるトラブルシューティングの時間が劇的に削減される」といった具体的な利点を記述する 54。 科学的・ドメイン横断的インパクト（Scientific and cross-domain impact）については、地震学（ObsPy）48 や、PyTorchを用いた重力波信号の機械学習（SDL: Signal Deep Learning や ml4gw など）47 において、メタデータを保持したバッチデータ生成がいかに重要であるかを補足する。

## **5\. ソフトウェア工学基準に基づく実装と品質管理の要求**

SoftwareXの査読基準において、ソフトウェアの品質管理（Quality Assurance）と自動化されたテストの存在は、採択を決定づける極めて重要な要素である 6。著者の作業 ToDo を検討した結果、以下の点でソフトウェアエンジニアリングのベストプラクティスが不足している。

### **5.1. 自動テストスイート（CI/CD）の必須化**

著者は「Executable Examples」としてJupyter notebookのCIによる自動チェックを計画しているが 、これだけでは不十分である。ソフトウェア論文として受理されるためには、コードのコアロジックを検証するユニットテスト（Unit tests）のフレームワーク（例えば pytest）が構築されていることが必須である 6。

* **要求される対応:** GitHub Actions等のCIパイプラインを構築し、Linux/macOS/Windowsの複数環境、およびサポートするPythonバージョン（\>= 3.11）においてテストが自動実行され、バッジ（Passing）が表示される状態にする。  
* テストカバレッジ（Test coverage）の測定（例：pytest-cov）を導入し、特に「メタデータが変換や計算の前後で消失していないこと」をアサート（assert）するテストケースを充実させること。

### **5.2. 依存関係（Dependencies）の分離とパッケージング**

作業計画 において、著者が pyproject.toml の依存関係を整理し、コアパッケージとオプショナルな拡張機能（GUIやPyTorch、ObsPy連携など）を明確に分離する方針を立てている点は高く評価できる。重力波解析の現場において、巨大な依存関係ツリーは環境構築（特に制御系やオフライン解析サーバー上）の妨げとなるためである 38。 この方針は、前述の「Code metadata」表の「C7」欄にも反映させ、「Core dependencies」と「Optional dependencies」を明確に分けて記載することで、ソフトウェアの設計思想としてアピールすべきである 12。

### **5.3. APIドキュメントの完全性**

査読者は、提供されたドキュメントのリンク（https://tatsuki-washimi.github.io/gwexpy/）にアクセスし、APIが「適切に文書化されているか（documented to a suitable level）」を確認する 6。SphinxやMkDocs等のツールを用いて、原稿内で言及されている全ての主要クラス（TimeSeriesMatrix, SpectrogramMatrix など）およびメソッドの引数、戻り値、使用例が記載されたドキュメントを公開前に完全に整備しなければならない。ドキュメントの欠如は、即座に「Revision」または「Reject」の理由となる。

## **6\. 著者の「原稿以外の作業ToDo」に対する批判的検討と発展的提案**

著者が作成した 原稿以外の作業ToDo.md は、SoftwareXへの投稿に向けた必須タスクを的確に捉えており、その自己分析能力は評価に値する。本査読レポートの観点から、このToDoリストに対するフィードバックと発展的提案を以下にまとめる。

1. **Template Migration:**  
   公式テンプレートへの移行は最優先事項である。移行時には、単なるテキストのコピーに留まらず、本レポートで指摘した「アーキテクチャの論証強化」と「既存ツール（dttxml, BruCo, kontrol 等）への参照追加」を同時に行うこと。  
2. **Zenodo DOI Acquisition:**  
   完璧なアプローチである。リリースには必ず、論文のIllustrative examplesを完全に再現するためのダミーデータ（合成データ）と、環境構築手順（requirements.txt）を同梱すること。  
3. **Repository Alignment (GUI機能の除外):**  
   初回リリースからGUI機能を実験的扱いとして除外する判断は極めて賢明である。ソフトウェア論文は焦点を絞るべきであり、「メタデータ保存型パイプライン」に特化したコア機能の堅牢性を主張することに注力すべきである。  
4. **DTT XML Reader Specifications:**  
   products 引数の必須化やエラーメッセージの改善は、ユーザーエクスペリエンスの向上に直結する。この仕様決定の背景にある「曖昧なデフォルト値の排除と測定意図の保存」という哲学を、原稿の本文（Section 2.3 または 2.4）で積極的にアピールすること。  
5. **Reference Formalization:** BruCoやObsPy、PyTorchなどの外部ライブラリへの適切な学術的引用（Citation）を整備する計画は不可欠である。特にBruCoについては、GitHubリポジトリだけでなく、関連する物理学の論文（例：Vajente et al., 2022 1）への参照を確実に組み込むこと。

## **7\. 結論と採択に向けた最終戦略**

GWexpy は、重力波干渉計のコミッショニングという高度に専門的で複雑な領域において、メタデータの散逸を防ぎ、データ駆動型のノイズ診断（BruCo、伝達関数解析、HHT等）を効率化する極めて実用的なソフトウェアである。その基本理念は、学術ソフトウェアの再利用性と科学的堅牢性を追求するSoftwareXの出版理念と完全に合致している。

しかしながら、本パッケージが『SoftwareX』にOriginal Software Publicationとして採択されるためには、現在の原稿が抱える「マニュアル的記述」から脱却し、厳密な「学術的ソフトウェア論文」へと変貌を遂げる必要がある。具体的には、以下の3点を至急実行することが求められる。

1. **フォーマットの完全な順守:** elsarticle テンプレートの適用と、C1からC9までの要件を厳格に満たすCode Metadata表の構築。  
2. **アーキテクチャの学術的論証:** GWpyの既存構造（TimeSeriesDict）との明確な差異の提示、および既存のパースライブラリ（dttxml, cdsutils 等）との差別化。  
3. **証拠としての視覚化と再現性:** コード例の出力結果（メタデータ付きのプロット図）の追加と、ユニットテスト・CI/CD環境を完備したZenodo Reproducible Capsuleの提供。

著者の事前の作業計画 は的を射ており、本報告書で提示した修正要件を統合的に適用することで、原稿は査読者の厳しい要求を満たす水準に到達するであろう。重力波データ解析コミュニティ、さらには関連する精密測定・制御工学分野に対しても高い波及効果をもたらすソフトウェアとして、本論文が完成することを強く期待する。迅速かつ徹底的な再構成を推奨する。

#### **引用文献**

1. main.tex  
2. PoS(ICRC2023)1564, 3月 14, 2026にアクセス、 [https://pos.sissa.it/444/1564/pdf](https://pos.sissa.it/444/1564/pdf)  
3. Identification of Noise-Associated Glitches in KAGRA O3GK with Hierarchical Veto | Progress of Theoretical and Experimental Physics | Oxford Academic, 3月 14, 2026にアクセス、 [https://academic.oup.com/ptep/article/2025/8/083F01/8196581](https://academic.oup.com/ptep/article/2025/8/083F01/8196581)  
4. Review of the Advanced LIGO Gravitational Wave Observatories Leading to Observing Run Four \- MDPI, 3月 14, 2026にアクセス、 [https://www.mdpi.com/2075-4434/10/1/36](https://www.mdpi.com/2075-4434/10/1/36)  
5. GW \- Detector Characterization Workshop \- IUCAA, 3月 14, 2026にアクセス、 [https://web.iucaa.in/ws/\~GWDCW@2025/index.html](https://web.iucaa.in/ws/~GWDCW@2025/index.html)  
6. How to review a software-tool paper? \- Academia Stack Exchange, 3月 14, 2026にアクセス、 [https://academia.stackexchange.com/questions/99929/how-to-review-a-software-tool-paper](https://academia.stackexchange.com/questions/99929/how-to-review-a-software-tool-paper)  
7. How to submit a manuscript Common Rejection Reasons | Publish your research | Springer Nature, 3月 14, 2026にアクセス、 [https://www.springernature.com/gp/authors/campaigns/how-to-submit-a-journal-article-manuscript/common-rejection-reasons](https://www.springernature.com/gp/authors/campaigns/how-to-submit-a-journal-article-manuscript/common-rejection-reasons)  
8. SOFTWAREX \- ResearchGate, 3月 14, 2026にアクセス、 [https://www.researchgate.net/file.PostFileLoader.html?id=55a0ddef5cd9e367578b45cd\&assetKey=AS%3A273810664689664%401442292957839](https://www.researchgate.net/file.PostFileLoader.html?id=55a0ddef5cd9e367578b45cd&assetKey=AS:273810664689664@1442292957839)  
9. Gravitational waves discovery shows why software should be every scientist's business, 3月 14, 2026にアクセス、 [https://www.elsevier.com/connect/gravitational-waves-discovery-shows-why-software-should-be-every-scientists](https://www.elsevier.com/connect/gravitational-waves-discovery-shows-why-software-should-be-every-scientists)  
10. Promoting your codes and software \- DATACC, 3月 14, 2026にアクセス、 [https://www.datacc.org/en/best-practices/open-your-codes-and-software/promoting-your-codes-and-software/](https://www.datacc.org/en/best-practices/open-your-codes-and-software/promoting-your-codes-and-software/)  
11. litstudy: A Python package for literature reviews \- UvA-DARE (Digital Academic Repository), 3月 14, 2026にアクセス、 [https://pure.uva.nl/ws/files/134862348/litstudy.pdf](https://pure.uva.nl/ws/files/134862348/litstudy.pdf)  
12. template \- Elsevier, 3月 14, 2026にアクセス、 [https://legacyfileshare.elsevier.com/promis\_misc/softwarex-osp-template.docx](https://legacyfileshare.elsevier.com/promis_misc/softwarex-osp-template.docx)  
13. CDS / software / dttxml \- LIGO GitLab, 3月 14, 2026にアクセス、 [https://git.ligo.org/cds/dttxml](https://git.ligo.org/cds/dttxml)  
14. TimeSeriesDict — GWpy 1.0.0 documentation, 3月 14, 2026にアクセス、 [https://gwpy.github.io/docs/1.0.0/api/gwpy.timeseries.TimeSeriesDict.html](https://gwpy.github.io/docs/1.0.0/api/gwpy.timeseries.TimeSeriesDict.html)  
15. (PDF) Why your manuscript was rejected and how to prevent it? \- ResearchGate, 3月 14, 2026にアクセス、 [https://www.researchgate.net/publication/50364302\_Why\_your\_manuscript\_was\_rejected\_and\_how\_to\_prevent\_it](https://www.researchgate.net/publication/50364302_Why_your_manuscript_was_rejected_and_how_to_prevent_it)  
16. Reasons for Peer Review Rejection – and how to avoid it \- Charlesworth Author Services, 3月 14, 2026にアクセス、 [https://www.cwauthors.com/article/reasons-for-peer-review-rejection-and-how-to-avoid-peer-review-rejection](https://www.cwauthors.com/article/reasons-for-peer-review-rejection-and-how-to-avoid-peer-review-rejection)  
17. Top 10 reasons your manuscript may be rejected without review \- PMC, 3月 14, 2026にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC9650869/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9650869/)  
18. Title (Name of your software: Then a short title) \- Elsevier, 3月 14, 2026にアクセス、 [https://legacyfileshare.elsevier.com/promis\_misc/softwarex-osp-template.tex](https://legacyfileshare.elsevier.com/promis_misc/softwarex-osp-template.tex)  
19. LaTeX instructions for authors \- Elsevier, 3月 14, 2026にアクセス、 [https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions)  
20. Version \[number\]- \[title of your first SoftwareX publication\] \- Elsevier, 3月 14, 2026にアクセス、 [https://www.elsevier.com/\_\_data/promis\_misc/Updated\_software-update-template.tex](https://www.elsevier.com/__data/promis_misc/Updated_software-update-template.tex)  
21. Elsevier Article (elsarticle) Template \- Overleaf, Online LaTeX Editor, 3月 14, 2026にアクセス、 [https://www.overleaf.com/latex/templates/elsevier-article-elsarticle-template/vdzfjgjbckgz](https://www.overleaf.com/latex/templates/elsevier-article-elsarticle-template/vdzfjgjbckgz)  
22. Elsevier template \- elsarticle \- LaTeX Stack Exchange, 3月 14, 2026にアクセス、 [https://tex.stackexchange.com/questions/407613/elsevier-template](https://tex.stackexchange.com/questions/407613/elsevier-template)  
23. Updated\_software-update-template.docx \- Elsevier, 3月 14, 2026にアクセス、 [https://www.elsevier.com/\_\_data/promis\_misc/Updated\_software-update-template.docx](https://www.elsevier.com/__data/promis_misc/Updated_software-update-template.docx)  
24. Version \[number\]- \[title of your first SoftwareX publication\] \- Elsevier, 3月 14, 2026にアクセス、 [https://legacyfileshare.elsevier.com/promis\_misc/softwarex-software-update-template.tex](https://legacyfileshare.elsevier.com/promis_misc/softwarex-software-update-template.tex)  
25. SoftwareX Niimpy: A toolbox for behavioral data analysis \- ResearchGate, 3月 14, 2026にアクセス、 [https://www.researchgate.net/profile/Talayeh-Aledavood/publication/372601133\_Niimpy\_A\_toolbox\_for\_behavioral\_data\_analysis/links/64c79651a6de650dca8115e4/Niimpy-A-toolbox-for-behavioral-data-analysis.pdf?origin=scientificContributions](https://www.researchgate.net/profile/Talayeh-Aledavood/publication/372601133_Niimpy_A_toolbox_for_behavioral_data_analysis/links/64c79651a6de650dca8115e4/Niimpy-A-toolbox-for-behavioral-data-analysis.pdf?origin=scientificContributions)  
26. Mistakes Reviewers Make \- Niklas Elmqvist \- Medium, 3月 14, 2026にアクセス、 [https://niklaselmqvist.medium.com/mistakes-reviewers-make-ce3a4c595aa2?source=follow\_footer-----a8550a976ea0----4----------------------------](https://niklaselmqvist.medium.com/mistakes-reviewers-make-ce3a4c595aa2?source=follow_footer-----a8550a976ea0----4----------------------------)  
27. TimeSeriesDict — GWpy 2.0.4 documentation, 3月 14, 2026にアクセス、 [https://gwpy.github.io/docs/2.0.4/api/gwpy.timeseries.TimeSeriesDict.html](https://gwpy.github.io/docs/2.0.4/api/gwpy.timeseries.TimeSeriesDict.html)  
28. TimeSeries — GWpy 2.1.1 documentation, 3月 14, 2026にアクセス、 [https://gwpy.github.io/docs/2.1.1/api/gwpy.timeseries.TimeSeries/](https://gwpy.github.io/docs/2.1.1/api/gwpy.timeseries.TimeSeries/)  
29. TimeSeries — GWpy 1.0.0 documentation, 3月 14, 2026にアクセス、 [https://gwpy.github.io/docs/1.0.0/api/gwpy.timeseries.TimeSeries.html](https://gwpy.github.io/docs/1.0.0/api/gwpy.timeseries.TimeSeries.html)  
30. software \- LIGO GitLab, 3月 14, 2026にアクセス、 [https://git.ligo.org/cds/software](https://git.ligo.org/cds/software)  
31. 3月 14, 2026にアクセス、 [https://raw.githubusercontent.com/regro/cf-graph-countyfair/master/mappings/pypi/grayskull\_pypi\_mapping.yaml](https://raw.githubusercontent.com/regro/cf-graph-countyfair/master/mappings/pypi/grayskull_pypi_mapping.yaml)  
32. 3月 14, 2026にアクセス、 [https://raw.githubusercontent.com/regro/cf-graph-countyfair/master/mappings/pypi/name\_mapping.yaml](https://raw.githubusercontent.com/regro/cf-graph-countyfair/master/mappings/pypi/name_mapping.yaml)  
33. Development \- LIGO GitLab, 3月 14, 2026にアクセス、 [https://git.ligo.org/cds/cdsutils/-/boards](https://git.ligo.org/cds/cdsutils/-/boards)  
34. conda-forge \- Anaconda.org, 3月 14, 2026にアクセス、 [https://conda-static.anaconda.org/conda-forge](https://conda-static.anaconda.org/conda-forge)  
35. terrencetec/kontrol: KAGRA control python package \- GitHub, 3月 14, 2026にアクセス、 [https://github.com/terrencetec/kontrol](https://github.com/terrencetec/kontrol)  
36. How to use Kontrol \- Python Kontrol Library \- Read the Docs, 3月 14, 2026にアクセス、 [https://kontrol.readthedocs.io/en/master/how\_to\_use\_kontrol.html](https://kontrol.readthedocs.io/en/master/how_to_use_kontrol.html)  
37. Guidelines for performing Systematic Literature Reviews in Software Engineering, 3月 14, 2026にアクセス、 [https://www.researchgate.net/publication/302924724\_Guidelines\_for\_performing\_Systematic\_Literature\_Reviews\_in\_Software\_Engineering](https://www.researchgate.net/publication/302924724_Guidelines_for_performing_Systematic_Literature_Reviews_in_Software_Engineering)  
38. Installation — Advanced LIGO Guardian 1.5.3 documentation, 3月 14, 2026にアクセス、 [https://cds.docs.ligo.org/software/guardian/install.html](https://cds.docs.ligo.org/software/guardian/install.html)  
39. Advanced LIGO Guardian Documentation, 3月 14, 2026にアクセス、 [https://dcc.ligo.org/public/0120/T1500292/001/AdvancedLIGOGuardian.pdf](https://dcc.ligo.org/public/0120/T1500292/001/AdvancedLIGOGuardian.pdf)  
40. Coherence DeepClean: Toward autonomous denoising of gravitational-wave detector data \- arXiv.org, 3月 14, 2026にアクセス、 [https://arxiv.org/pdf/2501.04883](https://arxiv.org/pdf/2501.04883)  
41. Virgo detector characterization and data quality: tools \- UvA-DARE (Digital Academic Repository), 3月 14, 2026にアクセス、 [https://pure.uva.nl/ws/files/164057239/2210.15634.pdf](https://pure.uva.nl/ws/files/164057239/2210.15634.pdf)  
42. Non-Linear Noise Subtraction for Low Frequency \- DCC, 3月 14, 2026にアクセス、 [https://dcc.ligo.org/public/0178/T2100342/001/SURF\_Final\_Report%20%285%29.pdf](https://dcc.ligo.org/public/0178/T2100342/001/SURF_Final_Report%20%285%29.pdf)  
43. Virgo detector characterization and data quality: tools \- DSpace, 3月 14, 2026にアクセス、 [https://dspace.library.uu.nl/bitstream/handle/1874/432666/F\_Acernese\_2023\_Class.\_Quantum\_Grav.\_40\_185005.pdf?sequence=1](https://dspace.library.uu.nl/bitstream/handle/1874/432666/F_Acernese_2023_Class._Quantum_Grav._40_185005.pdf?sequence=1)  
44. Virgo detector characterization and data quality: tools, 3月 14, 2026にアクセス、 [https://openaccess.inaf.it/bitstreams/6ed649d0-da62-4fe1-9f31-73dd3f22b074/download](https://openaccess.inaf.it/bitstreams/6ed649d0-da62-4fe1-9f31-73dd3f22b074/download)  
45. Detector Characterization and Mitigation of Noise in Ground-Based Gravitational-Wave Interferometers \- MDPI, 3月 14, 2026にアクセス、 [https://www.mdpi.com/2075-4434/10/1/12](https://www.mdpi.com/2075-4434/10/1/12)  
46. SoftwareX template \- For Authors \- SciSpace, 3月 14, 2026にアクセス、 [https://scispace.com/formats/elsevier/softwarex/135cffed0cc6bcb92c6abcd4dadde517](https://scispace.com/formats/elsevier/softwarex/135cffed0cc6bcb92c6abcd4dadde517)  
47. (PDF) ml4gw: PyTorch utilities for training neural networks in gravitational wave physics applications \- ResearchGate, 3月 14, 2026にアクセス、 [https://www.researchgate.net/publication/396760222\_ml4gw\_PyTorch\_utilities\_for\_training\_neural\_networks\_in\_gravitational\_wave\_physics\_applications](https://www.researchgate.net/publication/396760222_ml4gw_PyTorch_utilities_for_training_neural_networks_in_gravitational_wave_physics_applications)  
48. clawdia: A dictionary learning framework for gravitational-wave data analysis \- arXiv.org, 3月 14, 2026にアクセス、 [https://arxiv.org/html/2511.16750v1](https://arxiv.org/html/2511.16750v1)  
49. School of Physics and Astronomy \- \-ORCA \- Cardiff University, 3月 14, 2026にアクセス、 [https://orca.cardiff.ac.uk/id/eprint/149256/1/2021SklirisVPhD.pdf](https://orca.cardiff.ac.uk/id/eprint/149256/1/2021SklirisVPhD.pdf)  
50. Constraints on gravitational waves from the 2024 Vela pulsar glitch \- arXiv, 3月 14, 2026にアクセス、 [https://arxiv.org/html/2512.17990v1](https://arxiv.org/html/2512.17990v1)  
51. Open Data from LIGO, Virgo, and KAGRA through the First Part of the Fourth Observing Run, 3月 14, 2026にアクセス、 [https://arxiv.org/html/2508.18079v1](https://arxiv.org/html/2508.18079v1)  
52. SoftwareX PPGISr: An R package for Public Participatory GIS \- GLISA, 3月 14, 2026にアクセス、 [https://glisa.umich.edu/wp-content/uploads/2023/05/VanBerkel\_2023\_PPGISr.pdf](https://glisa.umich.edu/wp-content/uploads/2023/05/VanBerkel_2023_PPGISr.pdf)  
53. SoftwareX Introduction of the BiasAdjustCXX command-line tool for the application of fast and efficient bias corrections in clim \- EPIC, 3月 14, 2026にアクセス、 [https://epic.awi.de/59754/1/Schwertfeger\_et\_al\_2023.pdf](https://epic.awi.de/59754/1/Schwertfeger_et_al_2023.pdf)  
54. Digital tools for Reuse \- Guide Bâtiment Durable, 3月 14, 2026にアクセス、 [https://guidebatimentdurable.brussels/sites/default/files/documents/2024-05/fcrbe\_digital-tools-for-reuse\_final-version\_compressed.pdf](https://guidebatimentdurable.brussels/sites/default/files/documents/2024-05/fcrbe_digital-tools-for-reuse_final-version_compressed.pdf)  
55. Assessment methodologies for the combined seismic and energy retrofit of existing buildings \- JRC Publications Repository, 3月 14, 2026にアクセス、 [https://publications.jrc.ec.europa.eu/repository/bitstream/JRC131412/JRC131412\_01.pdf](https://publications.jrc.ec.europa.eu/repository/bitstream/JRC131412/JRC131412_01.pdf)  
56. The Channel — GWpy 1.0.0 documentation, 3月 14, 2026にアクセス、 [https://gwpy.github.io/docs/1.0.0/detector/channel.html](https://gwpy.github.io/docs/1.0.0/detector/channel.html)  
57. SoftwareX Review Form, 3月 14, 2026にアクセス、 [https://legacyfileshare.elsevier.com/promis\_misc/softwarex-reviewer-form.pdf](https://legacyfileshare.elsevier.com/promis_misc/softwarex-reviewer-form.pdf)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACEAAAAZCAYAAAC/zUevAAACGklEQVR4Xu2VPyjFURTHj1CEUPKnlD+bMCmTTAwGDBalbGIwMViVzAwGg5LBQBZJDIYXC5ktSiFlohSTwvfbfee9+zu/+5PnLdL71Lfe75x777v3nHPPFSlQ4J9SDZVaY4AyawhxAd1Dr9CO8YUYgy6heuv4hmuo1xp9RqF96BOaM74QN9CgNXp0WwNYgk7ERS9IEbQNvUN9xmfh2GOo3DrStEEPUEXAzmjPGHuGOuhK3AkbjM/SBY1bo8eIuIhauPk16FwSojEt0VQw18VZdwYW2KHET6nQzpA/WkeadnG+Zesgm+JSMQWdQs/QCzTkDwJN0J2xKTyE1XBkhEgVdCbuIDGYCk5iqFqhWnHVT/G30gO9ed8+3OCkuHVWxKXVXt8SaE/c/8XgwqyH1vR3JZQSd2ourjAyoXwrDDP9rIsktiQhmpy44H1r7lLiNqQwvEmb0I1zHucnwU08WSPDbW8Fq/gDGvBs5LtNaJSCRefBTfAKR2CemSfmS7mVbD0wHXpT2ENYwCFsKtgAW7LuDMF0TEg0FcQ/0axn75BAKNPwatLHMeRI4m1dr3DK2GVD4l1SrxdPcuDZtamx8Vi4uBYym1GoOHU+/zNCjTWIa0pcLPT6cXG7aYXFmTSPsCmyHtjC84IhXrTGH8C3hm8Oiz4UyZzhQ5Trafjqcl6ndfyWXWhVcjtRCpq3xnxphNahZusI0C/xFv53+AKh6Wnd3wQdegAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAZCAYAAABOxhwiAAACkUlEQVR4Xu2WS6iNURTH/0J5XK+uusmVSEqUJEauDAwYM1BiRIgRAyPySBnqphQiA0wUJa8yuKLIzMDEI4/IyERRBh7//1nf7uyzzj5n7/OFrrq/+nXPWWvf7+xv7b3X9wFjjPHXmEMP0bN0usuNWlbQc3QeXUyf0cGWEaOUYfoIVnXxnR5sptuYSGf4YCGTaJ8P1mU8nRJ9/0qPRN9jNOHdPphA467SB/QVnRDlVlf+UcbRF3S+T8Byx+lkn3Bo3Cn6lu6lP9BczZC/j/qrlmQTXeSDFUvpRx9MsIB+oNfoHvoLrRUX71G2ckVo+a77YIWqdJo+8YkEG2GTPeETEeFaHauuH+ynM30iQttCS7eDbqYX6L6WEcB62JJrUjku0290lU9ELKSf0OHmNOl3sLuXF13ucPX5PJpjgn6C6jJf6HIXj1kHuzl/rRvRmMA0+pDegnWaFrbBLqR9GS6oCQvt4zvV5xIuwYoQHzKP2uQA7AYfw35D41PtT3teZ+A5nR0ndJDewC4m1AmOwg6OULVLD8dU2FZShVSpHD9hq5gjWQy1Iz8x3cQWWFV0t/pbgio2Upmqnkcru8sHE2jin+kSn0hxj96ky3yiC71MXHkdOh2+HJq42mbR68VregXtfbUbYT/qf3OrpEmUbqnkVunES7rBBwvQ+0xJddbAJpQjnJsR5FexwW3kq5ZCezbXm8VWut8HE6iTqKOUHOIGqlwdwj7XlulEqOJcn0igQmgFQ5frin5cFamLupTeMTw7K3UgNZnwrOjGXdhjv2Rso+2s9MEe0KvDUx+EtT8dsu2wh08JKoCeNUXoMT7LB3tkCO0/qKfySdg7yhmXS6EXqwM++C/Qq4Tes+uwlh7zwf+e3/3td5KH4/QBAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAAAgklEQVR4XmNgGAWjYAQAWSDuBmIOdAlyAD8QbwZiTXQJckE5FFMFiAHxfiA2Q5cgF4AMOgLEKugSPEAsSQYOBuJHQMzJQCHgBuKFQNyHLkEqYAHiqUBcBsSMaHIkA1cgXs1ABe+BXAXynge6BDlAmgGSaEXQJcgBrEAsxECFsBqiAAAGOwxsFgKSAwAAAABJRU5ErkJggg==>