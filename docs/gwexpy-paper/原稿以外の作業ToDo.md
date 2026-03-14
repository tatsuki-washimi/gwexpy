# 原稿以外の作業 ToDo

## 最優先

* [ ] **SoftwareX の公式テンプレートへ移行**

  * 現在の独自 `article` ベースではなく、投稿用テンプレートに合わせる
  * タイトル、Abstract、Code metadata、References の体裁をテンプレート準拠にする

* [ ] **Zenodo DOI を取得**

  * GitHub Release を作成
  * Zenodo と連携してアーカイブ DOI を発行
  * 発行後、論文中の “Permanent link to reproducible capsule” を DOI に更新

* [ ] **リポジトリの公開状態を論文と一致させる**

  * 論文で述べる機能だけを初回公開版に含める
  * GUI 関連は初回リリースから外す
  * README, docs, package extras も GUI 除外方針に合わせて整理する

## 実装・パッケージ整理

* [ ] **GUI 機能を初回リリース対象から除外**

  * `pyproject.toml` の extras や依存関係を見直す
  * GUI に関するコード・依存・説明を release 対象から切り離す
  * experimental 扱いにするなら明示する

* [ ] **依存関係を整理**

  * core dependencies と optional extras を明確に分離
  * README と `pyproject.toml` の記述を一致させる
  * 不要に重い依存を core に入れない

* [ ] **API 名と論文中のコード例の整合を確認**

  * 論文に載せるメソッド名・引数名が実装と完全一致しているか確認
  * 特に `TimeSeries.read(...)`, `TimeSeriesMatrix.*`, DTT XML reader の呼び出しを点検
  * “illustrative” な疑似 API を残さない

* [ ] **DTT XML reader の仕様を明確化**

  * `products` 引数必須なら、コードとドキュメントで一貫して明示
  * エラーメッセージを分かりやすくする
  * public API 経由で使う想定に統一する

## 再現性・検証

* [ ] **サンプルコードを実行可能にする**

  * 論文掲載例をそのまま走る最小例として整備
  * examples/notebooks との対応を取る
  * 疑似コードではなく実コードにする

* [ ] **CI で examples / notebooks を検証**

  * Jupyter notebooks の実行確認を自動化
  * 論文に載せた Listing が壊れていないことを継続的に確認
  * 失敗時にすぐ気づけるようにする

* [ ] **再現用データを整理**

  * synthetic または anonymized data を examples 用に用意
  * notebooks が外部環境依存になりすぎないようにする
  * 実行に必要な最小データセットを明示する

## ドキュメント・公開情報

* [ ] **README を論文と整合させる**

  * 機能一覧、依存関係、対応フォーマット、除外機能を一致させる
  * インストール方法を core / optional で整理
  * examples/notebooks への導線を付ける

* [ ] **ドキュメントサイトを更新**

  * public API の説明を論文記述と一致させる
  * DTT XML, WIN, ATS, waveform/frame 系の読み込み例を明確化
  * metadata-preserving design の説明を補強する

* [ ] **引用・参考情報をリポジトリ側にも反映**

  * BruCo, GWpy, HHT/EMD など関連技術の出典整理
  * README や docs で関連プロジェクトへの参照を適切に示す

## 参考文献まわり

* [ ] **BruCo の参照先を正式化**

  * GitHub リポジトリ参照
  * Phys. Rev. D 論文参照
  * placeholder URL を完全に除去

* [ ] **追加文献の採否を最終決定**

  * LIGO/Virgo white paper
  * detector characterization / transient noise 論文
  * ObsPy
  * PyTorch
  * 必要最小限に絞る

## 投稿前の最終確認

* [ ] **論文本文と公開コードの対応を最終点検**

  * 論文で主張する機能が本当に release に入っているか確認
  * 逆に release にあるが論文で触れない機能の扱いを整理

* [ ] **Code metadata の実値を確定**

  * version
  * repository URL
  * DOI
  * minimum Python version
  * core dependencies
  * optional extras

* [ ] **図表と補助資料の数を抑える**

  * SoftwareX の制約内に収める
  * 補助的な詳細図は docs / supplement 側に逃がす

* [ ] **英語表現の最終校正**

  * SoftwareX 向けに冗長さを削る
  * “what the software does” と “why it matters” に集中させる
  * 実装詳細に寄りすぎる箇所を整理する

必要なら次に、これを **GitHub Issues 用のチェックリスト** か **論文投稿準備用の進行表** に整形します。
