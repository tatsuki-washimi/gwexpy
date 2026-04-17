# 前提条件と規約

このページは、GWexpy のガイドやチュートリアルを読む前に把握しておきたい**共通の前提条件**と**規約**の入口です。
個別アルゴリズムの仮定や数式の詳細は各ページに残し、このページでは「どこを先に確認すべきか」を整理します。

## 1. 環境前提

- 基本的な利用環境は **Python 3.11 以上** を前提にしています。
- 最低限の前提知識は、**Python の基礎**、**NumPy 配列操作**、必要に応じて **Matplotlib** です。
- optional dependency によって使える機能が増えます。導入手順は [インストールガイド](installation.md) を参照してください。

まず全体の学習導線を把握したい場合は [はじめに](getting_started.md) から入るのが最短です。

## 2. データと時刻の前提

- GWexpy は `gwpy` 系の時系列・周波数系列コンテナとの互換性を重視しています。
- 時刻の扱いでは **GPS 時刻** を前提とする API があります。特に ARIMA 予測時刻などは UTC の閏秒系と混同しないでください。
- ファイル形式によっては絶対時刻を保持しません。たとえば音声形式は `t0=0.0` を便宜的に使う場合があります。
- ローカル時刻だけを持つ形式では `timezone` の明示が必要です。代表例は [I/O 対応フォーマットガイド](io_formats.md) の GBD です。

アルゴリズムごとの仮定まで確認したい場合は [検証済みアルゴリズム](validated_algorithms.md) を参照してください。

## 3. FFT とスペクトルの規約

- GWexpy では、FFT の**正規化**、**片側/両側スペクトル**、**符号規約**を明示的に扱います。
- `fft_time` と `fft_space` では、対象軸と正規化の考え方が異なります。
- `spectral_density` では PSD と spectrum の意味を区別して扱います。

数式レベルの詳細は [FFT の仕様とコンベンション](../reference/FFT_Conventions.md) にまとめています。

## 4. GWpy 互換と GWexpy 拡張

- GWexpy は GWpy の上に構築されており、基本的なデータモデルと操作感を尊重しています。
- 一方で、Matrix 系コンテナ、Field API、追加 I/O、解析ユーティリティなど GWpy にはない拡張があります。
- 「GWpy と同じつもりで使ってよい部分」と「GWexpy 固有の追加要素」を分けて確認したい場合は、移行ガイドを見るのが早いです。

GWpy からの移行観点は [GWpy からの移行](gwexpy_for_gwpy_users_ja.md) と [GWpy 差分 API 一覧](gwpy_added_api_index_ja.md) を参照してください。

## 5. どこから読むか

- はじめて使う場合: [はじめに](getting_started.md)
- すぐに手を動かしたい場合: [チュートリアル一覧](tutorials/index.rst)
- 数学的な FFT 規約を確認したい場合: [FFT の仕様とコンベンション](../reference/FFT_Conventions.md)
- アルゴリズムごとの仮定や検証根拠を確認したい場合: [検証済みアルゴリズム](validated_algorithms.md)
- 既存コードを GWpy から移行したい場合: [GWpy からの移行](gwexpy_for_gwpy_users_ja.md)

## 関連ページ

- [インストールガイド](installation.md)
- [はじめに](getting_started.md)
- [ファイル I/O 対応フォーマットガイド](io_formats.md)
- [検証済みアルゴリズム](validated_algorithms.md)
- [FFT の仕様とコンベンション](../reference/FFT_Conventions.md)
