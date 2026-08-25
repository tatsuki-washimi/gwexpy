スペクトログラム (Spectrogram)
==============================

.. note::
   ページ種別: 二次 API カテゴリ

**安定性:** 安定

.. currentmodule:: gwexpy.spectrogram

概要
----

来歴情報
--------

``Spectrogram.provenance`` は解析結果に付随できる任意の、取得時に複製されるマッピングです。
v0.2.0 の形式は ``schema="gwexpy.spectrogram.provenance"`` と ``schema_version=1`` を持ちます。
JSON 安全な値だけを受け付けます。
乱数生成器のような実行時オブジェクトは受け付けません。
このマッピングは copy、スライス、Spectrogram を返す算術演算、pickle、明示的な HDF5 往復で保持されます。
GWpy が ``.h5`` または ``.hdf5`` のファイル名から HDF5 を推論する場合も sidecar を保持します。
``.hdf`` のファイル名では ``format="hdf5"`` の指定が必要です。
HDF5 では GWexpy のファイルレベル sidecar に保存するため、GWpy は本来のデータセットをそのまま読めます。
sidecar は書き込み前に検証し、サイズを 1 MB に制限します。
同一プロセス内では、同じ物理ファイルに対する provenance 対応の読み取りと更新はファイル単位のロックを共有します。
そのため、読み取り側は置換後のデータと置換前の sidecar の組を観測しません。
通常の更新失敗時には、元のデータセットリンクと sidecar の状態を復元します。
復元または rollback cleanup が失敗したときは、名前付きの recovery artifact を残そうとします。
エラーには、操作時、復元時、保存時、cleanup 時のすべての失敗を含めます。
artifact の作成にも失敗したときは、以前の sidecar snapshot を直接再適用してから recovery が利用できないことをエラーで報告します。
パスを置き換える書き込みでは、完全な一時 HDF5 ファイルを同じディレクトリに書き、``os.replace`` で置き換えます。
既存の通常ファイルでは、元のパーミッションビットを一時ファイルに適用します。
シンボリックリンクを対象とした置き換えは拒否します。
所有者、ACL、拡張属性はこの操作では保存しません。
ファイル単位のロックと復元はプロセス間 HDF5 トランザクションを保証しません。
その保証は v0.2.0 の対象外です。
来歴なしの pickle は GWpy だけで読み込めますが、来歴付き Spectrogram の pickle を読み込むには GWexpy が必要です。

.. note::
   学習ステップ:
   入門チュートリアルの後や、時間周波数ワークフローから正確な API 詳細に戻りたい場合にこのページを使ってください。

.. seealso::

   :doc:`../../user_guide/tutorials/index`
      機能別に学び始めるためのチュートリアル一覧。
   :doc:`../../user_guide/tutorials/intro_spectrogram`
      API を引く前に確認したい ``Spectrogram`` の基本例。
   :doc:`../FFT_Conventions`
      GWexpy が採用するフーリエ正規化と軸の規約。
   :doc:`../../user_guide/tutorials/case_signal_extraction`
      ``Spectrogram`` 系 API に戻れる時間周波数事例。
   :doc:`../../user_guide/numerical_stability`
      FFT ベースの時間周波数解析で確認すべき安定化ノート。
   :doc:`../topics`
      規約や高度・理論解説を概念別にたどる入口。

.. autosummary::
   :toctree: _autosummary

   Spectrogram

Spectrogram クラス
------------------

.. autoclass:: Spectrogram
   :no-index:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :member-order: bysource

   .. rubric:: メソッド

   .. autosummary::

      ~Spectrogram.plot
      ~Spectrogram.crop
      ~Spectrogram.percentile
      ~Spectrogram.ratio
      ~Spectrogram.filter

モジュール内容
--------------

.. automodule:: gwexpy.spectrogram
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Spectrogram
