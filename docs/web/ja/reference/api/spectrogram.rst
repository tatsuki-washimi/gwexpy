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
sidecar の key は ``Dataset.name`` と完全に一致する正規化済み絶対 HDF5 dataset 名（例: ``/group/disk``）です。
通常の更新失敗時には、元のデータセットリンクと sidecar の状態を復元します。
復元または rollback cleanup が失敗したときは、名前付きの recovery artifact を残そうとします。
エラーには、操作時、復元時、保存時、cleanup 時のすべての失敗を含めます。
報告する artifact には、復元可能な保存済みデータセットまたは以前の sidecar snapshot が含まれます。
空または利用不能な group は recovery が利用できないものとして報告します。
sidecar だけの artifact は、厳密な boolean の不在 marker、またはサイズ上限・JSON・パス・provenance schema の検証を通る保存済み sidecar を持つ場合だけ利用可能です。
この検査は artifact と公開状態のどちらも変更しません。
エラー一覧は実際の発生順であり、復元、保存、cleanup のカテゴリ別一覧も個別に利用できます。
エラーメッセージには長さを制限した安全な説明を使い、元の exception object を保ったまま信頼できない exception formatting を呼び出しません。
rollback error には常に少なくとも一つの因果 exception が残ります。
無効な内部構成では空または誤解を招く rollback state の代わりに synthetic invariant error を記録します。
有効な内部 rollback state は、操作、復元または cleanup、保存の順序と同一 object identity を厳密に保持します。
commit 済みの書き込みでは cleanup と保存の失敗だけを記録します。
direct internal construction で任意の event sequence を省略した場合は、検証済みの phase fields からこの順序を導出します。
明示した sequence は厳密に検証されます。
artifact の作成にも失敗したときは、以前の sidecar snapshot を直接再適用してから recovery が利用できないことをエラーで報告します。
データと sidecar の commit 後に rollback cleanup が失敗したとき、書き込み結果は commit 済みのままです。
構造化エラーの ``operation_committed=True`` がこの状態を示します。
path name で ``overwrite=True`` かつ既定の ``append=False`` を指定した場合、GWpy と同じ
file 全体の置換 semantics になります。GWexpy は要求された dataset と sidecar entry だけを
完全な sibling temporary HDF5 file に書き、``os.replace`` で置き換えます。
この操作では無関係な dataset と provenance entry も意図的に削除されます。
対して ``append=True`` は transaction lock 下で既存 file を mutation します。
異なる dataset path とその sidecar entry は保持され、同じ path は ``overwrite=True`` のときだけ置換されます。
open された ``h5py.File`` または ``h5py.Group`` も、同じ file 内・dataset 単位の mutation semantics を使います。
したがって、二つの pathname replacement writer は serialize されますが file scope では last-writer-wins です。
異なる path を保持するには ``append=True`` または open container を使ってください。
既存の通常ファイルでは、元のパーミッションビットを一時ファイルに適用します。
シンボリックリンクを対象とした置き換えは拒否します。
所有者、ACL、拡張属性はこの操作では保存しません。
provenance 対応のパス名 read/write は、耐久的な sibling lock file に対する、最大 10 秒の
POSIX advisory transaction lock も取得します。
相対・絶対・symbolic-link のパス別名は同じ lock に解決されます。
lock は data と sidecar の更新および recovery 全体を覆うため、別プロセスの
GWexpy provenance 対応 reader/writer は選択された operation scope の更新前または commit 済みの組だけを観測します。
取得できない場合は structured ``CrossProcessHDF5LockError`` で fail closed します。
lock file は意図的に残し stale とみなしたり steal したりせず、プロセス終了時の解放は OS に委ねます。
この保証には POSIX ``flock`` とそれを尊重する local filesystem が必要です。
未対応 platform と anonymous file object は fail closed します。
呼び出し側が所有する ``h5py`` handle は、GWexpy の provenance 対応 operation に渡された間だけ参加します。
独立して開かれた handle やその直接 mutation は transaction の対象外です。
distributed filesystem / network filesystem への保証は主張しません。
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
