スカラーフィールドのスライス操作ガイド
========================================

ScalarFieldの4次元構造維持の挙動
---------------------------------

``ScalarField`` は、インデクシング操作を行っても常に4次元構造を維持します。これはNumPy配列やGWpyの標準的な挙動とは異なります。

4次元構造の維持
~~~~~~~~~~~~~~~

**NumPy配列の場合**:

.. code-block:: python

   >>> import numpy as np
   >>> arr = np.zeros((10, 5, 5, 5))
   >>> arr[0].shape
   (5, 5, 5)  # 次元が削減される

**ScalarFieldの場合**:

.. code-block:: python

   >>> from gwexpy.fields import ScalarField
   >>> field = ScalarField(np.zeros((10, 5, 5, 5)), ...)
   >>> field[0].shape
   (1, 5, 5, 5)  # 4次元を維持

この挙動により、以下のメリットがあります:

1. **軸メタデータの保持**: ``axis0_domain``, ``space_domain`` などが欠落しません
2. **ブロードキャスト操作の一貫性**: 常に4次元として扱えます
3. **FFT操作の安全性**: ドメイン情報を保持したままFFTを実行できます

次元を削減したい場合
~~~~~~~~~~~~~~~~~~~~~~~~~~

明示的に次元を削減するには、``squeeze()`` メソッドを使用します:

.. code-block:: python

   >>> field[0].squeeze().shape
   (5, 5, 5)  # 長さ1の軸が削除される

スライスの例
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   >>> # 時間方向の特定の時刻を抽出
   >>> snapshot = field[100]  # shape: (1, 5, 5, 5)

   >>> # 空間的な断面を抽出
   >>> plane = field[:, :, :, 2]  # shape: (n_time, 5, 5, 1)

   >>> # 特定の空間点の時系列
   >>> point_ts = field[:, 2, 2, 2]  # shape: (n_time, 1, 1, 1)
   >>> # TimeSeries的に扱いたい場合はsqueeze
   >>> point_ts_1d = point_ts.squeeze()  # shape: (n_time,)

よくある質問（FAQ）
------------------

なぜ4次元を維持するのですか？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ScalarFieldは、軸ごとに異なるドメイン（時間/周波数、実空間/波数空間）を持つ物理量を表現します。次元を削減すると、これらのメタデータが失われ、FFT操作やドメイン変換が正しく動作しなくなります。

NumPyライクな挙動が必要な場合は？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``squeeze()`` メソッドを使用してください。これにより、長さ1の軸を削除し、NumPyライクな配列を取得できます。

トラブルシューティング
----------------------------

ブロードキャスト操作が期待通り動作しない
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ScalarFieldは常に4次元なので、他の配列との演算時には形状を合わせる必要があります。

.. code-block:: python

   # 誤り: 1次元配列との演算
   field + np.array([1, 2, 3])  # エラー

   # 正しい: 形状を合わせる
   field + np.array([1, 2, 3]).reshape(3, 1, 1, 1)

関連リンク
------------------

- :doc:`../tutorials/field_scalar_intro` - ScalarField 入門チュートリアル
- :doc:`../reference/ScalarField` - ScalarField API リファレンス
- :doc:`../reference/FieldList` - FieldList / FieldDict コレクション
