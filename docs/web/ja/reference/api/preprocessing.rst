前処理 (Preprocessing)
======================

**安定性:** Stable

.. seealso::

   :ref:`validated-ja-adaptive-whitening`
      適応ホワイトニングと自動安定化パラメータの前提条件および検証根拠。
   :doc:`../../user_guide/validated_algorithms`
      検証済みアルゴリズムの全体像。

前処理ユーティリティは ``gwexpy.signal.preprocessing`` にあります。

主なコンポーネント:

- ``MLPreprocessor``
- ``standardize`` / ``StandardizationModel``
- ``whiten`` / ``WhiteningModel``
- ``impute``

利用例:

- :doc:`../../user_guide/tutorials/case_ml_preprocessing`
