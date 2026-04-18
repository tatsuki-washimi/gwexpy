前処理 (Preprocessing)
======================

**安定性:** 安定

.. seealso::

   :ref:`validated-ja-adaptive-whitening`
      適応ホワイトニングと自動安定化パラメータの前提条件および検証根拠。
   :doc:`../../user_guide/numerical_stability`
      ``eps="auto"`` などの安定化デフォルトに関する説明。
   :doc:`../../user_guide/tutorials/case_ml_preprocessing`
      ホワイトニングを含む前処理フローの事例。
   :doc:`../../user_guide/tutorials/ml_preprocessing_methods`
      前処理ユーティリティを手法別に整理したガイド。
   :doc:`../../user_guide/tutorials/case_wiener_filter`
      前処理の選択がノイズ差し引きにどう効くかを見る関連ケーススタディ。

前処理ユーティリティは ``gwexpy.signal.preprocessing`` にあります。

主なコンポーネント:

- ``MLPreprocessor``
- ``standardize`` / ``StandardizationModel``
- ``whiten`` / ``WhiteningModel``
- ``impute``

利用例:

- :doc:`../../user_guide/tutorials/case_ml_preprocessing`
- :doc:`../../user_guide/tutorials/ml_preprocessing_methods`
