グラフィカルユーザーインターフェース (GUI)
==========================================

**安定性:** Experimental

.. warning::
   GWexpy GUI（``gwexpy.gui``）は **試験実装段階** であり、初回 PyPI の
   サポート対象には含めません。ソース / 開発環境で動作させる場合は
   ``.[gui]`` オプション依存関係（PyQt5, pyqtgraph）が必要です。
   API は予告なく変更される可能性があります。

オプション依存関係をインストールした後の起動方法：

.. code-block:: bash

   pip install "gwexpy[gui] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
   python -m gwexpy.gui

.. currentmodule:: gwexpy.gui

.. automodule:: gwexpy.gui
   :members:
   :undoc-members:
   :show-inheritance:
