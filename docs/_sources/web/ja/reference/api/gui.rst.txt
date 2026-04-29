グラフィカルユーザーインターフェース (GUI)
==========================================

**安定性:** Experimental

.. warning::
   GWexpy GUI（``gwexpy.gui``）は **試験実装段階** であり、初回 PyPI
   配布物には含めません。ソース / 開発環境で動作させる場合は GUI
   依存関係（PyQt5, pyqtgraph, qtpy, sounddevice）を明示的に
   インストールしてください。API は予告なく変更される可能性があります。

ソース checkout または開発インストールでオプション依存関係を入れた後の起動方法：

.. code-block:: bash

   pip install PyQt5 pyqtgraph qtpy sounddevice
   python -m gwexpy.gui

.. currentmodule:: gwexpy.gui

.. automodule:: gwexpy.gui
   :members:
   :undoc-members:
   :show-inheritance:
