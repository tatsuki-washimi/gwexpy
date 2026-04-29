Graphical User Interface
========================

**Stability:** Experimental

.. warning::
   The GWexpy GUI (``gwexpy.gui``) is **experimental** and not part of the first
   PyPI supported surface. It requires the ``.[gui]`` optional dependency
   (PyQt5, pyqtgraph) for source/development use. The API may change without
   notice.

To launch the GUI after installing the optional dependency:

.. code-block:: bash

   pip install "gwexpy[gui] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
   python -m gwexpy.gui

.. currentmodule:: gwexpy.gui

.. automodule:: gwexpy.gui
   :members:
   :undoc-members:
   :show-inheritance:
