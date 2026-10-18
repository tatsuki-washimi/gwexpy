---
orphan: true
---

# GUI applications: migration notice

GWexpy provides the analysis library, not a bundled GUI application. The former
source-checkout-only `gwexpy.gui` implementation has been removed from this
repository and is not part of the published GWexpy package.

The standalone [pyaggui project](https://github.com/tatsuki-washimi/pyaggui)
owns the extracted legacy GUI. GWexpy Studio is another independent GUI project,
maintained in a separate repository; do not confuse it with pyaggui.

Use GWexpy's Python API for analysis and plotting; consult each GUI project's
own installation and compatibility instructions for app-specific workflows.

This page is retained for existing inbound links and intentionally contains no
instructions to run or import the removed `gwexpy.gui` application.
