---
orphan: true
---

# Graphical applications are separate projects

**GWexpy is the analysis library.** It does not include GUI application code, a
GUI launch command, or Qt/PyQt runtime dependencies in its repository or wheel.

For interactive GUI workflows, use an independent application:

- [pyaggui](https://github.com/tatsuki-washimi/pyaggui) — a separate diaggui-like GUI with its own code, installation instructions and testing.
- **GWexpy Studio** — a separately maintained GUI project; consult its own repository for installation and supported features.

Existing analysis examples using `gwexpy.timeseries`, `gwexpy.plot` and other
library APIs remain supported as before. Removing the former development-only
GUI source does not remove GWexpy's plotting functions.

If you used the old `gwexpy.gui` development API, move the GUI application
imports, deployment and tests to the relevant external project. Historical GUI
implementation notes remain available in Git history; see
[repository split tracking](https://github.com/tatsuki-washimi/gwexpy/issues/645).
