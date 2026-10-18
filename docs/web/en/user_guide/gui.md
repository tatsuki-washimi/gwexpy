# GUI applications (separate repositories)

GWexpy is an analysis library, not a GUI application. The old development-only
`gwexpy.gui` module has been removed from the repository; it was not part of
the published GWexpy PyPI distribution.

- [pyaggui](https://github.com/tatsuki-washimi/pyaggui) is an independent GUI application. Follow that repository's installation, CLI, and test instructions.
- **GWexpy Studio** is developed independently in its own repository. Follow that project's installation and compatibility guidance.

Existing Python analysis and visualization APIs such as `gwexpy.plot` remain
available. This migration removes the old GUI application code, not the
library's plotting capabilities.

See [GWexpy Issue #645](https://github.com/tatsuki-washimi/gwexpy/issues/645)
for the separation record.
