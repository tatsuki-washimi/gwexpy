# GUI Testing

GWexpy no longer contains the legacy GUI implementation, GUI behavior tests,
or their test runner. GUI development and display-dependent testing are
maintained in the standalone [pyaggui project](https://github.com/tatsuki-washimi/pyaggui).
Follow that repository's current contributor instructions for GUI testing.

GWexpy's `tests/test_gui_boundary.py` checks the repository boundary. It does
not run or certify GUI behavior.
