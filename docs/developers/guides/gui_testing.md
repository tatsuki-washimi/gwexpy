# GUI testing: moved to application repositories

This GWexpy repository no longer contains the legacy `gwexpy.gui`
implementation or its GUI test runners. Consequently `tests/run_gui_tests.sh`,
`tests/run_gui_nds_tests.sh` and `tests/gui/` are no longer valid commands or paths.

The extracted GUI is developed and tested in
[pyaggui](https://github.com/tatsuki-washimi/pyaggui). GWexpy Studio is a
different, independently maintained GUI application; consult the corresponding
application repository for its test runners, display dependencies and CI.

Core GWexpy library testing remains in `tests/` and
`scripts/ci/run_gate.py`. Historical GUI testing procedures are available
in the Git history preceding [the cleanup](https://github.com/tatsuki-washimi/gwexpy/issues/645).
