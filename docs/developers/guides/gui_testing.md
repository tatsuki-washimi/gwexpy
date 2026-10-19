# GUI Testing (Historical Reference)

.. note::
   The legacy GUI implementation and tests (`gwexpy/gui/`, `tests/gui/`) were removed
   from the GWexpy repository and extracted into the standalone `pyaggui` project.
   For active GUI testing and development, refer to `pyaggui <https://github.com/tatsuki-washimi/pyaggui>`_.

The instructions below are preserved for historical reference only.

## Default Suite (Legacy)

```bash
tests/run_gui_tests.sh
```

The default runner used `xvfb-run` and excluded `@pytest.mark.pyautogui` tests
unless explicitly passing a custom `-m/--markexpr`.
