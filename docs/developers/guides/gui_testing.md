# GUI Testing

## Default Suite

```bash
tests/run_gui_tests.sh
```

The default runner uses `xvfb-run` and excludes `@pytest.mark.pyautogui` tests
unless you explicitly pass your own `-m/--markexpr`.

## PyAutoGUI

PyAutoGUI tests require a real system display with working pointer injection.
Run them explicitly:

```bash
GUI_TEST_TARGET=tests/gui/integration/test_pyautogui_smoke.py \
tests/run_gui_tests.sh
```

If you need to force the real display for another GUI target, use:

```bash
GUI_USE_SYSTEM_DISPLAY=1 tests/run_gui_tests.sh
```

The runner will populate `XAUTHORITY` automatically when possible.

GUI test failures attempt to capture a screenshot under `tests/gui/screenshots`.

## Core Dumps

Enable core dumps for the current shell:

```bash
ulimit -c unlimited
```

Run tests (headless is recommended). If a crash occurs, inspect with systemd:

```bash
coredumpctl list | tail -n 5
coredumpctl info <PID>
coredumpctl gdb <PID>
```

## gdb (Single Test)

```bash
PYTHONFAULTHANDLER=1 xvfb-run -a -s "-screen 0 1920x1080x24" \
  gdb --args python -m pytest -q tests/gui/test_gui_data_backend.py::test_start_stop_sequence
```

## valgrind (Single Test)

```bash
PYTHONFAULTHANDLER=1 xvfb-run -a -s "-screen 0 1920x1080x24" \
  valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
  python -m pytest -q tests/gui/test_gui_data_backend.py::test_start_stop_sequence
```
