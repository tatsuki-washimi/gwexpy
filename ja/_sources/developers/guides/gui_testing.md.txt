# GUI Testing (Headless)

## Quick Run

```bash
PYTHONFAULTHANDLER=1 xvfb-run -a -s "-screen 0 1920x1080x24" pytest -q
```

For convenience, you can also run:

```bash
scripts/run_gui_tests.sh
```

GUI test failures attempt to capture a screenshot to `PYTEST_SCREENSHOT_DIR`
(defaults to `tests/.screenshots`).

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
