import faulthandler
import os
from pathlib import Path
from pathlib import Path

import pytest

import numpy as np
from matplotlib import rcParams
from matplotlib import use as mpl_use

pytest_plugins = ["gwpy.testing.fixtures"]

faulthandler.enable()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        # Best-effort; permissions may be controlled by the environment.
        pass


_ROOT = Path(__file__).resolve().parent

# Match gwpy's test setup for consistent plotting behavior.
mpl_use("agg", force=True)
rcParams.update({"text.usetex": False})
np.random.seed(1)

# Keep Qt runtime warnings quiet by providing a private runtime dir.
_runtime_dir = _ROOT / ".qt-runtime"
_ensure_dir(_runtime_dir)
os.environ["XDG_RUNTIME_DIR"] = str(_runtime_dir)

# Provide a writable GMT user dir for pygmt tests.
_gmt_dir = _ROOT / ".gmt"
_ensure_dir(_gmt_dir)
os.environ.setdefault("GMT_USERDIR", str(_gmt_dir))

# Avoid IPython history errors by forcing a writable directory.
if not os.environ.get("IPYTHONDIR"):
    _ipy_dir = _ROOT / ".ipython"
    _ensure_dir(_ipy_dir)
    os.environ["IPYTHONDIR"] = str(_ipy_dir)

# Prefer offscreen Qt unless the user already configured a platform.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
if not os.environ.get("NUMBA_CACHE_DIR"):
    _numba_cache = _ROOT / ".numba-cache"
    _ensure_dir(_numba_cache)
    os.environ["NUMBA_CACHE_DIR"] = str(_numba_cache)

try:
    import sqlite3
    from IPython.core import history as _ip_hist

    _orig_end_session = _ip_hist.HistoryManager.end_session

    def _safe_end_session(self):  # type: ignore[override]
        try:
            _orig_end_session(self)
        except sqlite3.OperationalError:
            pass

    _ip_hist.HistoryManager.end_session = _safe_end_session  # type: ignore[assignment]
except Exception:
    pass

_FREEZE_ATTR = "_gwexpy_freezegun"

_MP_AVAILABLE = True
_MP_REASON = ""
try:
    import multiprocessing as _mp

    _sem = _mp.Semaphore(1)
    _sem.acquire()
    _sem.release()
except Exception as exc:
    _MP_AVAILABLE = False
    _MP_REASON = f"multiprocessing semaphores unavailable: {exc}"


def pytest_collection_modifyitems(config, items):
    if _MP_AVAILABLE:
        skip_mp = None
    else:
        skip_mp = pytest.mark.skip(reason=_MP_REASON)

    for item in items:
        path = str(item.fspath)
        if skip_mp and path.endswith("test_mp.py"):
            item.add_marker(skip_mp)
        if skip_mp and hasattr(item, "callspec"):
            nproc = item.callspec.params.get("nproc")
            if isinstance(nproc, int) and nproc > 1:
                item.add_marker(skip_mp)


def pytest_runtest_setup(item):
    marker = item.get_closest_marker("freeze_time")
    if marker:
        try:
            from freezegun import freeze_time
        except Exception:
            pytest.skip("freezegun not installed")
        freezer = freeze_time(*marker.args, **marker.kwargs)
        freezer.start()
        setattr(item, _FREEZE_ATTR, freezer)


def pytest_runtest_teardown(item, nextitem):
    freezer = getattr(item, _FREEZE_ATTR, None)
    if freezer is not None:
        freezer.stop()
        delattr(item, _FREEZE_ATTR)


def _screenshot_dir() -> Path:
    return Path(os.getenv("PYTEST_SCREENSHOT_DIR", "tests/.screenshots"))


def _find_main_window():
    try:
        from PyQt5 import QtWidgets
    except Exception:
        return None

    app = QtWidgets.QApplication.instance()
    if app is None:
        return None

    active = app.activeWindow()
    if active is not None:
        return active

    for widget in app.topLevelWidgets():
        if widget.isVisible():
            return widget
    return None


def pytest_runtest_makereport(item, call):
    if call.when != "call" or call.excinfo is None:
        return

    path_str = str(item.fspath)
    if "tests/gui" not in path_str and "tests/e2e" not in path_str:
        return

    window = _find_main_window()
    if window is None:
        return

    try:
        target_dir = _screenshot_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = item.nodeid.replace("/", "_").replace("::", "_")
        path = target_dir / f"{safe_name}.png"
        pixmap = window.grab()
        pixmap.save(str(path))
    except Exception:
        return
