import faulthandler
import importlib
import multiprocessing as _mp
import os
import socket
from pathlib import Path

import gwpy.conftest as _gwpy_conftest
import gwpy.time._tconvert as _tconvert
import numpy as np
import pytest
from gwpy.conftest import *  # noqa: F401,F403
from matplotlib import rcParams
from matplotlib import use as mpl_use

_PYTEST_QT_AVAILABLE = importlib.util.find_spec("pytestqt") is not None

if not _PYTEST_QT_AVAILABLE:
    @pytest.fixture
    def qtbot():  # type: ignore[override]
        pytest.skip("pytest-qt not installed")

pytest_plugins = ["gwpy.testing.fixtures"]

faulthandler.enable()

_QT_QPA_PLATFORM = os.environ.get("QT_QPA_PLATFORM", "").lower()
_HEADLESS = (
    not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
) or _QT_QPA_PLATFORM in {"offscreen", "minimal"}

if _HEADLESS:
    collect_ignore_glob = [
        "tests/gui/*",
        "tests/gui/**",
        "tests/nds/test_gui_nds_smoke.py",
        "tests/e2e/test_gui_smoke.py",
    ]

    _skip_gui = pytest.mark.skip(reason="GUI tests skipped (headless)")

    def pytest_collection_modifyitems(config, items):
        for item in items:
            if "gui" in item.keywords:
                item.add_marker(_skip_gui)

# =============================================================================
# Helper functions
# =============================================================================

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        # Best-effort; permissions may be controlled by the environment.
        pass

def _requirement_available(name: str) -> bool:
    if name == "uproot" and os.environ.get("GWEXPY_ALLOW_UPROOT", "") != "1":
        return False
    try:
        importlib.import_module(name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False

def _semlock_available() -> bool:
    try:
        ctx = _mp.get_context()
        ctx.Lock()
        return True
    except PermissionError:
        return False
    except OSError as exc:
        return exc.errno not in (None, 13)

def _is_network_error(exc: BaseException) -> bool:
    try:
        import requests
    except ImportError:
        requests = None
    try:
        import urllib3
    except ImportError:
        urllib3 = None

    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, socket.gaierror):
            return True
        if requests and isinstance(current, requests.exceptions.RequestException):
            return True
        if urllib3 and isinstance(current, urllib3.exceptions.HTTPError):
            return True
        current = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None
        )
    return False

# =============================================================================
# Environment Configuration
# =============================================================================

_ROOT = Path(__file__).resolve().parent

# Match gwpy's test setup for consistent plotting behavior.
mpl_use("agg", force=True)
rcParams.update({"text.usetex": False})
np.random.seed(1)

# Ensure a writable home for packages that write to ~/.<tool> directories.
_home_dir = _ROOT / ".home"
_ensure_dir(_home_dir)
try:
    _arviz_dir = Path.home() / "arviz_data"
    _arviz_dir.mkdir(exist_ok=True)
    _probe = _arviz_dir / ".write_probe"
    _probe.write_text("ok")
    _probe.unlink()
except Exception:
    os.environ["HOME"] = str(_home_dir)

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
    _sem = _mp.Semaphore(1)
    _sem.acquire()
    _sem.release()
except Exception as exc:
    _MP_AVAILABLE = False
    _MP_REASON = f"multiprocessing semaphores unavailable: {exc}"

# =============================================================================
# Hooks
# =============================================================================

def pytest_configure(config):
    # From gwexpy/conftest.py
    _gwpy_conftest.pytest_configure(config)
    config.addinivalue_line(
        "markers",
        "requires(module): skip test if optional dependency cannot be imported",
    )

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
    # 1. requirement check from gwexpy/conftest
    marker_req = item.get_closest_marker("requires")
    if marker_req:
        missing = [req for req in marker_req.args if not _requirement_available(str(req))]
        if missing:
            pytest.skip(f"missing optional dependency: {', '.join(missing)}")

    # 2. freeze_time setup from tests/conftest
    marker_freeze = item.get_closest_marker("freeze_time")
    if marker_freeze:
        try:
            from freezegun import freeze_time
        except Exception:
            pytest.skip("freezegun not installed")
        freezer = freeze_time(*marker_freeze.args, **marker_freeze.kwargs)
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

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def _mp_fallback(monkeypatch):
    if _semlock_available():
        return
    import gwpy.utils.mp as _gwpy_mp

    original = _gwpy_mp.multiprocess_with_queues

    def _serial_only(nproc, func, inputs, **kwargs):
        return original(1, func, inputs, **kwargs)

    monkeypatch.setattr(_gwpy_mp, "multiprocess_with_queues", _serial_only)


@pytest.fixture(autouse=True)
def _datafind_network_fallback(monkeypatch):
    import gwpy.timeseries as _gwpy_ts

    original = _gwpy_ts.TimeSeriesDict.find.__func__

    def _wrapped(cls, *args, **kwargs):
        network_exceptions = [OSError, TimeoutError, socket.gaierror]
        try:
            import requests
        except ImportError:
            requests = None
        try:
            import urllib3
        except ImportError:
            urllib3 = None
        if requests is not None:
            network_exceptions.append(requests.exceptions.RequestException)
        if urllib3 is not None:
            network_exceptions.append(urllib3.exceptions.HTTPError)
        try:
            return original(cls, *args, **kwargs)
        except tuple(network_exceptions) as exc:
            if _is_network_error(exc) and kwargs.get("observatory") is None:
                raise RuntimeError("datafind network unavailable") from exc
            raise

    monkeypatch.setattr(_gwpy_ts.TimeSeriesDict, "find", classmethod(_wrapped))


@pytest.fixture
def requests_mock():
    requests_mock = pytest.importorskip("requests_mock")
    with requests_mock.Mocker() as mock:
        yield mock


@pytest.fixture(autouse=True)
def _freeze_time_marker(monkeypatch, request):
    marker = request.node.get_closest_marker("freeze_time")
    if marker is None:
        return
    freeze_value = marker.args[0] if marker.args else marker.kwargs.get("time")
    if not freeze_value:
        return

    frozen_dt = _tconvert.datetime.datetime.fromisoformat(str(freeze_value))
    frozen_date = frozen_dt.date()

    def _now():
        return frozen_dt.replace(microsecond=0)

    def _today():
        return frozen_date

    def _today_delta(**delta):
        return frozen_date + _tconvert.datetime.timedelta(**delta)

    def _tomorrow():
        return _today_delta(days=1)

    def _yesterday():
        return _today_delta(days=-1)

    monkeypatch.setattr(_tconvert, "_now", _now, raising=False)
    monkeypatch.setattr(_tconvert, "_today", _today, raising=False)
    monkeypatch.setattr(_tconvert, "_today_delta", _today_delta, raising=False)
    monkeypatch.setattr(_tconvert, "_tomorrow", _tomorrow, raising=False)
    monkeypatch.setattr(_tconvert, "_yesterday", _yesterday, raising=False)
    date_strings = dict(_tconvert.DATE_STRINGS)
    date_strings.update(
        {
            "now": _now,
            "today": _today,
            "tomorrow": _tomorrow,
            "yesterday": _yesterday,
        }
    )
    monkeypatch.setattr(_tconvert, "DATE_STRINGS", date_strings, raising=False)


@pytest.fixture(autouse=True)
def _cleanup_qt_widgets(request):
    yield

    path_str = str(request.fspath)
    if "tests/gui" not in path_str and "tests/e2e" not in path_str:
        return

    try:
        from PyQt5 import QtCore, QtWidgets
    except Exception:
        return

    app = QtWidgets.QApplication.instance()
    if app is None:
        return

    for widget in app.topLevelWidgets():
        try:
            widget.close()
        except Exception:
            continue

    app.processEvents()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()
