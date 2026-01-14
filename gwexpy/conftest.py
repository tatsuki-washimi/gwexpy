import importlib
import multiprocessing as _mp
import socket

import gwpy.conftest as _gwpy_conftest
import gwpy.time._tconvert as _tconvert
import pytest
from gwpy.conftest import *  # noqa: F401,F403


def pytest_configure(config):
    _gwpy_conftest.pytest_configure(config)
    config.addinivalue_line(
        "markers",
        "requires(module): skip test if optional dependency cannot be imported",
    )


def _requirement_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def pytest_runtest_setup(item):
    marker = item.get_closest_marker("requires")
    if marker is None:
        return
    missing = [req for req in marker.args if not _requirement_available(str(req))]
    if missing:
        pytest.skip(f"missing optional dependency: {', '.join(missing)}")


def _semlock_available() -> bool:
    try:
        ctx = _mp.get_context()
        ctx.Lock()
        return True
    except PermissionError:
        return False
    except OSError as exc:
        return exc.errno not in (None, 13)


@pytest.fixture(autouse=True)
def _mp_fallback(monkeypatch):
    if _semlock_available():
        return
    import gwpy.utils.mp as _gwpy_mp

    original = _gwpy_mp.multiprocess_with_queues

    def _serial_only(nproc, func, inputs, **kwargs):
        return original(1, func, inputs, **kwargs)

    monkeypatch.setattr(_gwpy_mp, "multiprocess_with_queues", _serial_only)


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
