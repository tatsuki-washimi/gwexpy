"""Tests for exact PyPI artifact readback."""

from __future__ import annotations

import hashlib
import http.client
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "ci" / "validate_pypi_readback.py"
)


def _load_validator():
    spec = importlib.util.spec_from_file_location("pypi_readback_validator", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pypi_case():
    version = "99.88.77"
    package_bytes = {
        f"gwexpy-{version}-py3-none-any.whl": b"synthetic wheel bytes",
        f"gwexpy-{version}.tar.gz": b"synthetic sdist bytes",
    }
    manifest = {
        "schema": "gwexpy-release-promotion-manifest-v1",
        "version": version,
        "artifacts": [
            {
                "role": "payload",
                "files": [
                    {
                        "name": name,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "size_in_bytes": len(data),
                    }
                    for name, data in package_bytes.items()
                ],
            }
        ],
    }
    metadata = {
        "info": {"name": "gwexpy", "version": version},
        "urls": [
            {
                "filename": name,
                "digests": {"sha256": hashlib.sha256(data).hexdigest()},
                "url": f"https://files.pythonhosted.org/packages/synthetic/{name}",
            }
            for name, data in package_bytes.items()
        ],
    }
    return manifest, metadata, package_bytes


def test_exact_manifest_wheel_and_sdist_read_back_successfully(pypi_case):
    validator = _load_validator()
    manifest, metadata, package_bytes = pypi_case

    validator.validate_pypi_readback(
        metadata=metadata,
        manifest=manifest,
        payload_files=package_bytes,
        downloaded_files=package_bytes,
        github_release_files=package_bytes,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_metadata_file",
        "extra_metadata_file",
        "duplicate_filename",
        "wrong_advertised_hash",
        "wrong_version",
        "wrong_filename_version",
        "invalid_url_scheme",
        "wrong_url_host",
        "wrong_url_filename",
    ],
)
def test_rejects_nonmatching_pypi_metadata(pypi_case, mutation):
    validator = _load_validator()
    manifest, metadata, package_bytes = pypi_case
    metadata = {
        **metadata,
        "info": dict(metadata["info"]),
        "urls": list(metadata["urls"]),
    }

    if mutation == "missing_metadata_file":
        metadata["urls"].pop()
    elif mutation == "extra_metadata_file":
        metadata["urls"].append(
            {
                "filename": "gwexpy-99.88.77-extra.tar.gz",
                "digests": {"sha256": "0" * 64},
                "url": "https://files.pythonhosted.org/packages/synthetic/extra.tar.gz",
            }
        )
    elif mutation == "duplicate_filename":
        metadata["urls"][1] = dict(metadata["urls"][0])
    elif mutation == "wrong_advertised_hash":
        metadata["urls"][0] = {
            **metadata["urls"][0],
            "digests": {"sha256": "0" * 64},
        }
    elif mutation == "wrong_version":
        metadata["info"]["version"] = "99.88.76"
    elif mutation == "wrong_filename_version":
        metadata["urls"][0] = {
            **metadata["urls"][0],
            "filename": "gwexpy-99.88.76-py3-none-any.whl",
        }
    elif mutation == "invalid_url_scheme":
        metadata["urls"][0] = {
            **metadata["urls"][0],
            "url": metadata["urls"][0]["url"].replace("https://", "http://"),
        }
    elif mutation == "wrong_url_host":
        metadata["urls"][0] = {
            **metadata["urls"][0],
            "url": metadata["urls"][0]["url"].replace(
                "files.pythonhosted.org", "example.invalid"
            ),
        }
    elif mutation == "wrong_url_filename":
        metadata["urls"][0] = {
            **metadata["urls"][0],
            "url": "https://files.pythonhosted.org/packages/synthetic/other.whl",
        }

    with pytest.raises(validator.PypiReadbackError):
        validator.validate_pypi_readback(
            metadata=metadata,
            manifest=manifest,
            payload_files=package_bytes,
            downloaded_files=package_bytes,
            github_release_files=package_bytes,
        )


@pytest.mark.parametrize("mismatch", ["missing", "extra", "wrong_bytes"])
def test_rejects_downloaded_file_set_or_bytes_mismatch(pypi_case, mismatch):
    validator = _load_validator()
    manifest, metadata, package_bytes = pypi_case
    downloaded_files = dict(package_bytes)
    if mismatch == "missing":
        downloaded_files.pop(next(iter(downloaded_files)))
    elif mismatch == "extra":
        downloaded_files["unexpected.whl"] = b"extra"
    else:
        name = next(iter(downloaded_files))
        downloaded_files[name] = b"bytes do not match the advertised digest"

    with pytest.raises(validator.PypiReadbackError):
        validator.validate_pypi_readback(
            metadata=metadata,
            manifest=manifest,
            payload_files=package_bytes,
            downloaded_files=downloaded_files,
            github_release_files=package_bytes,
        )


def test_rejects_release_asset_bytes_that_differ_from_pypi(pypi_case):
    validator = _load_validator()
    manifest, metadata, package_bytes = pypi_case
    release_files = dict(package_bytes)
    release_files[next(iter(release_files))] = b"different release asset bytes"

    with pytest.raises(validator.PypiReadbackError):
        validator.validate_pypi_readback(
            metadata=metadata,
            manifest=manifest,
            payload_files=package_bytes,
            downloaded_files=package_bytes,
            github_release_files=release_files,
        )


@pytest.mark.parametrize(
    "transient", ["not_found", "server_error", "transport", "truncated_response"]
)
def test_readback_retries_transient_responses_with_injected_http(pypi_case, transient):
    validator = _load_validator()
    manifest, metadata, package_bytes = pypi_case
    calls = []
    current_time = [0.0]

    def http_get(url, *, timeout):
        calls.append(url)
        if len(calls) == 1:
            if transient == "not_found":
                return validator.HttpResponse(status=404, url=url, body=b"")
            if transient == "server_error":
                return validator.HttpResponse(status=503, url=url, body=b"")
            if transient == "truncated_response":
                raise http.client.IncompleteRead(b"partial", 100)
            raise OSError("synthetic connection reset")
        if url.endswith("/json"):
            import json

            return validator.HttpResponse(
                status=200, url=url, body=json.dumps(metadata).encode()
            )
        name = url.rsplit("/", maxsplit=1)[-1]
        return validator.HttpResponse(status=200, url=url, body=package_bytes[name])

    def sleep(seconds):
        current_time[0] += seconds

    result = validator.read_pypi_with_retry(
        manifest=manifest,
        payload_files=package_bytes,
        github_release_files=package_bytes,
        http_get=http_get,
        retry_window_seconds=5,
        retry_interval_seconds=2,
        clock=lambda: current_time[0],
        sleep=sleep,
    )

    assert result == package_bytes
    assert len(calls) == 4
    assert current_time[0] == 2


@pytest.mark.parametrize("mismatch", ["version", "hash"])
def test_identity_or_hash_mismatch_fails_without_retry(pypi_case, mismatch):
    validator = _load_validator()
    manifest, metadata, package_bytes = pypi_case
    if mismatch == "version":
        metadata["info"]["version"] = "99.88.76"
    else:
        metadata["urls"][0]["digests"]["sha256"] = "0" * 64
    calls = []

    def http_get(url, *, timeout):
        calls.append(url)
        import json

        return validator.HttpResponse(
            status=200, url=url, body=json.dumps(metadata).encode()
        )

    expected_message = "version" if mismatch == "version" else "advertised SHA-256"
    with pytest.raises(validator.PypiReadbackError, match=expected_message):
        validator.read_pypi_with_retry(
            manifest=manifest,
            payload_files=package_bytes,
            github_release_files=package_bytes,
            http_get=http_get,
            retry_window_seconds=5,
            sleep=lambda _: pytest.fail("identity mismatch must not retry"),
        )
    assert len(calls) == 1


def test_corrupt_first_file_fails_before_requesting_next_file(pypi_case):
    validator = _load_validator()
    manifest, metadata, package_bytes = pypi_case
    file_requests = []

    def http_get(url, *, timeout):
        if url.endswith("/json"):
            import json

            return validator.HttpResponse(
                status=200, url=url, body=json.dumps(metadata).encode()
            )
        file_requests.append(url)
        if len(file_requests) == 1:
            return validator.HttpResponse(status=200, url=url, body=b"corrupt bytes")
        return validator.HttpResponse(status=503, url=url, body=b"")

    with pytest.raises(validator.PypiReadbackError, match="PyPI download"):
        validator.read_pypi_with_retry(
            manifest=manifest,
            payload_files=package_bytes,
            github_release_files=package_bytes,
            http_get=http_get,
            retry_window_seconds=5,
            sleep=lambda _: pytest.fail("file mismatch must fail without retry"),
        )

    assert len(file_requests) == 1


@pytest.mark.parametrize("expire_at", ["metadata", "last_file"])
def test_deadline_expiry_fails_before_more_requests_or_success(pypi_case, expire_at):
    validator = _load_validator()
    manifest, metadata, package_bytes = pypi_case
    current_time = [0.0]
    requests = []
    window = 5.0

    def http_get(url, *, timeout):
        started = current_time[0]
        requests.append((url, timeout, started))
        assert timeout <= max(0.0, window - started)
        if url.endswith("/json"):
            import json

            response = validator.HttpResponse(
                status=200, url=url, body=json.dumps(metadata).encode()
            )
            if expire_at == "metadata":
                current_time[0] = window + 1.0
            return response

        name = url.rsplit("/", maxsplit=1)[-1]
        if expire_at == "last_file" and len(requests) == 3:
            current_time[0] = window + 1.0
        return validator.HttpResponse(status=200, url=url, body=package_bytes[name])

    with pytest.raises(validator.PypiReadbackError, match="retry window"):
        validator.read_pypi_with_retry(
            manifest=manifest,
            payload_files=package_bytes,
            github_release_files=package_bytes,
            http_get=http_get,
            retry_window_seconds=window,
            clock=lambda: current_time[0],
            sleep=lambda _: pytest.fail("expired readback must not retry"),
        )

    assert len(requests) == (1 if expire_at == "metadata" else 3)


def test_http_body_read_stops_at_absolute_deadline(monkeypatch):
    validator = _load_validator()
    now = [0.0]
    socket_timeouts = []

    class FakeSocket:
        timeout = 0.0

        def settimeout(self, timeout):
            self.timeout = timeout
            socket_timeouts.append(timeout)

    fake_socket = FakeSocket()

    class FakeRaw:
        _sock = fake_socket

    class FakeFile:
        raw = FakeRaw()

    class FakeResponse:
        status = 200
        fp = FakeFile()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def geturl(self):
            return "https://pypi.org/pypi/gwexpy/99.88.77/json"

        def read1(self, size):
            if fake_socket.timeout < 0.4:
                raise TimeoutError("remaining deadline is shorter than next chunk")
            now[0] += 0.4
            return b"slow chunk"[:size]

    class FakeOpener:
        def open(self, request, *, timeout):
            assert request.full_url.endswith("/99.88.77/json")
            assert timeout <= 1.0
            fake_socket.settimeout(timeout)
            return FakeResponse()

    monkeypatch.setattr(
        validator.urllib.request, "build_opener", lambda *_: FakeOpener()
    )

    with pytest.raises(validator.RetryablePypiReadbackError):
        validator._http_get(
            "https://pypi.org/pypi/gwexpy/99.88.77/json",
            timeout=10.0,
            deadline=1.0,
            clock=lambda: now[0],
        )

    assert now[0] <= 1.0
    assert socket_timeouts == pytest.approx([1.0, 1.0, 0.6, 0.2])


@pytest.mark.parametrize("blocked_stage", ["headers", "body"])
def test_http_get_interrupts_blocking_io_at_global_deadline(monkeypatch, blocked_stage):
    import signal
    import time

    if not hasattr(signal, "setitimer"):
        pytest.skip("hard HTTP deadline test requires setitimer support")
    validator = _load_validator()
    url = "https://pypi.org/pypi/gwexpy/99.88.77/json"

    class FakeSocket:
        def settimeout(self, _timeout):
            return None

    class FakeRaw:
        _sock = FakeSocket()

    class FakeFile:
        raw = FakeRaw()

    class FakeResponse:
        status = 200
        fp = FakeFile()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def geturl(self):
            return url

        def read1(self, _size):
            time.sleep(0.6)
            return b"slow chunk"

    class FakeOpener:
        def open(self, _request, *, timeout):
            assert timeout <= 5.0
            if blocked_stage == "headers":
                time.sleep(0.6)
            return FakeResponse()

    monkeypatch.setattr(
        validator.urllib.request, "build_opener", lambda *_: FakeOpener()
    )
    started = time.monotonic()
    with pytest.raises(validator.PypiReadbackError, match="retry window"):
        validator._http_get(
            url,
            timeout=5.0,
            deadline=started + 0.05,
        )
    assert time.monotonic() - started < 0.4


def test_request_timeout_does_not_replace_global_readback_deadline(monkeypatch):
    validator = _load_validator()
    now = [0.0]
    url = "https://pypi.org/pypi/gwexpy/99.88.77/json"

    class FakeSocket:
        def settimeout(self, _timeout):
            return None

    class FakeRaw:
        _sock = FakeSocket()

    class FakeFile:
        raw = FakeRaw()

    class FakeResponse:
        status = 200
        fp = FakeFile()
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def geturl(self):
            return url

        def read1(self, _size):
            self.calls += 1
            if self.calls == 1:
                now[0] += 0.1
                return b"slow but valid"
            return b""

    class FakeOpener:
        def open(self, _request, *, timeout):
            assert timeout == pytest.approx(0.05)
            return FakeResponse()

    monkeypatch.setattr(
        validator.urllib.request, "build_opener", lambda *_: FakeOpener()
    )
    response = validator._http_get(
        url,
        timeout=0.05,
        deadline=5.0,
        clock=lambda: now[0],
    )
    assert response.body == b"slow but valid"
    assert now[0] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("wire_body", "expected_body"),
    [
        (
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
            b"5\r\nhello\r\n5\r\nworld\r\n0\r\n\r\n",
            b"helloworld",
        ),
        (
            b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nhelloworld",
            b"helloworld",
        ),
    ],
    ids=["chunked", "content-length"],
)
def test_http_body_reader_preserves_http_framing(wire_body, expected_body):
    import io

    validator = _load_validator()

    class FakeSocket:
        def __init__(self):
            self.closed = False

        def makefile(self, mode):
            assert mode == "rb"
            owner = self

            class FakeBuffered(io.BytesIO):
                def close(self):
                    super().close()
                    owner.closed = True

            stream = FakeBuffered(wire_body)
            stream.raw = type("FakeRaw", (), {"_sock": self})()
            return stream

        def settimeout(self, _timeout):
            if self.closed:
                raise OSError("fake socket is closed")

    response = http.client.HTTPResponse(FakeSocket())
    response.begin()

    body = validator._read_response_body(
        response,
        request_timeout=5.0,
        deadline=10.0,
        clock=lambda: 0.0,
    )
    assert body == expected_body


def test_http_get_disables_redirects_before_request(monkeypatch):
    import io

    validator = _load_validator()
    url = "https://pypi.org/pypi/gwexpy/99.88.77/json"
    opens = []

    class FakeOpener:
        def open(self, request, *, timeout):
            opens.append(request.full_url)
            raise validator.urllib.error.HTTPError(
                request.full_url, 302, "Found", {}, io.BytesIO(b"redirect")
            )

    def fake_build_opener(*handlers):
        redirect_handlers = [
            handler
            for handler in handlers
            if isinstance(handler, validator.urllib.request.HTTPRedirectHandler)
        ]
        assert len(redirect_handlers) == 1
        assert (
            redirect_handlers[0].redirect_request(
                validator.urllib.request.Request(url),
                None,
                302,
                "Found",
                {},
                "https://redirect.example/target",
            )
            is None
        )
        return FakeOpener()

    monkeypatch.setattr(validator.urllib.request, "build_opener", fake_build_opener)
    monkeypatch.setattr(
        validator.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("redirect-capable urlopen was used"),
    )

    response = validator._http_get(url, timeout=5.0)
    assert response.status == 302
    assert opens == [url]

    with pytest.raises(validator.PypiReadbackError, match="nonretryable HTTP 302"):
        validator._request(
            lambda *_args, **_kwargs: response, url, 5.0, api_version="99.88.77"
        )


@pytest.mark.parametrize("status", [302, 400, 429])
def test_does_not_retry_nontransient_http_status(pypi_case, status):
    validator = _load_validator()
    manifest, _, package_bytes = pypi_case
    calls = []

    def http_get(url, *, timeout):
        calls.append(url)
        return validator.HttpResponse(status=status, url=url, body=b"")

    with pytest.raises(validator.PypiReadbackError, match="nonretryable HTTP"):
        validator.read_pypi_with_retry(
            manifest=manifest,
            payload_files=package_bytes,
            github_release_files=package_bytes,
            http_get=http_get,
            retry_window_seconds=5,
            sleep=lambda _: pytest.fail("nontransient status must not retry"),
        )
    assert len(calls) == 1


def test_retry_window_is_capped_at_fifteen_minutes(pypi_case):
    validator = _load_validator()
    manifest, _, package_bytes = pypi_case
    current_time = [0.0]
    calls = []

    def http_get(url, *, timeout):
        calls.append((url, timeout))
        return validator.HttpResponse(status=503, url=url, body=b"")

    def sleep(seconds):
        current_time[0] += seconds

    with pytest.raises(validator.PypiReadbackError, match="retry window"):
        validator.read_pypi_with_retry(
            manifest=manifest,
            payload_files=package_bytes,
            github_release_files=package_bytes,
            http_get=http_get,
            retry_window_seconds=9999,
            retry_interval_seconds=300,
            clock=lambda: current_time[0],
            sleep=sleep,
        )

    assert current_time[0] <= 900
    assert calls[-1][1] <= 900
