"""Validate the exact files PyPI serves for a promoted candidate release."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import re
import signal
import stat
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

SCHEMA = "gwexpy-release-promotion-manifest-v1"
PACKAGE_NAME = "gwexpy"
RETRY_WINDOW_SECONDS = 900.0
RETRY_INTERVAL_SECONDS = 10.0
MAX_REQUEST_TIMEOUT_SECONDS = 30.0
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9.+!-]*)?$")


class PypiReadbackError(ValueError):
    """Raised when PyPI or its downloaded files differ from the candidate."""


class RetryablePypiReadbackError(PypiReadbackError):
    """Raised only for PyPI 404, 5xx, or transport failures."""


class _AbsoluteDeadlineInterrupt(Exception):
    """Internal signal used to interrupt blocking HTTP operations at the deadline."""


@dataclass(frozen=True)
class HttpResponse:
    """Small HTTP response value so request behavior can be injected in tests."""

    status: int
    url: str
    body: bytes


ExpectedFile = tuple[str, int]
HttpGet = Callable[..., HttpResponse]


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects so URLs are validated before any follow-up request."""

    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _expected_files(manifest: Any) -> tuple[str, dict[str, ExpectedFile]]:
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise PypiReadbackError("promotion manifest schema is missing or unsupported")
    version = manifest.get("version")
    if not isinstance(version, str) or not VERSION_RE.fullmatch(version):
        raise PypiReadbackError("promotion manifest version is malformed")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise PypiReadbackError("promotion manifest artifacts are malformed")
    payloads = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("role") == "payload"
    ]
    if len(payloads) != 1:
        raise PypiReadbackError("promotion manifest must identify one payload artifact")
    files = payloads[0].get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise PypiReadbackError(
            "promotion manifest payload must contain exactly two files"
        )

    expected: dict[str, ExpectedFile] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise PypiReadbackError("promotion manifest payload file is malformed")
        name = entry.get("name")
        digest = entry.get("sha256")
        size = entry.get("size_in_bytes")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or "\\" in name
            or name.startswith(".")
            or name in expected
        ):
            raise PypiReadbackError(
                "promotion manifest has an unsafe or duplicate filename"
            )
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise PypiReadbackError(f"promotion manifest SHA-256 is malformed: {name}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise PypiReadbackError(f"promotion manifest file size is invalid: {name}")
        expected[name] = (digest, size)

    sdist_name = f"{PACKAGE_NAME}-{version}.tar.gz"
    wheels = [
        name
        for name in expected
        if name.startswith(f"{PACKAGE_NAME}-{version}-") and name.endswith(".whl")
    ]
    if sdist_name not in expected or len(wheels) != 1 or len(expected) != 2:
        raise PypiReadbackError(
            "promotion manifest must contain the version's one wheel and one sdist"
        )
    return version, expected


def _validate_package_bytes(
    files: Mapping[str, bytes],
    expected: Mapping[str, ExpectedFile],
    label: str,
) -> None:
    if not isinstance(files, Mapping) or set(files) != set(expected):
        raise PypiReadbackError(
            f"{label} file set is not exactly the manifest wheel and sdist"
        )
    for name, (expected_sha256, expected_size) in expected.items():
        data = files[name]
        if not isinstance(data, bytes):
            raise PypiReadbackError(f"{label} file bytes are malformed: {name}")
        if len(data) != expected_size:
            raise PypiReadbackError(
                f"{label} file size differs from the manifest: {name}"
            )
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_sha256:
            raise PypiReadbackError(
                f"{label} file SHA-256 differs from the manifest: {name}"
            )


def _validate_file_url(url: Any, filename: str) -> str:
    if not isinstance(url, str):
        raise PypiReadbackError(f"PyPI file URL is missing: {filename}")
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise PypiReadbackError(f"PyPI file URL is malformed: {filename}") from exc
    path_parts = [urllib.parse.unquote(part) for part in parsed.path.split("/")]
    if (
        parsed.scheme != "https"
        or parsed.hostname != "files.pythonhosted.org"
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/packages/")
        or any(part in (".", "..") for part in path_parts)
        or not path_parts
        or path_parts[-1] != filename
    ):
        raise PypiReadbackError(f"PyPI file URL is invalid: {filename}")
    return url


def _metadata_file_urls(
    metadata: Any, version: str, expected: Mapping[str, ExpectedFile]
) -> dict[str, str]:
    if not isinstance(metadata, dict):
        raise PypiReadbackError("PyPI JSON response is not an object")
    info = metadata.get("info")
    if not isinstance(info, dict):
        raise PypiReadbackError("PyPI JSON response has no package identity")
    if info.get("name") != PACKAGE_NAME or info.get("version") != version:
        raise PypiReadbackError(
            "PyPI package name or version differs from the manifest"
        )

    urls = metadata.get("urls")
    if not isinstance(urls, list) or len(urls) != 2:
        raise PypiReadbackError(
            "PyPI must advertise exactly the manifest wheel and sdist"
        )
    found: dict[str, str] = {}
    for entry in urls:
        if not isinstance(entry, dict):
            raise PypiReadbackError("PyPI file metadata entry is malformed")
        filename = entry.get("filename")
        if not isinstance(filename, str) or filename in found:
            raise PypiReadbackError(
                "PyPI file metadata has a missing or duplicate filename"
            )
        if filename not in expected:
            raise PypiReadbackError(f"PyPI advertises an unexpected file: {filename}")
        digests = entry.get("digests")
        advertised = digests.get("sha256") if isinstance(digests, dict) else None
        if not isinstance(advertised, str) or not SHA256_RE.fullmatch(advertised):
            raise PypiReadbackError(f"PyPI SHA-256 is malformed: {filename}")
        if advertised != expected[filename][0]:
            raise PypiReadbackError(
                f"PyPI advertised SHA-256 differs from the manifest: {filename}"
            )
        found[filename] = _validate_file_url(entry.get("url"), filename)

    if set(found) != set(expected):
        raise PypiReadbackError("PyPI file list is missing a manifest distribution")
    return found


def validate_pypi_readback(
    *,
    metadata: Any,
    manifest: Any,
    payload_files: Mapping[str, bytes],
    downloaded_files: Mapping[str, bytes],
    github_release_files: Mapping[str, bytes],
) -> None:
    """Require PyPI, manifest, candidate payload, and Release bytes to match exactly."""
    version, expected = _expected_files(manifest)
    _metadata_file_urls(metadata, version, expected)

    _validate_package_bytes(payload_files, expected, "candidate payload")
    _validate_package_bytes(downloaded_files, expected, "PyPI download")
    _validate_package_bytes(github_release_files, expected, "GitHub Release")
    for name in expected:
        if downloaded_files[name] != payload_files[name]:
            raise PypiReadbackError(
                f"PyPI bytes differ from the candidate payload: {name}"
            )
        if github_release_files[name] != payload_files[name]:
            raise PypiReadbackError(
                f"GitHub Release bytes differ from the candidate payload: {name}"
            )


def _validate_api_url(url: str, version: str) -> None:
    expected = f"https://pypi.org/pypi/{PACKAGE_NAME}/{version}/json"
    if url != expected:
        raise PypiReadbackError("PyPI JSON API URL is not canonical")


def _request(
    http_get: HttpGet, url: str, timeout: float, *, api_version: Optional[str] = None
) -> HttpResponse:
    try:
        response = http_get(url, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = HttpResponse(status=exc.code, url=exc.geturl() or url, body=b"")
    except RetryablePypiReadbackError:
        raise
    except (
        urllib.error.URLError,
        OSError,
        TimeoutError,
        http.client.HTTPException,
    ) as exc:
        raise RetryablePypiReadbackError(f"PyPI transport error: {exc}") from exc

    if not isinstance(response, HttpResponse):
        raise PypiReadbackError("injected HTTP client returned an invalid response")
    if not isinstance(response.status, int) or isinstance(response.status, bool):
        raise PypiReadbackError("HTTP response status is malformed")
    if api_version is not None:
        _validate_api_url(url, api_version)
        if response.url != url:
            raise PypiReadbackError(
                "PyPI JSON API response came from an unexpected URL"
            )
    else:
        filename = Path(urllib.parse.urlsplit(url).path).name
        _validate_file_url(url, filename)
        _validate_file_url(response.url, filename)
        if response.url != url:
            raise PypiReadbackError(
                "PyPI file response was redirected to a different URL"
            )
    if response.status == 404 or 500 <= response.status <= 599:
        raise RetryablePypiReadbackError(
            f"PyPI returned transient HTTP {response.status}"
        )
    if response.status != 200:
        raise PypiReadbackError(f"PyPI returned nonretryable HTTP {response.status}")
    if not isinstance(response.body, bytes):
        raise PypiReadbackError("HTTP response body is not bytes")
    return response


def _read_response_body(
    response: Any,
    *,
    request_timeout: float,
    deadline: float,
    clock: Callable[[], float],
) -> bytes:
    stream = getattr(response, "fp", None)
    raw_stream = getattr(stream, "raw", None)
    response_socket = getattr(raw_stream, "_sock", None)
    read_chunk = getattr(response, "read1", None)
    if response_socket is None or not callable(read_chunk):
        raise PypiReadbackError("PyPI HTTP response cannot enforce its read deadline")

    chunks: list[bytes] = []
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise PypiReadbackError(
                "PyPI readback retry window expired during download"
            )
        response_socket.settimeout(min(request_timeout, remaining))
        chunk = read_chunk(64 * 1024)
        if clock() >= deadline:
            raise PypiReadbackError(
                "PyPI readback retry window expired during download"
            )
        if not chunk:
            return b"".join(chunks)
        if not isinstance(chunk, bytes):
            raise PypiReadbackError("PyPI HTTP response returned a non-bytes chunk")
        chunks.append(chunk)
        if getattr(response, "fp", None) is None:
            return b"".join(chunks)


def _call_with_absolute_deadline(
    operation: Callable[[], HttpResponse],
    *,
    deadline: float,
    clock: Callable[[], float],
) -> HttpResponse:
    remaining = deadline - clock()
    if remaining <= 0:
        raise PypiReadbackError("PyPI readback retry window expired before request")
    if (
        not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        raise PypiReadbackError(
            "cannot enforce the PyPI readback deadline in this runtime"
        )
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        raise PypiReadbackError(
            "cannot enforce the PyPI readback deadline while another alarm is active"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)

    def interrupt(_signum: int, _frame: Any) -> None:
        raise _AbsoluteDeadlineInterrupt

    try:
        signal.signal(signal.SIGALRM, interrupt)
        signal.setitimer(signal.ITIMER_REAL, remaining)
    except (OSError, ValueError) as exc:
        signal.signal(signal.SIGALRM, previous_handler)
        raise PypiReadbackError(
            "cannot arm the PyPI readback absolute deadline"
        ) from exc

    try:
        try:
            response = operation()
        except _AbsoluteDeadlineInterrupt as exc:
            raise PypiReadbackError(
                "PyPI readback retry window expired during HTTP request"
            ) from exc
        if clock() >= deadline:
            raise PypiReadbackError(
                "PyPI readback retry window expired during HTTP request"
            )
        return response
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _http_get(
    url: str,
    *,
    timeout: float,
    deadline: Optional[float] = None,
    clock: Callable[[], float] = time.monotonic,
) -> HttpResponse:
    started = clock()
    absolute_deadline = (
        deadline if deadline is not None else started + RETRY_WINDOW_SECONDS
    )
    remaining = absolute_deadline - started
    if remaining <= 0:
        raise PypiReadbackError("PyPI readback retry window expired before request")
    request_timeout = min(MAX_REQUEST_TIMEOUT_SECONDS, timeout, remaining)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json, application/octet-stream",
            "User-Agent": "gwexpy-release-readback",
        },
    )

    def operation() -> HttpResponse:
        opener = urllib.request.build_opener(_NoRedirectHandler())
        try:
            with opener.open(request, timeout=request_timeout) as response:
                return HttpResponse(
                    status=response.status,
                    url=response.geturl(),
                    body=_read_response_body(
                        response,
                        request_timeout=request_timeout,
                        deadline=absolute_deadline,
                        clock=clock,
                    ),
                )
        except urllib.error.HTTPError as exc:
            response_url = exc.geturl() or url
            exc.close()
            return HttpResponse(status=exc.code, url=response_url, body=b"")

    try:
        return _call_with_absolute_deadline(
            operation, deadline=absolute_deadline, clock=clock
        )
    except (
        urllib.error.URLError,
        OSError,
        TimeoutError,
        http.client.HTTPException,
    ) as exc:
        raise RetryablePypiReadbackError(f"PyPI transport error: {exc}") from exc


def _decode_metadata(body: bytes) -> Any:
    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PypiReadbackError(f"PyPI JSON contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(body.decode("utf-8"), object_pairs_hook=object_from_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PypiReadbackError(f"PyPI JSON response is malformed: {exc}") from exc


def read_pypi_with_retry(
    *,
    manifest: Any,
    payload_files: Mapping[str, bytes],
    github_release_files: Mapping[str, bytes],
    http_get: Optional[HttpGet] = None,
    retry_window_seconds: float = RETRY_WINDOW_SECONDS,
    retry_interval_seconds: float = RETRY_INTERVAL_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, bytes]:
    """Fetch and validate PyPI readback, retrying only transient HTTP failures."""
    version, expected = _expected_files(manifest)
    _validate_package_bytes(payload_files, expected, "candidate payload")
    _validate_package_bytes(github_release_files, expected, "GitHub Release")
    if any(payload_files[name] != github_release_files[name] for name in expected):
        raise PypiReadbackError(
            "GitHub Release bytes differ from the candidate payload"
        )
    if (
        not isinstance(retry_window_seconds, (int, float))
        or isinstance(retry_window_seconds, bool)
        or not math.isfinite(retry_window_seconds)
        or retry_window_seconds <= 0
    ):
        raise PypiReadbackError("retry window must be a positive finite number")
    if (
        not isinstance(retry_interval_seconds, (int, float))
        or isinstance(retry_interval_seconds, bool)
        or not math.isfinite(retry_interval_seconds)
        or retry_interval_seconds <= 0
    ):
        raise PypiReadbackError("retry interval must be a positive finite number")

    window = min(float(retry_window_seconds), RETRY_WINDOW_SECONDS)
    deadline = clock() + window
    api_url = f"https://pypi.org/pypi/{PACKAGE_NAME}/{version}/json"
    attempted = False

    def production_get(url: str, *, timeout: float) -> HttpResponse:
        return _http_get(url, timeout=timeout, deadline=deadline, clock=clock)

    getter = http_get or production_get

    def request_before_deadline(
        url: str, *, api_version: Optional[str] = None
    ) -> HttpResponse:
        remaining = deadline - clock()
        if remaining <= 0:
            raise PypiReadbackError("PyPI readback retry window expired")
        response = _request(
            getter,
            url,
            min(MAX_REQUEST_TIMEOUT_SECONDS, remaining),
            api_version=api_version,
        )
        if clock() >= deadline:
            raise PypiReadbackError("PyPI readback retry window expired")
        return response

    while True:
        now = clock()
        if attempted and now >= deadline:
            raise PypiReadbackError("PyPI readback retry window expired")
        try:
            metadata_response = request_before_deadline(api_url, api_version=version)
            metadata = _decode_metadata(metadata_response.body)
            file_urls = _metadata_file_urls(metadata, version, expected)
            downloads: dict[str, bytes] = {}
            for name, url in file_urls.items():
                response = request_before_deadline(url)
                data = response.body
                expected_sha256, expected_size = expected[name]
                if len(data) != expected_size:
                    raise PypiReadbackError(
                        f"PyPI download file size differs from the manifest: {name}"
                    )
                if hashlib.sha256(data).hexdigest() != expected_sha256:
                    raise PypiReadbackError(
                        f"PyPI download SHA-256 differs from the manifest: {name}"
                    )
                if data != payload_files[name]:
                    raise PypiReadbackError(
                        f"PyPI download bytes differ from the candidate payload: {name}"
                    )
                downloads[name] = data
            validate_pypi_readback(
                metadata=metadata,
                manifest=manifest,
                payload_files=payload_files,
                downloaded_files=downloads,
                github_release_files=github_release_files,
            )
            if clock() >= deadline:
                raise PypiReadbackError("PyPI readback retry window expired")
            return downloads
        except RetryablePypiReadbackError as exc:
            remaining = deadline - clock()
            if remaining <= 0:
                raise PypiReadbackError("PyPI readback retry window expired") from exc
            sleep(min(float(retry_interval_seconds), remaining))
            attempted = True


def _read_regular_files(directory: Path, label: str) -> dict[str, bytes]:
    if not directory.is_dir() or directory.is_symlink():
        raise PypiReadbackError(f"{label} path is not a regular directory")
    result: dict[str, bytes] = {}
    for path in directory.iterdir():
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise PypiReadbackError(
                f"cannot inspect {label} entry: {path.name}"
            ) from exc
        if not stat.S_ISREG(mode) or path.name in result:
            raise PypiReadbackError(f"{label} directory contains an unsafe entry")
        try:
            result[path.name] = path.read_bytes()
        except OSError as exc:
            raise PypiReadbackError(f"cannot read {label} entry: {path.name}") from exc
    return result


def _read_manifest(path: Path) -> Any:
    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PypiReadbackError(f"manifest contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=object_from_pairs
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PypiReadbackError(f"cannot read promotion manifest: {exc}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--payload-dir", type=Path, required=True)
    parser.add_argument("--github-release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        manifest = _read_manifest(args.manifest)
        _, expected = _expected_files(manifest)
        payload_files = _read_regular_files(args.payload_dir, "candidate payload")
        release_entries = _read_regular_files(args.github_release_dir, "GitHub Release")
        package_files = {
            name: data
            for name, data in release_entries.items()
            if name.endswith((".whl", ".tar.gz"))
        }
        if set(package_files) != set(expected):
            raise PypiReadbackError(
                "GitHub Release must contain exactly the manifest wheel and sdist"
            )
        downloads = read_pypi_with_retry(
            manifest=manifest,
            payload_files=payload_files,
            github_release_files=package_files,
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, data in downloads.items():
            (args.output_dir / name).write_bytes(data)
    except (OSError, PypiReadbackError) as exc:
        print(f"PyPI readback rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
