try:
    from gwpy.testing.errors import (
        NETWORK_ERROR,
        SSLError,
        URLError,
        pytest,
        pytest_skip_cvmfs_read_error,
        pytest_skip_network_error,
        requests,
        socket,
        wraps,
    )
except ImportError:
    # Handle gwpy version differences
    import socket
    import ssl
    from functools import wraps
    import requests
    import pytest
    from urllib.error import URLError
    NETWORK_ERROR = (socket.timeout, requests.RequestException, URLError)
    SSLError = ssl.SSLError
    def pytest_skip_network_error(func):
        return pytest.mark.skipif(True, reason="Network tests skipped")(func)
    def pytest_skip_cvmfs_read_error(func):
        return pytest.mark.skipif(True, reason="CVMFS tests skipped")(func)

__all__ = [
    "NETWORK_ERROR",
    "SSLError",
    "URLError",
    "pytest",
    "pytest_skip_cvmfs_read_error",
    "pytest_skip_network_error",
    "requests",
    "socket",
    "wraps",
]
