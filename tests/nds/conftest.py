import os
import socket

import pytest


def resolve_nds_endpoint():
    """
    Resolve NDS endpoint from environment variables.
    Returns (host, port)
    """
    host = os.getenv("GWEXPY_NDS_HOST")
    port = os.getenv("GWEXPY_NDS_PORT")

    if host and port:
        return host, int(port)

    # Try NDSSERVER (port 8088/NDS1) first as it is more likely to be the online source at KAGRA
    nds1_server = os.getenv("NDSSERVER")
    if nds1_server:
        for entry in nds1_server.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" in entry:
                h, p = entry.split(":", 1)
                return h, int(p)
            else:
                return entry, 8088

    # Try NDS2SERVER: "host1:port1,host2:port2"
    nds2_server = os.getenv("NDS2SERVER")
    if nds2_server:
        for entry in nds2_server.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" in entry:
                h, p = entry.split(":", 1)
                return h, int(p)
            else:
                return entry, 31200

    # Fallback
    return "nds.ligo.caltech.edu", 31200


@pytest.fixture(scope="session")
def nds_available():
    """
    Check if NDS test should run.
    Returns (bool, reason)
    """
    if os.getenv("GWEXPY_ENABLE_NDS_TESTS") != "1":
        return False, "GWEXPY_ENABLE_NDS_TESTS is not set to 1"

    try:
        import nds2
    except ImportError:
        return False, "nds2-client is not installed"

    host, port = resolve_nds_endpoint()

    try:
        # Set explicit timeout to avoid long hangs on network issues
        socket.setdefaulttimeout(10)
        conn = nds2.connection(host, port)
        conn.close()
    except Exception as e:
        # Print full details for debugging (-s flag), but keep skip reason short
        print(f"[NDS detail] {host}:{port} -> {e}")
        return False, f"Failed to connect to NDS server {host}:{port}"
    finally:
        socket.setdefaulttimeout(None)

    return True, ""


@pytest.fixture
def nds_backend(nds_available):
    """
    Fixture to skip tests if NDS is not available.
    """
    available, reason = nds_available
    if not available:
        pytest.skip(f"NDS not available: {reason}")

    host, port = resolve_nds_endpoint()
    return {
        "host": host,
        "port": port,
    }
