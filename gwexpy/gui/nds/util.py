"""
Utility functions for NDS connectivity.
"""

import logging
import time

logger = logging.getLogger(__name__)


def parse_server_string(server: str | None) -> tuple[str, int]:
    """
    Parse 'host:port' string into (host, port).
    Handles comma-separated lists (takes first).
    Default port is 31200 if not specified.
    """
    if not server:
        return "localhost", 31200

    # Take the first server if it's a list
    if "," in server:
        server = server.split(",")[0]

    if ":" in server:
        parts = server.split(":")
        if len(parts) == 2:
            try:
                return parts[0], int(parts[1])
            except (TypeError, ValueError):
                logger.warning("Invalid port in server string: %s", server)
                return parts[0], 31200
        else:
            # Handle cases like ipv6 or weird formats?
            # For now assume host:port is the last colon if multiple?
            # Or just take first two parts.
            try:
                return parts[0], int(parts[1])
            except (TypeError, ValueError):
                logger.warning("Invalid port in server string: %s", server)
                return parts[0], 31200
    else:
        return server, 31200


def gps_now() -> float:
    """
    Return current GPS time.
    Falls back to system time converted to GPS epoch if gpstime is unavailable.
    GPS epoch is 1980-01-06 00:00:00 UTC.
    """
    try:
        from gpstime import gpsnow

        return float(gpsnow())
    except ImportError as exc:
        logger.info("gpstime unavailable, using system time fallback: %s", exc)
        # Fallback: approximated GPS time using unix time.
        return time.time() - 315964800 + 18
