"""
Utility functions for NDS connectivity.
"""
import time
import os

try:
    from gpstime import gpsnow
except ImportError:
    gpsnow = None

def parse_server_string(server):
    """
    Parse 'host:port' string into (host, port).
    Handles comma-separated lists (takes first).
    Default port is 31200 if not specified.
    """
    if not server:
        return 'localhost', 31200
        
    # Take the first server if it's a list
    if ',' in server:
        server = server.split(',')[0]
        
    if ':' in server:
        parts = server.split(':')
        if len(parts) == 2:
            return parts[0], int(parts[1])
        else:
            # Handle cases like ipv6 or weird formats? 
            # For now assume host:port is the last colon if multiple?
            # Or just take first two parts.
            return parts[0], int(parts[1])
    else:
        return server, 31200

def gps_now():
    """
    Return current GPS time.
    Falls back to system time converted to GPS epoch if gpstime is unavailable.
    GPS epoch is 1980-01-06 00:00:00 UTC.
    """
    if gpsnow:
        return float(gpsnow())
    else:
        # Fallback: approximated GPS time using unix time
        # GPS time = Unix time - 315964800 + leap seconds (approx 18s currently)
        # This is a rough fallback. Accurate conversion requires leap second table.
        # For display purposes roughly aligning with now, this might suffice if simple.
        # But better to rely on t0 from NDS data.
        # Here we just use a simple offset.
        # Unix 0 = 1970-01-01
        # GPS 0 = 1980-01-06
        # Diff = 315964800 seconds
        return time.time() - 315964800 + 18 
