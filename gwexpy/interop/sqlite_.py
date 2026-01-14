import json

import numpy as np


def _ensure_schema(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS series (
        series_id TEXT PRIMARY KEY,
        channel TEXT,
        unit TEXT,
        t0 REAL,
        dt REAL,
        n INTEGER,
        attrs_json TEXT
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS samples (
        series_id TEXT,
        i INTEGER,
        value REAL,
        PRIMARY KEY(series_id, i)
    )
    """)


def to_sqlite(ts, conn, series_id=None, overwrite=False):
    _ensure_schema(conn)

    sid = series_id or (ts.name if ts.name else "unknown")

    # Check exist
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM series WHERE series_id=?", (sid,))
    if cur.fetchone():
        if overwrite:
            cur.execute("DELETE FROM series WHERE series_id=?", (sid,))
            cur.execute("DELETE FROM samples WHERE series_id=?", (sid,))
        else:
            raise ValueError(f"Series ID {sid} exists")

    # Insert meta
    attrs = {"name": str(ts.name)}
    cur.execute(
        "INSERT INTO series (series_id, channel, unit, t0, dt, n, attrs_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            sid,
            str(ts.channel) if ts.channel else "",
            str(ts.unit),
            ts.t0.value,
            ts.dt.value,
            len(ts),
            json.dumps(attrs),
        ),
    )

    # Insert data (bulk)
    # Using executemany with generator
    data = ts.value
    params = ((sid, i, float(v)) for i, v in enumerate(data))
    cur.executemany(
        "INSERT INTO samples (series_id, i, value) VALUES (?, ?, ?)", params
    )

    return sid


def from_sqlite(cls, conn, series_id):
    cur = conn.cursor()
    cur.execute("SELECT t0, dt, unit, n FROM series WHERE series_id=?", (series_id,))
    row = cur.fetchone()
    if not row:
        raise KeyError(f"Series {series_id} not found")

    t0, dt, unit, n = row

    # Read samples
    cur.execute("SELECT value FROM samples WHERE series_id=? ORDER BY i", (series_id,))
    data_rows = cur.fetchall()

    data = np.array([r[0] for r in data_rows])

    return cls(data, t0=t0, dt=dt, unit=unit, name=series_id)
