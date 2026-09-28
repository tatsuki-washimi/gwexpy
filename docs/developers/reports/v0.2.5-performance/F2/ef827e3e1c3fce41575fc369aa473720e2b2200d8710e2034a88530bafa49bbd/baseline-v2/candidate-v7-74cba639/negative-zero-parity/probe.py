import hashlib
import json
import sqlite3
import warnings
from pathlib import Path

import numpy as np
from gwexpy.timeseries.io.sdb import read_timeseriesdict_sdb

path = Path('/tmp/sdb-negzero-probe/mixed-negzero.sdb')
path.unlink(missing_ok=True)
with sqlite3.connect(path) as connection:
    connection.execute('CREATE TABLE archive (dateTime INTEGER, outHumidity)')
    connection.executemany(
        'INSERT INTO archive VALUES (?, ?)',
        [(1_700_000_000, '-0'), (1_700_000_300, 1.5), (1_700_000_600, '2')],
    )
full = read_timeseriesdict_sdb(path, columns=['outHumidity'])['outHumidity']
start = float(full.times[0].value)
end = float(full.times[1].value)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    result = read_timeseriesdict_sdb(
        path, columns=['outHumidity'], start=start, end=end
    )['outHumidity']
values = np.asarray(result.value, dtype=np.float64)
print(json.dumps({
    'path': str(path),
    'source_file': __import__('gwexpy.timeseries.io.sdb', fromlist=['x']).__file__,
    'start': start,
    'end': end,
    'length': len(result),
    'dtype': str(result.dtype),
    'values': values.tolist(),
    'float64_bytes_hex': values.tobytes().hex(),
    'signbit': np.signbit(values).tolist(),
    'warnings': [str(item.message) for item in caught],
    'full_bytes_hex': np.asarray(full.value, dtype=np.float64).tobytes().hex(),
    'full_signbit': np.signbit(np.asarray(full.value, dtype=np.float64)).tolist(),
}, sort_keys=True))
