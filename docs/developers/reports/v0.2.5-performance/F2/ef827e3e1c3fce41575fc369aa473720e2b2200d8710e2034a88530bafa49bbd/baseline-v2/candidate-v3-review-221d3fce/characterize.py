from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from gwexpy.io.time_selection import apply_time_selection
from gwexpy.timeseries.io.sdb import read_timeseriesdict_sdb

source = Path(sys.argv[1])
output = Path(sys.argv[2])
original = pd.read_sql_query

def snapshot(call):
    sql = []
    def record(query, connection, *args, **kwargs):
        frame = original(query, connection, *args, **kwargs)
        sql.append({'query': query, 'rows': len(frame)})
        return frame
    pd.read_sql_query = record
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            series = call()
        except Exception as error:
            result = {'outcome':'error', 'error_type':type(error).__name__, 'error':str(error)}
        else:
            result = {
                'outcome':'return',
                'length':len(series),
                't0':float(series.t0.value),
                'dt':float(series.dt.value),
                'values':np.asarray(series.value).tolist(),
                'times':np.asarray(series.times.value).tolist(),
            }
    pd.read_sql_query = original
    result['warnings'] = [{'category':f'{x.category.__module__}.{x.category.__name__}', 'message':str(x.message)} for x in caught]
    result['sql'] = sql
    return result

full = read_timeseriesdict_sdb(source, columns=['outTemp'])
series = full['outTemp']
assert not hasattr(series, '_xindex'), 'full-read oracle was not fresh'
t0 = float(series.t0.value)
dt = float(series.dt.value)
start = t0 + 20*dt
end = t0 + 24*dt
specs = {
    'half_sample': (start + dt/2, end + dt/2),
    'start_ulp_below': (np.nextafter(start, -np.inf), end),
    'start_ulp_above': (np.nextafter(start, np.inf), end),
    'end_ulp_below': (start, np.nextafter(end, -np.inf)),
    'end_ulp_above': (start, np.nextafter(end, np.inf)),
    'empty_inside': (start+dt/2, start+dt/2),
    'empty_post_source': (t0+65*dt, t0+66*dt),
    'nan_start': (float('nan'), end),
    'posinf_start': (float('inf'), end),
    'neginf_start': (float('-inf'), end),
    'nan_end': (start, float('nan')),
    'posinf_end': (start, float('inf')),
    'neginf_end': (start, float('-inf')),
}
result = {'fresh_oracle': {'t0':t0,'dt':dt,'length':len(series),'has_xindex':hasattr(series,'_xindex')}, 'cases':{}}
for name, (lo, hi) in specs.items():
    result['cases'][name] = {
        'full_crop': snapshot(lambda lo=lo,hi=hi: apply_time_selection(full,lo,hi)['outTemp']),
        'bounded_read': snapshot(lambda lo=lo,hi=hi: read_timeseriesdict_sdb(source,columns=['outTemp'],start=lo,end=hi)['outTemp']),
    }
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
