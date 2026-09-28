from __future__ import annotations
import argparse
import gc
import hashlib
import json
import weakref
from pathlib import Path
import numpy as np
import gwexpy.timeseries._gwf_io as io
from gwexpy.timeseries import TimeSeriesDict

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(root: Path, count: int) -> dict:
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['schema'] == 'gwexpy-v025-b-f4-stress-fixtures-v1'
    paths = [root/name for name in manifest['source_order'][:count]]
    for path in paths:
        expected = next(item['sha256'] for item in manifest['files'] if item['name']==path.name)
        assert sha(path)==expected
    original = io._coerce_gwf_timeseriesdict
    live = 0
    peak = 0
    created = 0
    def counted(*args, **kwargs):
        nonlocal live, peak, created
        result = original(*args, **kwargs)
        live += 1
        created += 1
        peak = max(peak, live)
        def release():
            nonlocal live
            live -= 1
        weakref.finalize(result, release)
        return result
    io._coerce_gwf_timeseriesdict = counted
    try:
        result = TimeSeriesDict.read(paths, [manifest['channel']], format='gwf', parallel=False)
    finally:
        io._coerce_gwf_timeseriesdict = original
    values = np.ascontiguousarray(result[manifest['channel']].value)
    return {'count':count,'peak_live_coerced_part_objects':peak,'created_coerced_dict_objects':created,'live_after_read':live,'shape':list(values.shape),'dtype':values.dtype.str,'values_sha256':hashlib.sha256(values.tobytes()).hexdigest(),'fixture_manifest_sha256':sha(root/'manifest.json'),'harness_sha256':sha(Path(__file__))}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--count',type=int,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.root,args.count),indent=2,sort_keys=True))
