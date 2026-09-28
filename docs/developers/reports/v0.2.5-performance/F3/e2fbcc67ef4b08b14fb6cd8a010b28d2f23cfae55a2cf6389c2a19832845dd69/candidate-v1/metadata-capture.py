"""Wheel-only public F3 PSD metadata fault capture using frozen harness bytes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

FROZEN = Path('/tmp/gwexpy-v025-f3-frozen-harness/benchmarks/io')
sys.path.insert(0, str(FROZEN))
import f3_dttxml_harness as harness
import run


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    manifest=json.loads(Path(args.manifest).read_text())
    cases={}
    for name,case in manifest['cases'].items():
        with np.errstate(under=case.get('underflow_mode','ignore')):
            cases[name]=harness.capture_public_case(case,'native',instrument=False)
    output={'schema':'gwexpy-f3-psd-metadata-capture-v1',
            'manifest_sha256':sha(args.manifest),
            'frozen_harness_digest':run.harness_digest(),
            'capture_source_sha256':sha(__file__),
            'cases':cases}
    with Path(args.output).open('x') as stream:
        json.dump(output,stream,sort_keys=True,indent=2)
        stream.write('\n')

if __name__=='__main__':
    main()
