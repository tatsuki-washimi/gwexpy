from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HELPER = Path('/home/washimi/work/gwexpy-v025-b-f4-gwf/benchmarks/io/f4_gwf_fixtures.py')
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--case',required=True)
    parser.add_argument('--route',choices=['serial','parallel'],required=True)
    args=parser.parse_args()
    assert sha(HELPER)=='e01018185bae922cb582088bc30f62457d551b2951319dc4a4f37dbfc48c3dc7'
    spec=importlib.util.spec_from_file_location('f4_gwf_fixtures',HELPER)
    assert spec and spec.loader
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest=json.loads((args.root/'manifest.json').read_text())
    assert manifest['schema']=='gwexpy-v025-b-f4-fixtures-v1'
    assert args.case in module._READ_CASES
    names,channels,read_kwargs=module._READ_CASES[args.case]
    file_hashes={item['name']:item['sha256'] for item in manifest['files']}
    for name in names:
        assert sha(args.root/name)==file_hashes[name]
    capture=module._capture_public_read([args.root/name for name in names],channels=channels,parallel=False if args.route=='serial' else 2,**read_kwargs)
    print(json.dumps({'case':args.case,'route':args.route,'fixture_manifest_sha256':sha(args.root/'manifest.json'),'helper_sha256':sha(HELPER),'harness_sha256':sha(Path(__file__)),'capture':capture},indent=2,sort_keys=True))

if __name__=='__main__':
    main()
