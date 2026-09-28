import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/washimi/work/gwexpy-v025-b-f4-gwf/benchmarks/io')
import f4_gwf_fixtures as frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fixture', type=Path)
    parser.add_argument('case')
    parser.add_argument('route', choices=('serial', 'parallel'))
    args = parser.parse_args()
    names, channels, read_kwargs = frozen._READ_CASES[args.case]
    expected = {item['name']: item['sha256'] for item in json.loads((args.fixture/'manifest.json').read_text())['files']}
    for name in names:
        assert frozen._sha256(args.fixture/name) == expected[name]
    capture = frozen._capture_public_read([args.fixture/name for name in names], channels=channels, parallel=False if args.route == 'serial' else 2, **read_kwargs)
    print(json.dumps(capture, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
