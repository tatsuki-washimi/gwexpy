from pathlib import Path
import json
import subprocess
import sys

mode = sys.argv[1]
assert mode in ('wall', 'pss')
root = Path('/tmp/gwexpy-v025-bx-formal-7f13-v1')
root.mkdir(exist_ok=True)
harness = Path('/home/washimi/work/gwexpy-v025-performance-copy-formal/benchmarks/copy_audit/bx_measure.py')
fixtures = Path('/tmp/gwexpy-v025-bx-fixtures-v1')
prex = '7f13c4423addab4f92eafc2740da05faa4a444fa'
arms = {
    'B0': ('/tmp/gwexpy-v025-b-env-a/bin/python', '/tmp/gwexpy-v025-b-wheels/gwexpy-0.2.4-py3-none-any.whl', '522e52a082925da4dd37966d82a7616bdd2a5248'),
    'B1': ('/tmp/gwexpy-v025-b-env-b/bin/python', '/tmp/gwexpy-v025-b-wheels/gwexpy-0.2.5-py3-none-any.whl', '1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c'),
    'preX': ('/tmp/gwexpy-v025-bx-env-7f13/bin/python', '/tmp/gwexpy-v025-bx-prex-7f13-wheel/gwexpy-0.2.5-py3-none-any.whl', prex),
}
cases = [(name, 'warn') for name in ('ats32', 'ats64', 'matrix')]
# Structure is measured only for primary scenarios.
results = []
for phase, left, right in [('historical', 'B0', 'B1'), ('prex', 'B1', 'preX')]:
    for scenario, seterr in cases:
        label = f'{phase}-{mode}-{scenario}-{seterr}'
        output = root / label
        if (output / 'manifest.json').is_file():
            print(f'SKIP {label}', flush=True)
            results.append({'label': label, 'status': 'already_complete'})
            continue
        if output.exists():
            raise RuntimeError(f'partial output exists: {output}')
        a = arms[left]
        b = arms[right]
        command = [sys.executable, str(harness), 'capture', '--fixtures', str(fixtures), '--scenario', scenario, '--mode', mode, '--numpy-seterr', seterr, '--output', str(output), '--phase', phase, '--pre-x-sha', prex, '--samples', '9', '--python-a', a[0], '--wheel-a', a[1], '--source-a', a[2], '--python-b', b[0], '--wheel-b', b[1], '--source-b', b[2]]
        print(f'START {label}', flush=True)
        completed = subprocess.run(command, capture_output=True, text=True)
        (root / f'{label}.stdout').write_text(completed.stdout)
        (root / f'{label}.stderr').write_text(completed.stderr)
        status = 'complete' if completed.returncode == 0 and (output / 'manifest.json').is_file() else 'failed'
        results.append({'label': label, 'status': status, 'returncode': completed.returncode, 'command': command})
        (root / f'{mode}-execution-index.json').write_text(json.dumps(results, indent=2) + '\n')
        print(f'END {label} {status}', flush=True)
        if status == 'failed':
            print(completed.stderr[-3000:], flush=True)
            raise SystemExit(1)
