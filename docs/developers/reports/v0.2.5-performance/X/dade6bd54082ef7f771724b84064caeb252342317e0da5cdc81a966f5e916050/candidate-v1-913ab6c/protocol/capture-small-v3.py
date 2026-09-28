from pathlib import Path
import json
import subprocess
import sys

mode = sys.argv[1]
assert mode in ('wall', 'pss')
root = Path('/tmp/gwexpy-v025-bx-x913-small-v3')
root.mkdir(exist_ok=True)
harness = Path('/tmp/gwexpy-v025-bx-supplementary-edge.py')
fixtures = Path('/tmp/gwexpy-v025-bx-fixtures-v1')
prex = '7f13c4423addab4f92eafc2740da05faa4a444fa'
candidate = '913ab6c779d83ddde4952194a33e00a94c602ff4'
arms = {
    'A': ('/tmp/gwexpy-v025-bx-env-7f13/bin/python', '/tmp/gwexpy-v025-bx-prex-7f13-wheel/gwexpy-0.2.5-py3-none-any.whl', prex),
    'B': ('/tmp/gwexpy-v025-bx-env-x913/bin/python', '/tmp/gwexpy-v025-bx-x913-wheel/gwexpy-0.2.5-py3-none-any.whl', candidate),
}
results = []
for scenario in ('ats_overflow','matrix_int64_extrema','matrix_nan_inf'):
    label = f'candidate-{mode}-{scenario}-warn'
    out = root / label
    if out.exists():
        raise RuntimeError(f'Existing output: {out}')
    cmd = [sys.executable, str(harness), 'capture', '--fixtures', str(fixtures), '--scenario', scenario, '--mode', mode, '--numpy-seterr', 'warn', '--output', str(out), '--phase', 'candidate', '--pre-x-sha', prex, '--samples', '9']
    for arm in ('A','B'):
        py,wheel,source=arms[arm]
        suffix=arm.lower()
        cmd += [f'--python-{suffix}', py, f'--wheel-{suffix}', wheel, f'--source-{suffix}', source]
    print('START',label,flush=True)
    p=subprocess.run(cmd,capture_output=True,text=True)
    (root/f'{label}.stdout').write_text(p.stdout)
    (root/f'{label}.stderr').write_text(p.stderr)
    status='complete' if p.returncode==0 and (out/'manifest.json').is_file() else 'failed'
    results.append({'label':label,'status':status,'returncode':p.returncode,'command':cmd})
    (root/f'{mode}-execution-index.json').write_text(json.dumps(results,indent=2)+'\n')
    print('END',label,status,flush=True)
    if status!='complete':
        print(p.stderr[-4000:],flush=True)
        raise SystemExit(1)
