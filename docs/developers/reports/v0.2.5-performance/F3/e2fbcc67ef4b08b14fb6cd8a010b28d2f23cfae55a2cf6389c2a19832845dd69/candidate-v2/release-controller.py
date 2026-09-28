"""Nine-per-arm F3 candidate comparison using immutable baseline harness modules."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
from pathlib import Path

HARNESS_ROOT = Path('/tmp/gwexpy-v025-f3-frozen-harness/benchmarks/io')
sys.path.insert(0, str(HARNESS_ROOT))
import f3_campaign as frozen
import f3_verify as oracle
import run

ORDER = tuple('ABBABAABABBABAABAB')  # ABBA · BAAB · ABBA · BAAB · AB
assert len(ORDER) == 18 and ORDER.count('A') == ORDER.count('B') == 9
SOURCE_A = '1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c'
SOURCE_B = 'ffba2e5e27472c098cc1fb45f47a21fffcf918a1'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(record: dict, mode: str) -> tuple[dict, dict, bool]:
    sample = record['sample']
    if mode == 'memory':
        worker = sample['worker']
        return worker['fingerprint']['values'], worker['fingerprint']['frequencies'], worker['selection_pushdown_supported']
    return sample['selected_values'], sample['selected_frequencies'], sample['selection_pushdown_supported']


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('warm', 'cold', 'memory'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--warmup-manifest', required=True)
    parser.add_argument('--python-a', required=True)
    parser.add_argument('--python-b', required=True)
    parser.add_argument('--wheel-a', required=True)
    parser.add_argument('--wheel-b', required=True)
    args = parser.parse_args()
    args.version_a = '0.2.5'
    args.version_b = '0.2.5'
    args.source_sha_a = SOURCE_A
    args.source_sha_b = SOURCE_B
    fixture = json.loads(Path(args.manifest).read_text())
    warmup_fixture = json.loads(Path(args.warmup_manifest).read_text())
    source = fixture['cases']['many_valid']
    warmup = warmup_fixture['cases']['many_valid']
    frozen.harness._checked_case_hash(source)
    frozen.harness._checked_case_hash(warmup)
    expected_values, expected_axis = oracle._selected_oracle(fixture)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit = {arm: frozen._subprocess(frozen._command(args, arm, '_audit'))['audit'] for arm in ('A', 'B')}
    if audit['A']['python'] != audit['B']['python'] or audit['A']['distributions'] != audit['B']['distributions']:
        raise RuntimeError('Python or dependency distributions differ')
    frozen._write_new(output / 'manifest.json', {
        'schema': 'gwexpy-f3-candidate-9-arm-v1',
        'status': 'CANDIDATE_UNREVIEWED',
        'mode': args.mode,
        'order': ORDER,
        'samples_per_arm': 9,
        'source_commits': {'A': SOURCE_A, 'B': SOURCE_B},
        'harness_source_commit': '000f1f4988fef13ed33eaf37661ca8deef4561d3',
        'harness_digest': run.harness_digest(),
        'controller_sha256': sha(Path(__file__)),
        'fixture_manifest_sha256': sha(Path(args.manifest)),
        'warmup_manifest_sha256': sha(Path(args.warmup_manifest)),
        'host': {'platform': platform.platform(), 'uname': list(platform.uname())},
        'arms': {arm: {**audit[arm], 'source_sha': SOURCE_A if arm == 'A' else SOURCE_B, 'install_mode': 'wheel-no-deps'} for arm in ('A', 'B')},
    })
    records = {'A': [], 'B': []}
    services = {arm: frozen._start_service(args, arm, source['path'], warmup['path']) for arm in ('A','B')} if args.mode == 'warm' else {}
    try:
        for index, arm in enumerate(ORDER):
            if args.mode == 'memory':
                record = frozen._pss(args, arm, warmup['path'])
            elif args.mode == 'warm':
                record = frozen._warm_sample(services[arm])
            else:
                record = frozen._subprocess(frozen._command(args, arm, '_timer') + ['--path', source['path'], '--warmup', warmup['path'], '--mode', 'cold'])
            frozen._verify_sample_identity(record, audit[arm], args.mode)
            values, axis, pushdown = fingerprint(record, args.mode)
            if values != expected_values or axis != expected_axis or pushdown != (arm == 'B'):
                raise RuntimeError(f'fingerprint/pushdown mismatch at {index}/{arm}')
            record['audit'] = {key: audit[arm][key] for key in ('gwexpy_path','gwexpy_version','wheel_sha256')}
            records[arm].append(record)
            frozen._write_new(output / f'sample-{index:02d}-{arm}.json', record)
    finally:
        for process in services.values():
            process.stdin.write('quit\n')
            process.stdin.flush()
            process.communicate(timeout=20)
    for arm in ('A','B'):
        frozen._write_new(output / f'raw-{arm}.json', records[arm])
    frozen._write_new(output / 'summary.json', {arm: frozen._summary(records[arm], args.mode) for arm in ('A','B')})
    frozen._write_new(output / 'artifact-hashes.json', {item.name: sha(item) for item in sorted(output.iterdir()) if item.is_file()})
    print(output)

if __name__ == '__main__':
    main()
