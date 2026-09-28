"""Supplementary edge-fixture wall control using frozen B-X read/clock code.

Only the frozen harness's primary-fixture eligibility set is extended for this
candidate-phase diagnostic. This wrapper is not part of the formal B-X harness.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

harness = Path('/home/washimi/work/gwexpy-v025-performance-copy-implementation/benchmarks/copy_audit/bx_measure.py')
spec = spec_from_file_location('bx_frozen_harness', harness)
assert spec is not None and spec.loader is not None
module = module_from_spec(spec)
spec.loader.exec_module(module)
module.PRIMARY_SCENARIOS = tuple(module.PRIMARY_SCENARIOS) + (
    'ats_overflow', 'matrix_int64_extrema', 'matrix_nan_inf'
)
args = module._parser().parse_args(sys.argv[1:])
assert args.mode == 'wall' and args.scenario in ('ats_overflow', 'matrix_int64_extrema', 'matrix_nan_inf')
module._capture(args)
