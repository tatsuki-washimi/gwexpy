#!/usr/bin/env python3
"""Runtime probe for #781 / ERR-021: coupling auto-calibration warning (line 618).

Forces optimizer non-success via mock; does not assert a diagnostic defect.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gwexpy.analysis.coupling import CouplingFunctionAnalysis
from gwexpy.timeseries import TimeSeries


def main() -> int:
    fs = 4096
    data = np.random.randn(fs)
    raw = TimeSeries(data, sample_rate=fs, unit="m")
    tgt = TimeSeries(data * 0.1, sample_rate=fs, unit="m")
    inst = CouplingFunctionAnalysis()
    fake = type("R", (), {"success": False, "x": 1.5, "fun": 0.25})()

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with patch("scipy.optimize.minimize_scalar", return_value=fake):
            out = inst.auto_calibrate_percentile_factor(raw, tgt, fftlength=1.0)

    result = {
        "method": "CouplingFunctionAnalysis.auto_calibrate_percentile_factor",
        "source_line_baseline": "gwexpy/analysis/coupling.py:618",
        "forced_non_convergence": "unittest.mock minimize_scalar success=False",
        "returned_success_field": out.get("success"),
        "warnings": [],
    }
    for w in rec:
        result["warnings"].append(
            {
                "category": w.category.__name__,
                "message": str(w.message),
                "filename": w.filename,
                "lineno": w.lineno,
            }
        )

    out_path = Path(__file__).with_name("coupling_probe_result.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
