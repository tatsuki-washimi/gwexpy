import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_ROOT = ROOT / "docs" / "web"


def _read_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def _joined_code(nb: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _joined_text(nb: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))


def test_case_violin_mode_notebook_uses_physical_limits_and_resolution_checks():
    for locale in ("en", "ja"):
        nb = _read_notebook(TUTORIAL_ROOT / locale / "user_guide" / "tutorials" / "case_violin_mode.ipynb")
        joined = _joined_code(nb)

        assert "limits=FIT_LIMITS" in joined
        assert 'limits=cfg["limits"]' in joined
        assert '"x0": (169.8, 170.2)' in joined
        assert '"Q": (1e3, 5e4)' in joined
        assert "background_1st = np.median" in joined
        assert "line_only_1st = np.clip(asd_1st.value - background_1st, 0, None)" in joined
        assert "fftlength = 16.0" in joined
        assert "TRACK_BAND = (169.9, 170.2)" in joined
        assert "relative_error = abs(recovered_drift_hz - injected_drift_hz) / injected_drift_hz" in joined
        assert "if relative_error <= 0.2:" in joined
        assert "Tracking acceptance: PASS (within ±20% of injected drift)" in joined


def test_case_bootstrap_gls_fitting_notebook_uses_psd_consistent_peak_model():
    for locale in ("en", "ja"):
        nb = _read_notebook(
            TUTORIAL_ROOT / locale / "user_guide" / "tutorials" / "case_bootstrap_gls_fitting.ipynb"
        )
        joined = _joined_code(nb)
        notebook_text = _joined_text(nb)

        assert "background_psd = background_asd.value ** 2" in joined
        assert "gamma_hwhm = peak_freq / (2 * peak_Q)" in joined
        assert "peak_psd_amplitude = peak_amplitude ** 2" in joined
        assert "peak_psd = peak_psd_amplitude * gamma_hwhm**2" in joined
        assert "total_psd_values = background_psd + peak_psd" in joined
        assert "total_asd_values = np.sqrt(total_psd_values)" in joined
        assert "peak_amplitude = 5e-21" in joined
        assert "n_boot = 300" in joined
        assert "\\frac{\\gamma^2}{(f-f_0)^2 + \\gamma^2}" in notebook_text
        assert "gamma = f_0/(2Q)" in notebook_text
        assert "gamma_hwhm = f0 / (2 * Q)" in joined
        assert "peak_psd = A_peak * gamma_hwhm**2" in joined
        assert "return background + peak_psd" in joined
        assert "background_asd.value ** 2 + lorentzian ** 2" not in joined
