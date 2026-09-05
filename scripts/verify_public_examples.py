#!/usr/bin/env python3
"""Execute public onboarding lessons and render their shared introductory figure."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import runpy
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


def verify(source: Path, expected_version: str | None = None) -> None:
    """Run examples in a disposable directory, checking numerical and I/O results."""
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy import units as u

    import gwexpy
    from gwexpy.timeseries import TimeSeriesDict

    if expected_version is not None:
        assert gwexpy.__version__ == expected_version, (
            gwexpy.__version__,
            expected_version,
        )

    source = source.resolve()
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="gwexpy-intro-") as temporary:
        try:
            os.chdir(temporary)
            quickstart = runpy.run_path(str(source / "_static/downloads/quickstart.py"))
            for spectrum in quickstart["spectra"].values():
                assert spectrum.unit == u.V / u.Hz**0.5
                assert spectrum.frequencies[np.argmax(spectrum.value)] == 40 * u.Hz
            shutil.copy2("asd.png", source / "_static/images/quickstart-asd.png")

            for name, namespace in (
                ("first_analysis", quickstart),
                ("scientific_python", {}),
            ):
                lesson = source / "tutorials" / f"{name}.md"
                for code in re.findall(
                    r"^```python\n(.*?)^```", lesson.read_text(), re.M | re.S
                ):
                    exec(compile(code, str(lesson), "exec"), namespace)
                if name == "scientific_python":
                    for channel, spectrum in namespace["spectra"].items():
                        expected_frequency, expected_asd = namespace["numpy_spectra"][
                            channel
                        ]
                        np.testing.assert_allclose(
                            spectrum.frequencies.value, expected_frequency
                        )
                        np.testing.assert_allclose(
                            spectrum.value, expected_asd, rtol=1e-12
                        )

            commissioner = runpy.run_path(
                str(source / "_static/downloads/commissioner.py")
            )
            for name, channel in commissioner["channels"].items():
                np.testing.assert_array_equal(
                    channel.value, commissioner["loaded"][name].value
                )
                assert commissioner["segment"][name].duration == 24 * u.s
            coherence = commissioner["coherence"]
            assert (
                coherence.value[np.argmin(abs(coherence.frequencies.value - 40))] > 0.95
            )

            if importlib.util.find_spec("dttxml") is not None:
                xml_data = TimeSeriesDict.read(
                    source / "_static/downloads/commissioner.xml",
                    format="xml.diaggui",
                    products="TS",
                    unit="V",
                )
                channel = xml_data["TEST:SYNTHETIC_INPUT"]
                np.testing.assert_array_equal(channel.value, [0, 1, 0, -1])
                assert channel.sample_rate == 4 * u.Hz
                assert channel.t0 == 1234567890 * u.s
            else:
                print(
                    "Optional DiagGUI XML example skipped: install dttxml to verify it"
                )
        finally:
            plt.close("all")
            os.chdir(original)
    print("Onboarding examples passed; shared Quickstart figure regenerated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--expected-version")
    args = parser.parse_args()
    verify(args.source, args.expected_version)
