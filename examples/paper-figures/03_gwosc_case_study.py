"""
Example 3: GWOSC public data -> coherence analysis (marimo-compatible notebook)

This notebook demonstrates GWexpy's use with real LIGO detector data
from the Gravitational-Wave Open Science Center (GWOSC).

Since auxiliary channels are not publicly available through GWOSC, we derive
band-limited proxy channels from the public strain data for demonstration.

Prerequisites:
  pip install "gwexpy[analysis,gw]"

Run with: marimo edit 03_gwosc_case_study.py
Or as a script: python 03_gwosc_case_study.py

Optional cache support:
  GWEXPY_GWOSC_CACHE=/path/to/gwosc_gw150914_h1_1024s.npz python 03_gwosc_case_study.py
"""

import os
import sys
from pathlib import Path

# Ensure the development version of gwexpy is preferred over any installed release.
# When running as `python examples/paper-figures/03_gwosc_case_study.py` from the project root,
# Python sets sys.path[0] to the examples/paper-figures/ directory, not the project root.
# This line restores the project root so the local source tree takes priority.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _load_cached_strain(cache_path):
    """Load cached GWOSC strain data from a local NPZ artifact."""
    import numpy as np

    from gwexpy.timeseries import TimeSeries

    with np.load(cache_path, allow_pickle=False) as npz:
        return TimeSeries(
            npz["value"],
            dt=float(npz["dt"]),
            t0=float(npz["t0"]),
            name=str(npz["name"]),
        )


def _save_cached_strain(cache_path, strain):
    """Save fetched GWOSC strain data to a local NPZ artifact."""
    import numpy as np

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        value=strain.value,
        dt=float(strain.dt.value),
        t0=float(strain.t0.value),
        name=str(strain.name),
    )


def main():
    """Main execution function"""
    import matplotlib
    matplotlib.use("Agg")

    import numpy as np
    from pathlib import Path
    from scipy.signal import butter, sosfilt

    from gwexpy.timeseries import TimeSeries, TimeSeriesList, TimeSeriesMatrix

    # ==== 1. Fetch GWOSC public strain data ====
    print("=" * 60)
    print("1. Fetching GWOSC public strain data")
    print("=" * 60)

    try:
        from gwosc.datasets import event_gps

        gps = event_gps("GW150914")
    except Exception:
        # Fallback GPS time for GW150914 if gwosc package is not available
        gps = 1126259462.4

    gps_start = int(gps) - 512
    gps_end = int(gps) + 512
    print(f"GPS time of GW150914: {gps}")
    print(f"Fetching H1 strain: [{gps_start}, {gps_end}]")

    cache_env = os.environ.get("GWEXPY_GWOSC_CACHE")
    cache_path = Path(cache_env) if cache_env else None

    if cache_path is not None and cache_path.exists():
        print(f"Loading cached strain from: {cache_path}")
        strain = _load_cached_strain(cache_path)
    else:
        try:
            strain = TimeSeries.fetch_open_data("H1", gps_start, gps_end)
        except Exception as exc:
            if cache_path is not None:
                raise RuntimeError(
                    "GWOSC download failed and no readable cache was available at "
                    f"{cache_path}. Set GWEXPY_GWOSC_CACHE to a valid NPZ cache file "
                    "or rerun in a network-enabled environment."
                ) from exc
            raise RuntimeError(
                "GWOSC download failed. Rerun in a network-enabled environment or set "
                "GWEXPY_GWOSC_CACHE=/path/to/gwosc_gw150914_h1_1024s.npz to use a "
                "locally prepared cache."
            ) from exc

        if cache_path is not None:
            _save_cached_strain(cache_path, strain)
            print(f"Saved cache to: {cache_path}")

    print(f"Strain channel: {strain.name}")
    print(f"Sample rate: {strain.sample_rate}")
    print(f"Duration: {strain.duration}")

    # ==== 2. Create band-limited proxy auxiliary channels ====
    print("\n" + "=" * 60)
    print("2. Creating band-limited proxy auxiliary channels")
    print("=" * 60)

    rng = np.random.default_rng(42)
    fs = float(strain.sample_rate.value)
    aux_channels = []
    center_freqs = [20, 60, 120, 250, 500]

    for fc in center_freqs:
        f_low = max(fc - 10, 1)
        f_high = min(fc + 10, fs / 2 - 1)
        sos = butter(4, [f_low, f_high], btype="band", fs=fs, output="sos")
        filt = sosfilt(sos, strain.value)
        noise = rng.normal(0, np.std(filt) * 0.5, len(filt))
        aux = TimeSeries(
            filt + noise,
            dt=strain.dt,
            t0=strain.t0,
            name=f"AUX:BAND_{fc}Hz",
        )
        aux_channels.append(aux)
        print(f"  Created {aux.name} (bandpass {f_low}-{f_high} Hz)")

    # ==== 3. Build TimeSeriesMatrix and run coherence ranking ====
    print("\n" + "=" * 60)
    print("3. Building TimeSeriesMatrix and computing coherence ranking")
    print("=" * 60)

    matrix = TimeSeriesList([strain] + aux_channels).to_matrix()
    print(f"Matrix shape: {matrix.shape}")
    print(f"Channels: {matrix.channel_names}")

    target_name = strain.name
    result = matrix.coherence_ranking(
        target=target_name,
        band=(10, 300),
        top_n=3,
        fftlength=4.0,
        overlap=2.0,
        parallel=1,
    )

    print(f"BrucoResult target: {result.target_name}")
    print(f"Frequency bins: {result.n_bins}")

    # ==== 4. Results and visualization ====
    print("\n" + "=" * 60)
    print("4. Results and visualization")
    print("=" * 60)

    top_channels = result.topk(n=5)
    print("Top channels (10-300 Hz):", top_channels)

    for band_name, band in [("10-50 Hz", (10, 50)), ("50-150 Hz", (50, 150)),
                             ("150-300 Hz", (150, 300))]:
        top = result.topk(n=3, band=band)
        print(f"Top channels ({band_name}): {top}")

    fig = result.plot_ranked(top_k=3)
    fig.suptitle(
        f"Coherence ranking: GWOSC H1 strain vs proxy channels\n"
        f"(1024 s around GW150914)",
        y=1.02,
    )

    output_dir = Path(__file__).parent.parent / "docs" / "gwexpy-paper"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        str(output_dir / "figure5_gwosc_case_study.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        str(output_dir / "figure5_gwosc_case_study.pdf"),
        bbox_inches="tight",
    )
    print(f"\nSaved figures to {output_dir}/")

    # ==== 5. Validation ====
    print("\n" + "=" * 60)
    print("5. Validation")
    print("=" * 60)

    import matplotlib.pyplot as plt

    assert len(top_channels) > 0, "No channels returned from ranking"
    # Verify that band-filtered channels appear in ranking
    band_names = {f"AUX:BAND_{fc}Hz" for fc in center_freqs}
    found = set(top_channels) & band_names
    assert len(found) > 0, (
        f"Expected at least one band proxy in top channels, got {top_channels}"
    )
    print(f"Validation passed: {len(found)} proxy channel(s) in top ranking.")
    plt.close("all")

    print("\n" + "=" * 60)
    print("All steps completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
