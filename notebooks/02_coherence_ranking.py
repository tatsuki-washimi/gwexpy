"""
Example 2: Heterogeneous format fusion → coherence ranking (marimo-compatible notebook)

This notebook reproduces the paper Example 2 using **synthetic data**.

It demonstrates:
- Building a `TimeSeriesMatrix` from a list of `TimeSeries` objects
- `TimeSeriesMatrix.coherence_ranking(target, band)` for Welch-based BruCo-style analysis
- `BrucoResult.topk(n, band)` for band-limited channel ranking
- `BrucoResult.plot_ranked(top_k)` for visualization

Note on heterogeneous formats: In real workflows each `TimeSeries` could come from
different format readers (GWF, WIN, WAV, NDS2, …). Here we use synthetic signals
for CI reproducibility without requiring proprietary data.

Run with: marimo edit 02_coherence_ranking.py
Or as a script: python 02_coherence_ranking.py
"""


def main():
    """Main execution function"""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for file output

    import numpy as np
    from pathlib import Path
    from gwpy.timeseries import TimeSeries

    from gwexpy.timeseries import TimeSeriesMatrix

    # ==== 1. Generate synthetic multi-channel data ====
    print("=" * 60)
    print("1. Generating synthetic data")
    print("=" * 60)

    rng = np.random.default_rng(0)
    sr = 2048           # sample rate [Hz]
    duration = 60       # seconds
    n = sr * duration
    t = np.arange(n) / sr

    # Common 50 Hz source (simulates environmental coupling)
    common = np.sin(2 * np.pi * 50 * t)

    ifo   = TimeSeries(common + 0.1 * rng.standard_normal(n), t0=0, sample_rate=sr, name='IFO:CH')
    corr  = TimeSeries(common + 0.3 * rng.standard_normal(n), t0=0, sample_rate=sr, name='AUX:CORR')
    noise = TimeSeries(rng.standard_normal(n),                 t0=0, sample_rate=sr, name='AUX:NOISE')

    print('Channels:')
    for ts in [ifo, corr, noise]:
        print(f'  {ts.name:12s}  sample_rate={ts.sample_rate}  duration={ts.duration}')

    # ==== 2. Build TimeSeriesMatrix ====
    print("\n" + "=" * 60)
    print("2. Building TimeSeriesMatrix")
    print("=" * 60)

    # This mirrors the paper Listing 3:
    #   matrix = TimeSeriesMatrix([[t1, t2, t3]]).resample(rate=2048)
    # (No resample needed here since all channels are already at 2048 Hz)
    matrix = TimeSeriesMatrix([[ifo, corr, noise]])

    print(f'Matrix shape : {matrix.shape}')
    print(f'Channels     : {matrix.channel_names}')

    # ==== 3. Coherence ranking ====
    print("\n" + "=" * 60)
    print("3. Computing coherence ranking")
    print("=" * 60)

    result = matrix.coherence_ranking(
        target='IFO:CH',
        band=(10, 100),    # Hz – search band
        top_n=2,
        fftlength=4.0,
        overlap=2.0,
        parallel=1,
    )

    print(f'BrucoResult target : {result.target_name}')
    print(f'Frequency bins     : {result.n_bins}')

    # ==== 4. Top-k channels and visualization ====
    print("\n" + "=" * 60)
    print("4. Identifying top channels and creating visualization")
    print("=" * 60)

    top_channels = result.topk(n=2)
    print('Top channels (all band):', top_channels)

    top_band = result.topk(n=2, band=(40, 60))
    print('Top channels (40-60 Hz):', top_band)

    fig = result.plot_ranked(top_k=2)
    fig.suptitle('Coherence ranking: top-2 channels vs IFO:CH', y=1.02)

    # Save figures for paper
    output_dir = Path(__file__).parent.parent / "docs" / "gwexpy-paper"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(output_dir / "figure3_coherence_ranking.png"), dpi=300, bbox_inches='tight')
    fig.savefig(str(output_dir / "figure3_coherence_ranking.pdf"), bbox_inches='tight')

    print(f"✅ Saved figures to {output_dir}/")
    print(f"   - figure3_coherence_ranking.png")
    print(f"   - figure3_coherence_ranking.pdf")

    # ==== 5. Validation ====
    print("\n" + "=" * 60)
    print("5. Validation")
    print("=" * 60)

    import matplotlib.pyplot as plt

    assert top_channels[0] == 'AUX:CORR', (
        f'Expected AUX:CORR as top channel, got {top_channels[0]}'
    )
    assert top_band[0] == 'AUX:CORR', (
        f'Expected AUX:CORR top in 40-60 Hz band, got {top_band[0]}'
    )
    print('✅ Validation passed: AUX:CORR correctly identified as most coherent channel.')
    plt.close('all')

    print("\n" + "=" * 60)
    print("✅ All steps completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
