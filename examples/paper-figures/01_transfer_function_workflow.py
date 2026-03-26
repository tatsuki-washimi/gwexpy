"""
Example 1: Transfer-function visualization (marimo-compatible notebook)

This notebook reproduces the paper Example 1 using **synthetic data** that simulates
a second-order low-pass system with known transfer function.

It demonstrates:
- Computing a transfer function from two `TimeSeries` using `TimeSeries.transfer_function()`
- Converting a complex `FrequencySeries` to dB magnitude via `to_db()`
- Extracting phase in degrees via `degree()`
- Producing a Bode-style two-panel plot via `gwexpy.plot.Plot`

Note on DTT XML: In real workflows the `FrequencySeriesMatrix` would be read
directly from a DTT XML file: `FrequencySeriesMatrix.read("file.xml", format="dttxml", products="TF")`.
Here we construct an equivalent `FrequencySeries` from synthetic time-domain data.

Run with: marimo edit 01_transfer_function_workflow.py
Or as a script: python 01_transfer_function_workflow.py
"""


def main():
    """Main execution function"""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for file output

    import numpy as np
    from scipy import signal as scipy_signal
    from pathlib import Path

    from gwexpy.timeseries import TimeSeries
    from gwexpy.plot import Plot

    # ==== 1. Generate synthetic input/output time series ====
    print("=" * 60)
    print("1. Generating synthetic data")
    print("=" * 60)

    rng = np.random.default_rng(42)
    sr = 2048          # sample rate [Hz]
    duration = 60      # seconds
    n = sr * duration

    # White-noise excitation
    x = rng.standard_normal(n)

    # Second-order low-pass: cutoff 50 Hz, Q = 0.707 (Butterworth)
    f_cut = 50.0  # Hz
    sos = scipy_signal.butter(2, f_cut / (sr / 2), btype='low', output='sos')
    y = scipy_signal.sosfilt(sos, x)

    ts_in  = TimeSeries(x, t0=0, sample_rate=sr, name='INPUT')
    ts_out = TimeSeries(y, t0=0, sample_rate=sr, name='OUTPUT')

    print(f'Sample rate : {sr} Hz')
    print(f'Duration    : {duration} s')
    print(f'Cutoff freq : {f_cut} Hz')

    # ==== 2. Estimate transfer function ====
    print("\n" + "=" * 60)
    print("2. Estimating transfer function")
    print("=" * 60)

    tf = ts_in.transfer_function(ts_out, fftlength=4.0, overlap=2.0, mode='steady')

    print(f'FrequencySeries length : {len(tf)}')
    print(f'Frequency resolution   : {tf.df:.3f} Hz')
    print(f'Max frequency          : {tf.frequencies.value[-1]:.1f} Hz')

    # ==== 3. Bode-style plot ====
    print("\n" + "=" * 60)
    print("3. Creating Bode plot and saving figures")
    print("=" * 60)

    mag_db = tf.to_db()
    phase_deg = tf.degree()

    plot = Plot(mag_db, phase_deg, separate=True, sharex=True, figsize=(7, 5))

    # Set y-axis labels explicitly
    axes = plot.axes if hasattr(plot, 'axes') else plot.figure.axes
    if len(axes) >= 2:
        axes[0].set_ylabel('Amplitude [dB]')
        axes[1].set_ylabel('Phase [°]')

    # Save figures for paper
    output_dir = Path(__file__).parent.parent / "docs" / "gwexpy-paper"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot.save(str(output_dir / "figure2_transfer_function.png"), dpi=300, bbox_inches='tight')
    plot.save(str(output_dir / "figure2_transfer_function.pdf"), bbox_inches='tight')

    print(f"✅ Saved figures to {output_dir}/")
    print(f"   - figure2_transfer_function.png")
    print(f"   - figure2_transfer_function.pdf")

    # ==== 4. Validation ====
    print("\n" + "=" * 60)
    print("4. Validation")
    print("=" * 60)

    import matplotlib.pyplot as plt

    freqs = tf.frequencies.value
    # Analytical magnitude of the 2nd-order LP filter
    w, H_theory = scipy_signal.sosfreqz(sos, worN=freqs, fs=sr)
    H_theory_db = 20 * np.log10(np.abs(H_theory) + 1e-30)

    est_db = mag_db.value

    # Compare in the 1–200 Hz band where SNR is good
    mask = (freqs >= 1.0) & (freqs <= 200.0)
    rms_error_db = np.sqrt(np.mean((est_db[mask] - H_theory_db[mask])**2))
    print(f'RMS error between estimated and theoretical TF magnitude: {rms_error_db:.2f} dB')

    # Should be within a few dB given 60 s of data
    assert rms_error_db < 3.0, (
        f'TF estimation error too large: {rms_error_db:.2f} dB (expected < 3 dB)'
    )
    print('✅ Validation passed.')
    plt.close('all')

    print("\n" + "=" * 60)
    print("✅ All steps completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
