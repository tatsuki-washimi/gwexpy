import inspect
import os

import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.frequencyseries import FrequencySeries as GwexFrequencySeries
from gwexpy.plot import Plot as GwexPlot
from gwexpy.timeseries import TimeSeries as GwexTimeSeries


def log_result(test_name, success, message=""):
    status = "PASS" if success else "FAIL"
    print(f"[{status}] {test_name}")
    if message:
        print(f"       Details: {message}")
    return success


def test_attribute_preservation():
    print("\n--- Test: Internal Attribute Preservation (__array_finalize__) ---")
    data = np.random.randn(100)
    fs = GwexFrequencySeries(data, df=1, unit="Hz")

    # Set internal attribute manually (simulating ifft setup)
    fs._gwex_fft_mode = "transient"
    fs._gwex_target_nfft = 200

    # 1. Slicing
    fs_slice = fs[10:90]
    preserved_slice = hasattr(fs_slice, "_gwex_fft_mode")

    # 2. Arithmetic
    fs_math = fs * 2.0
    preserved_math = hasattr(fs_math, "_gwex_fft_mode")

    # 3. Ufunc (e.g. abs)
    fs_abs = np.abs(fs)
    # Note: np.abs returns a new object, might drop attributes if not handled
    preserved_abs = hasattr(fs_abs, "_gwex_fft_mode")

    success = preserved_slice and preserved_math
    msg = []
    if not preserved_slice:
        msg.append("Lost attribute after slicing")
    if not preserved_math:
        msg.append("Lost attribute after arithmetic (*)")
    if not preserved_abs:
        msg.append("Lost attribute after ufunc (abs)")

    return log_result("Attribute Preservation", success, ", ".join(msg))


def test_plot_show_save_side_effect():
    print("\n--- Test: Plot.show() Side Effects ---")
    ts = GwexTimeSeries(np.random.randn(100), sample_rate=100)
    plot = ts.plot(show=False)

    # Mock plt.show to avoid actual display (must accept block parameter)
    original_show = plt.show
    plt.show = lambda block=None: None  # type: ignore

    try:
        # gwexpy's show() now supports 'close' and 'block' arguments.
        # We set close=False and block=False to verify we can still use the plot object.
        plot.show(close=False, block=False)

        # Now plot should be OPEN.
        try:
            plot.savefig("test_after_show.png")

            # Use plt.fignum_exists to verify the figure is still managed by pyplot
            # plot.number should exist if it's a standard Figure
            is_active = False
            if hasattr(plot, "number"):
                is_active = plt.fignum_exists(plot.number)

            if not is_active:
                return log_result(
                    "Plot.show() Side Effect",
                    False,
                    "Figure was closed or detached despite close=False",
                )
            else:
                return log_result(
                    "Plot.show() Side Effect",
                    True,
                    "Figure remained active and savefig worked",
                )

        except Exception as e:
            return log_result("Plot.show() Side Effect", False, f"savefig failed: {e}")
    finally:
        plt.show = original_show
        try:
            plt.close(plot)
        except Exception:
            pass
        if os.path.exists("test_after_show.png"):
            os.remove("test_after_show.png")


def test_whiten_signature():
    print("\n--- Test: Method Signature Compatibility (whiten) ---")
    # Gwex TimeSeries might not implement whiten directly, inheriting from Gwpy
    # or it might implement it via Mixin.

    gwpy_sig = inspect.signature(GwpyTimeSeries.whiten)
    gwex_sig = inspect.signature(GwexTimeSeries.whiten)

    success = gwpy_sig == gwex_sig
    msg = f"\n       Gwpy: {gwpy_sig}\n       Gwex: {gwex_sig}" if not success else ""
    return log_result("Whiten Signature Match", success, msg)


def test_mixed_unit_labels():
    print("\n--- Test: Mixed Unit Y-Labeling ---")
    ts1 = GwexTimeSeries(np.random.randn(100), unit="m", name="Displacement")
    ts2 = GwexTimeSeries(np.random.randn(100), unit="V", name="Voltage")

    # Plotting mixed units together
    plot = GwexPlot(ts1, ts2, show=False)
    ax = plot.gca()
    ylabel = ax.get_ylabel()

    # Gwpy behavior: usually picks the first unit or errors/warns?
    # Gwexpy logic: if units differ, it might suppress global ylabel.

    # Let's see what we got
    print(f"       Resulting Y-Label: '{ylabel}'")

    # Verify expectations: Ideally it should NOT show just 'm' if 'V' is also there,
    # unless they are compatible. m and V are not.

    # If it is empty, that is arguably "Safe" (Gwexpy logic).
    # If it is 'm', that is potentially misleading.

    if ylabel == "":
        return log_result("Mixed Unit Labeling", True, "Y-Label suppressed (Safe)")
    elif ylabel == "m":
        return log_result(
            "Mixed Unit Labeling",
            False,
            "Y-Label is 'm' despite mixed units (Potentially misleading)",
        )
    else:
        return log_result("Mixed Unit Labeling", True, f"Y-Label is '{ylabel}'")


if __name__ == "__main__":
    failures = []

    if not test_attribute_preservation():
        failures.append("Attribute Preservation")
    if not test_plot_show_save_side_effect():
        failures.append("Plot.show() Side Effect")
    if not test_whiten_signature():
        failures.append("Whiten Signature")
    if not test_mixed_unit_labels():
        failures.append("Mixed Unit Labeling")

    print("\n========================================")
    if failures:
        print(f"SUMMARY: Found {len(failures)} issues that require fixing.")
        print("Please record these in FIX_REQUIRED_LIST.md")
    else:
        print("SUMMARY: No critical issues found in this run.")
