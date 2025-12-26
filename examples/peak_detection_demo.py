
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import sys
from typing import Optional, List
from gwexpy.frequencyseries import FrequencySeries

def _as_plain_array(x):
    return np.asarray(getattr(x, "value", x))


def _try_savefig(fig, output_path: Path) -> Optional[Path]:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        return output_path
    except OSError as exc:
        print(f"Could not save plot to '{output_path}': {exc}")
        return None


def _is_headless() -> bool:
    return os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Peak detection demo for gwexpy FrequencySeries.")
    parser.add_argument(
        "-o",
        "--output",
        default="peak_detection_demo.png",
        help="Output image path (defaults to ./peak_detection_demo.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window.",
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")

    # 1. データの準備
    # 複数のピークを持つ擬似的なスペクトルデータを作成します
    df = 1.0   # 周波数分解能 [Hz]
    f_axis = np.arange(0, 500, df)
    data = np.exp(-f_axis / 100) * np.random.normal(1, 0.1, len(f_axis))  # ノイズ成分

    # 特定の周波数にピークを追加
    peak_info = [(50, 2.0), (120, 1.5), (130, 1.8), (300, 1.2)]
    for f, amp in peak_info:
        idx = int(f / df)
        data[idx] += amp

    # FrequencySeries オブジェクトの作成
    spec = FrequencySeries(data, df=df, unit="m")
    print(f"FrequencySeries created with units: {spec.unit}")

    # 2. 基本的なピーク検出 (find_peaks)
    peak_indices, _props = spec.find_peaks(threshold=1.5)
    peak_freqs = spec.frequencies[peak_indices]
    peak_values = np.abs(spec[peak_indices])
    print(f"Detected peaks at frequencies: {peak_freqs}")

    # 3. 可視化 (Visualization)
    plt.figure(figsize=(12, 6))
    plt.plot(spec.frequencies, np.abs(spec), label="Magnitude Spectrum", color="navy", alpha=0.7)
    plt.scatter(peak_freqs, peak_values, color="red", marker="o", s=100, label="Basic Peaks (threshold=1.5)")

    for f, v in zip(_as_plain_array(peak_freqs), _as_plain_array(peak_values)):
        plt.text(f, v + 0.1, f"{f:.0f}Hz\n{v:.2f}", ha="center", va="bottom", color="red", weight="bold")

    # 4. 高度なピーク検出 (distance, prominence)
    peak_indices_adv, _props_adv = spec.find_peaks(distance=20, prominence=0.5)
    peak_freqs_adv = spec.frequencies[peak_indices_adv]
    peak_values_adv = np.abs(spec[peak_indices_adv])
    plt.scatter(
        peak_freqs_adv,
        peak_values_adv,
        color="gold",
        marker="x",
        s=150,
        linewidths=3,
        label="Advanced Peaks (dist=20, prom=0.5)",
    )

    plt.xlabel(f"Frequency [{spec.frequencies.unit}]")
    plt.ylabel(f"Amplitude [{spec.unit}]")
    plt.title("Peak Detection Example in FrequencySeries")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    output_path = _try_savefig(plt.gcf(), Path(args.output))
    if output_path is not None:
        print(f"Demo plot saved to '{output_path}'")

    if not args.no_show and not _is_headless():
        plt.show()

    # 5. 異なるメソッドでの検出 (db)
    peak_indices_db, _ = spec.find_peaks(threshold=0, method="db")  # 0dB以上
    print(f"Detected peaks in dB above 0dB: {spec.frequencies[peak_indices_db]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
