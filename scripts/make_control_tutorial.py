import json
import os


def create_cell(source, cell_type="code"):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else source.splitlines(True)
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def make_notebook():
    cells = []

    # Title and Introduction
    cells.append(create_cell([
        "# Control Engineering with gwexpy and python-control\n",
        "\n",
        "**Introduction**\n",
        "\n",
        "This tutorial demonstrates how to combine **python-control** for system modeling and **gwexpy** for data management and visualization.\n",
        "We will explore continuous-time systems, discretization methods (Zero-Order Hold vs Bilinear Transform), and analyze them in both frequency and time domains.\n",
        "\n",
        "Reference: This tutorial is based on concepts from standard control engineering texts regarding discretization."
    ], cell_type="markdown"))

    # Imports
    cells.append(create_cell([
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import control\n",
        "from control.matlab import tf, c2d, step, lsim, bode, pole, zero\n",
        "import gwexpy\n",
        "from gwexpy import TimeSeries, FrequencySeries\n",
        "\n",
        "print(f\"gwexpy version: {gwexpy.__version__}\")\n",
        "print(f\"control version: {control.__version__}\")"
    ]))

    # 1. Continuous Time System
    cells.append(create_cell([
        "## 1. Continuous Time System Model\n",
        "\n",
        "We define a simple first-order continuous-time transfer function:\n",
        "$$ P(s) = \\frac{1}{0.5s + 1} $$"
    ], cell_type="markdown"))

    cells.append(create_cell([
        "# Define Transfer Function P(s) = 1 / (0.5s + 1)\n",
        "num = [0, 1]\n",
        "den = [0.5, 1]\n",
        "P = tf(num, den)\n",
        "print(\"Continuous Time System P(s):\")\n",
        "print(P)"
    ]))

    cells.append(create_cell([
        "# Check poles and zeros\n",
        "print(f\"Poles: {pole(P)}\")\n",
        "print(f\"Zeros: {zero(P)}\")"
    ]))

    # 2. Discretization
    cells.append(create_cell([
        "## 2. Discretization\n",
        "\n",
        "We will convert the continuous system to discrete-time systems using two methods with a sampling time $t_s = 0.2s$:\n",
        "1.  **Zero-Order Hold (ZOH)**: Good for step response (time domain).\n",
        "2.  **Bilinear Transformation (Tustin)**: Good for frequency response (frequency domain)."
    ], cell_type="markdown"))

    cells.append(create_cell([
        "ts = 0.2  # Sampling time (seconds)\n",
        "\n",
        "# 1. Zero-Order Hold\n",
        "Pd_zoh = c2d(P, ts, method='zoh')\n",
        "print(\"Discrete System (ZOH):\")\n",
        "print(Pd_zoh)\n",
        "\n",
        "# 2. Bilinear (Tustin)\n",
        "Pd_tustin = c2d(P, ts, method='tustin')\n",
        "print(\"\\nDiscrete System (Tustin):\")\n",
        "print(Pd_tustin)"
    ]))

    # 3. Frequency Response (Bode Plot) using gwexpy
    cells.append(create_cell([
        "## 3. Frequency Response (Bode Plot)\n",
        "\n",
        "We compare the frequency responses. Note how Tustin matches the continuous phase better at high frequencies.\n",
        "We use `gwexpy.FrequencySeries` for plotting to leverage its unit handling."
    ], cell_type="markdown"))

    cells.append(create_cell([
        "# Calculate Bode data\n",
        "omega = np.logspace(-2, 2, 100)\n",
        "\n",
        "mag_c, phase_c, w_c = bode(P, omega, plot=False)\n",
        "mag_z, phase_z, w_z = bode(Pd_zoh, omega, plot=False)\n",
        "mag_t, phase_t, w_t = bode(Pd_tustin, omega, plot=False)\n",
        "\n",
        "# Convert to gwexpy FrequencySeries\n",
        "# Note: control.bode returns magnitude (not dB) and phase in radians.\n",
        "# We define units explicitly.\n",
        "\n",
        "fs_mag_c = FrequencySeries(20 * np.log10(mag_c), frequencies=w_c, unit='dB', name='Continuous (Mag)')\n",
        "fs_mag_z = FrequencySeries(20 * np.log10(mag_z), frequencies=w_z, unit='dB', name='ZOH (Mag)')\n",
        "fs_mag_t = FrequencySeries(20 * np.log10(mag_t), frequencies=w_t, unit='dB', name='Tustin (Mag)')\n",
        "\n",
        "fs_phase_c = FrequencySeries(np.rad2deg(phase_c), frequencies=w_c, unit='deg', name='Continuous (Phase)')\n",
        "fs_phase_z = FrequencySeries(np.rad2deg(phase_z), frequencies=w_z, unit='deg', name='ZOH (Phase)')\n",
        "fs_phase_t = FrequencySeries(np.rad2deg(phase_t), frequencies=w_t, unit='deg', name='Tustin (Phase)') # fixed: was w_z\n"
    ]))

    cells.append(create_cell([
        "# Plotting\n",
        "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)\n",
        "\n",
        "fs_mag_c.plot(ax=ax1, label='Continuous')\n",
        "fs_mag_z.plot(ax=ax1, label='ZOH', linestyle='--')\n",
        "fs_mag_t.plot(ax=ax1, label='Tustin', linestyle=':')\n",
        "ax1.set_xscale('log')\n",
        "ax1.set_ylabel('Magnitude [dB]')\n",
        "ax1.legend()\n",
        "ax1.grid(which='both', linestyle='-', alpha=0.5)\n",
        "\n",
        "fs_phase_c.plot(ax=ax2, label='Continuous')\n",
        "fs_phase_z.plot(ax=ax2, label='ZOH', linestyle='--')\n",
        "fs_phase_t.plot(ax=ax2, label='Tustin', linestyle=':')\n",
        "ax2.set_xscale('log')\n",
        "ax2.set_ylabel('Phase [deg]')\n",
        "ax2.set_xlabel('Frequency [rad/s]')\n",
        "ax2.legend()\n",
        "ax2.grid(which='both', linestyle='-', alpha=0.5)\n",
        "\n",
        "plt.suptitle('Bode Plot Comparison')\n",
        "plt.show()"
    ]))

    # 4. Time Response (Step)
    cells.append(create_cell([
        "## 4. Time Domain: Step Response\n",
        "\n",
        "ZOH is accurate for step responses since the input is held constant between samples."
    ], cell_type="markdown"))

    cells.append(create_cell([
        "T_end = 3\n",
        "Tc = np.arange(0, T_end, 0.01)\n",
        "Td = np.arange(0, T_end, ts)\n",
        "\n",
        "# Step Response\n",
        "yc, tc = step(P, Tc)\n",
        "yd_z, td_z = step(Pd_zoh, Td)\n",
        "yd_t, td_t = step(Pd_tustin, Td)\n",
        "\n",
        "# Wrap in gwexpy TimeSeries\n",
        "ts_c = TimeSeries(yc, times=tc, unit='V', name='Continuous')\n",
        "ts_z = TimeSeries(yd_z, times=td_z, unit='V', name='ZOH')\n",
        "ts_t = TimeSeries(yd_t, times=td_t, unit='V', name='Tustin')\n",
        "\n",
        "plot = ts_c.plot(label='Continuous', title='Step Response Comparison')\n",
        "ax = plot.gca()\n",
        "ts_z.plot(ax=ax, label='ZOH', marker='o', linestyle='None')\n",
        "ts_t.plot(ax=ax, label='Tustin', marker='x', linestyle='None')\n",
        "ax.legend()\n",
        "ax.grid(True)\n",
        "plt.show()"
    ]))

    # 5. Arbitrary Input Response
    cells.append(create_cell([
        "## 5. Time Domain: Arbitrary Input\n",
        "\n",
        "Response to $u(t) = 0.5\\sin(6t) + 0.5\\cos(8t)$.\n",
        "Tustin (Bilinear) generally approximates arbitrary smooth inputs better."
    ], cell_type="markdown"))

    cells.append(create_cell([
        "# Use gwexpy to generate input signal\n",
        "# Continuous high-res time\n",
        "t_fine = np.arange(0, 3, 0.01)\n",
        "u_c_ts = TimeSeries(0.5 * np.sin(6*t_fine) + 0.5 * np.cos(8*t_fine), times=t_fine)\n",
        "\n",
        "# Discrete sampling points\n",
        "t_disc = np.arange(0, 3, ts)\n",
        "u_d_ts = TimeSeries(0.5 * np.sin(6*t_disc) + 0.5 * np.cos(8*t_disc), times=t_disc)\n",
        "\n",
        "# Simulate\n",
        "yc_arb, tc_arb, _ = lsim(P, u_c_ts.value, t_fine)\n",
        "yd_z_arb, td_z_arb, _ = lsim(Pd_zoh, u_d_ts.value, t_disc)\n",
        "yd_t_arb, td_t_arb, _ = lsim(Pd_tustin, u_d_ts.value, t_disc)\n",
        "\n",
        "# Wrap results\n",
        "res_c = TimeSeries(yc_arb, times=tc_arb, unit='V', name='Continuous')\n",
        "res_z = TimeSeries(yd_z_arb, times=td_z_arb, unit='V', name='ZOH')\n",
        "res_t = TimeSeries(yd_t_arb, times=td_t_arb, unit='V', name='Tustin')\n",
        "\n",
        "plot = res_c.plot(label='Continuous', title='Response to Arbitrary Input')\n",
        "ax = plot.gca()\n",
        "res_z.plot(ax=ax, label='ZOH', marker='o', linestyle='None')\n",
        "res_t.plot(ax=ax, label='Tustin', marker='x', linestyle='None')\n",
        "ax.legend()\n",
        "ax.grid(True)\n",
        "plt.show()"
    ]))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    output_path = "examples/tutorials/tutorial_ControlEngineering.ipynb"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=2)

    print(f"Notebook generated at: {output_path}")

if __name__ == "__main__":
    make_notebook()
