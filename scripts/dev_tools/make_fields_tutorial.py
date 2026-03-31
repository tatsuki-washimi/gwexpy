"""Generate multi-dimensional field analysis tutorial notebooks (EN + JA)."""

import json
from pathlib import Path


def md(source):
    return {"cell_type": "markdown", "id": f"md_{abs(hash(source))%10**8:08x}",
            "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None,
            "id": f"cd_{abs(hash(source))%10**8:08x}",
            "metadata": {}, "outputs": [], "source": source}


# ---------------------------------------------------------------------------
# English notebook
# ---------------------------------------------------------------------------

EN_CELLS = [
    md("""\
# Multi-Dimensional Field Analysis: Beam Profiles and Spatial Statistics

The `gwexpy.fields` module extends gwexpy to **4-D spatial-temporal data**:
`(time, x, y, z)`.  This enables analysis of phenomena that have both
temporal and spatial structure — such as:

- **Optical beam profiles** measured across a mirror surface
- **Seismic wavefield maps** from distributed accelerometer arrays
- **Environmental field maps** (magnetic, acoustic, temperature) around
  the detector

**What this tutorial covers:**
1. Creating synthetic beam and wavefield data with `ScalarField`
2. Spatial statistics: mean profile, variance map, and spatial PSD
3. Coherence map: how coherent is the field across space?
4. Time-delay map: estimating signal propagation velocity
5. `VectorField` operations: gradient, curl (rotation), norm
6. k-space analysis: from spatial FFT to wavenumber spectrum

**Relation to `field_scalar_intro.ipynb`**: That notebook introduces the
`ScalarField` data structure.  This tutorial focuses on **advanced spatial
analysis** — coherence maps, time-delay estimation, and wavenumber spectra —
building on those foundations.
"""),

    md("## Setup"),

    code("""\
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from gwexpy.fields import ScalarField, VectorField
from gwexpy.fields.demo import make_demo_scalar_field, make_propagating_gaussian
from gwexpy.fields.signal import (
    spectral_density, coherence_map, time_delay_map, compute_xcorr,
)
"""),

    md("""\
## 1. Synthetic Beam Profile Data

We create a 4-D `ScalarField` representing the optical intensity of a
laser beam measured at 64 spatial points across a 1 m × 1 m mirror surface
at 100 time steps.

The beam has a Gaussian profile with a slow random walk in centroid position
— simulating thermal drift of the optic.
"""),

    code("""\
rng = np.random.default_rng(42)

nt = 100    # time steps
nx = ny = 32   # spatial grid points per axis
dt = 0.1 * u.s
dx = dy = 0.03 * u.m   # 3 cm per pixel (1 m aperture)

# Beam centroid drifts slowly (thermal / suspension drift)
cx = rng.normal(0, 0.05, size=nt).cumsum() * 0.002   # metres
cy = rng.normal(0, 0.05, size=nt).cumsum() * 0.002

x = (np.arange(nx) - nx//2) * dx.value   # metres
y = (np.arange(ny) - ny//2) * dy.value

# Build intensity field I(t, x, y) = Gaussian + noise
XX, YY = np.meshgrid(x, y, indexing="ij")   # (nx, ny)
beam_radius = 0.12   # 12 cm beam radius (1/e^2)

data = np.empty((nt, nx, ny, 1))   # 4-D: (t, x, y, z=1 for 2-D problem)
for i in range(nt):
    I = np.exp(-2 * ((XX - cx[i])**2 + (YY - cy[i])**2) / beam_radius**2)
    I += rng.normal(0, 0.02, size=(nx, ny))   # detector noise
    data[i, :, :, 0] = I

sf = ScalarField(
    data,
    unit=u.W / u.m**2,
    axis0=np.arange(nt) * dt.value,   # time axis [s]
    axis1=x,                           # x axis [m]
    axis2=y,                           # y axis [m]
    axis3=np.array([0.0]),             # z = 0
    axis_names=["time", "x", "y", "z"],
    axis0_domain="time",
    space_domain="real",
)

print(f"ScalarField shape: {sf.data.shape}")
print(f"Axes: t={nt} steps, x={nx}, y={ny}, z=1")
print(f"Units: {sf.unit}")
"""),

    md("""\
## 2. Mean Beam Profile and Variance Map

Averaging over time reveals the mean beam shape.
The variance map shows where intensity fluctuates most — typically at
the beam edge where pointing jitter is amplified.
"""),

    code("""\
mean_profile = sf.data[:, :, :, 0].mean(axis=0)    # (nx, ny)
var_map      = sf.data[:, :, :, 0].var(axis=0)     # (nx, ny)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

im1 = ax1.imshow(mean_profile.T, origin="lower",
                 extent=[x[0], x[-1], y[0], y[-1]], cmap="inferno")
ax1.set_title("Mean Beam Profile")
ax1.set_xlabel("x [m]")
ax1.set_ylabel("y [m]")
plt.colorbar(im1, ax=ax1, label="Mean intensity [W/m²]")

im2 = ax2.imshow(var_map.T, origin="lower",
                 extent=[x[0], x[-1], y[0], y[-1]], cmap="viridis")
ax2.set_title("Intensity Variance Map")
ax2.set_xlabel("x [m]")
plt.colorbar(im2, ax=ax2, label="Variance [(W/m²)²]")

plt.suptitle("Beam Profile Statistics (100 time steps)")
plt.tight_layout()
plt.show()
print(f"Peak mean intensity : {mean_profile.max():.3f} W/m²")
print(f"Peak variance       : {var_map.max():.4f}  (edge > centre as expected)")
"""),

    md("""\
## 3. Spatial PSD — Wavenumber Spectrum

`spectral_density(axis='x')` computes the power spectral density along the
spatial x-axis.  The wavenumber spectrum reveals the spatial frequency content
of the beam — a Gaussian beam has a Gaussian wavenumber spectrum.
"""),

    code("""\
# PSD along x at the beam centre (y = ny//2)
centre_slice = ScalarField(
    sf.data[:, :, ny//2:ny//2+1, :],
    unit=sf.unit,
    axis0=sf.axis0,
    axis1=sf.axis1,
    axis2=sf.axis2[ny//2:ny//2+1],
    axis3=sf.axis3,
    axis_names=sf.axis_names,
)

# Temporal PSD at each spatial x position
psd_t = spectral_density(centre_slice, axis=0, method="welch", fftlength=20.0)

# Wavenumber PSD averaged over time
kx = np.fft.rfftfreq(nx, dx.value)   # spatial frequencies [1/m = m^-1]
spat_fft = np.fft.rfft(sf.data[:, :, ny//2, 0], axis=1)   # (nt, nx//2+1)
spat_psd  = np.mean(np.abs(spat_fft)**2, axis=0) / (nx**2 * dx.value)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Temporal PSD at x = centre
centre_x_idx = nx // 2
if hasattr(psd_t, "data"):
    ax1.semilogy(psd_t.axis0 if hasattr(psd_t, "axis0") else
                 np.fft.rfftfreq(nt, dt.value),
                 np.abs(psd_t.data[:, centre_x_idx, 0, 0]),
                 color="steelblue", lw=1.5)
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("PSD [(W/m²)²/Hz]")
    ax1.set_title("Temporal PSD at beam centre")
    ax1.grid(True, which="both", alpha=0.4)

# Wavenumber spectrum
ax2.semilogy(kx[1:], spat_psd[1:], color="darkorange", lw=1.5)
ax2.set_xlabel("Spatial frequency kx [m⁻¹]")
ax2.set_ylabel("Spatial PSD")
ax2.set_title("Wavenumber Spectrum (y = centre)")
ax2.grid(True, which="both", alpha=0.4)
# Gaussian beam prediction: PSD ~ exp(-pi^2 * beam_radius^2 * kx^2 / 2)
kx_model = kx[1:50]
psd_gauss = spat_psd[1] * np.exp(-(np.pi * beam_radius * kx_model)**2 / 2)
ax2.semilogy(kx_model, psd_gauss, "--", color="tomato", lw=1.5,
             label=f"Gaussian model (r={beam_radius} m)")
ax2.legend()

plt.tight_layout()
plt.show()
"""),

    md("""\
## 4. Wavefield Propagation and Time-Delay Map

Now we switch to a seismic wavefield scenario: a propagating Gaussian
pulse measured at a 2-D accelerometer array.  The `time_delay_map()`
function estimates the arrival time of the pulse at each sensor relative
to a reference sensor — giving the apparent propagation velocity.
"""),

    code("""\
# Use the built-in demo function for a propagating Gaussian
wave = make_propagating_gaussian(
    nt=200, nx=16, ny=16, nz=1,
    dt=0.005 * u.s,           # 5 ms sample interval
    dx=0.5 * u.m,             # 0.5 m sensor spacing
    speed=2.5,                # propagation speed [m/s]
    direction=(1.0, 0.5),     # (vx, vy) direction vector (normalised internally)
    amplitude=1.0,
    width=1.5,                # pulse width [m]
    seed=7,
)

print(f"Wavefield shape: {wave.data.shape}   [t, x, y, z]")
print(f"Time axis range: {wave.axis0[0]:.3f} – {wave.axis0[-1]:.3f} s")

# Compute time-delay map relative to sensor at (0, 0)
# Returns a ScalarField with shape (1, nx, ny, 1)
td_map = time_delay_map(wave, ref_indices=(0, 0, 0), axis=0)
td_data = td_map.data[0, :, :, 0]   # (nx, ny)

# Infer propagation velocity from linear fit to delay vs. distance
x_w = wave.axis1
y_w = wave.axis2
XX_w, YY_w = np.meshgrid(x_w, y_w, indexing="ij")
dist = np.sqrt(XX_w**2 + YY_w**2).ravel()
delay = td_data.ravel()
valid = dist > 0.01
slope = np.polyfit(dist[valid], delay[valid], 1)[0]
speed_est = 1.0 / slope if abs(slope) > 1e-6 else np.nan

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Time-delay map
im1 = ax1.imshow(td_data.T, origin="lower", cmap="RdBu_r",
                 extent=[x_w[0], x_w[-1], y_w[0], y_w[-1]])
ax1.set_title("Time-Delay Map (ref: [0,0])")
ax1.set_xlabel("x [m]")
ax1.set_ylabel("y [m]")
plt.colorbar(im1, ax=ax1, label="Arrival delay [s]")

# Delay vs. distance scatter
ax2.scatter(dist[valid], delay[valid], s=20, alpha=0.6, color="steelblue")
d_line = np.linspace(0, dist.max(), 100)
ax2.plot(d_line, d_line * slope, color="tomato", lw=2,
         label=f"Fit: v = {speed_est:.2f} m/s")
ax2.set_xlabel("Distance from reference [m]")
ax2.set_ylabel("Time delay [s]")
ax2.set_title("Delay vs. Distance → Propagation Velocity")
ax2.legend()
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()
print(f"Estimated propagation speed: {speed_est:.2f} m/s  (true: 2.50 m/s)")
"""),

    md("""\
## 5. Coherence Map

`coherence_map()` computes the coherence between each spatial location and
a reference location as a function of frequency.  High coherence at a
specific frequency indicates correlated oscillation — e.g. a common
seismic mode or an optical cavity resonance seen across multiple positions.
"""),

    code("""\
# Coherence map at 1 Hz relative to reference sensor
coh = coherence_map(wave, ref_indices=(0, 0, 0), axis=0,
                    fftlength=0.5, overlap=0.25)

# coh has shape (n_freq, nx, ny, 1); pick a frequency bin near 1 Hz
freqs_coh = np.fft.rfftfreq(int(0.5 / 0.005), 0.005)   # fftlength / dt
target_hz  = 2.0
bin_idx    = np.argmin(np.abs(freqs_coh - target_hz))
coh_slice  = np.abs(coh.data[bin_idx, :, :, 0])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(coh_slice.T, origin="lower", vmin=0, vmax=1,
               extent=[x_w[0], x_w[-1], y_w[0], y_w[-1]], cmap="viridis")
ax.set_title(f"Coherence Map at {freqs_coh[bin_idx]:.2f} Hz\n(ref: [0, 0])")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
plt.colorbar(im, ax=ax, label="Coherence")
plt.tight_layout()
plt.show()
print(f"Mean coherence at {freqs_coh[bin_idx]:.2f} Hz: {coh_slice.mean():.3f}")
"""),

    md("""\
## 6. VectorField: Gradient and Norm

For physical fields with vector character (e.g. magnetic field, velocity),
`VectorField` stores multiple `ScalarField` components.  Here we compute
the beam intensity gradient — useful for wavefront sensing.
"""),

    code("""\
# Intensity gradient: dI/dx and dI/dy from the mean beam profile
grad_x = np.gradient(mean_profile, dx.value, axis=0)   # (nx, ny)
grad_y = np.gradient(mean_profile, dy.value, axis=1)

def make_static_sf(arr2d, x_ax, y_ax, unit):
    d = arr2d[np.newaxis, :, :, np.newaxis]   # (1, nx, ny, 1)
    return ScalarField(d, unit=unit,
                       axis0=np.array([0.0]),
                       axis1=x_ax, axis2=y_ax, axis3=np.array([0.0]),
                       axis_names=["time", "x", "y", "z"])

sf_gx = make_static_sf(grad_x, x, y, u.W / u.m**3)
sf_gy = make_static_sf(grad_y, x, y, u.W / u.m**3)

vf = VectorField(components={"x": sf_gx, "y": sf_gy})
grad_norm = vf.norm().data[0, :, :, 0]   # (nx, ny)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
titles = ["dI/dx", "dI/dy", "‖∇I‖"]
arrays = [grad_x, grad_y, grad_norm]
cmaps  = ["RdBu_r", "RdBu_r", "hot"]
for ax, arr, title, cmap in zip(axes, arrays, titles, cmaps):
    im = ax.imshow(arr.T, origin="lower",
                   extent=[x[0], x[-1], y[0], y[-1]], cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    plt.colorbar(im, ax=ax)

axes[0].set_ylabel("y [m]")
plt.suptitle("Beam Intensity Gradient (VectorField)")
plt.tight_layout()
plt.show()
print("VectorField components:", list(vf.components.keys()))
print("Gradient norm max:", f"{grad_norm.max():.3f} W/m³")
"""),

    md("""\
## Summary

| Analysis | API | Use case |
|----------|-----|----------|
| Mean / variance profile | `sf.data.mean(axis=0)` | Beam centring, stability |
| Temporal PSD | `spectral_density(sf, axis=0)` | Frequency content at each pixel |
| Wavenumber spectrum | `np.fft.rfft` + `ScalarField` | Spatial frequency content |
| Time-delay map | `time_delay_map(sf, ref_indices, axis)` | Propagation velocity |
| Coherence map | `coherence_map(sf, ref_indices, axis, fftlength)` | Correlated oscillations |
| Gradient / norm | `VectorField(components={...}).norm()` | Wavefront sensing |

**Applications at KAGRA / LIGO:**
- Beam profile monitoring and pointing drift correction
- Seismic wavefield characterisation with distributed accelerometer arrays
- Environmental field mapping (magnetic, acoustic) for noise hunting
- Mirror surface deformation from displacement sensor arrays
"""),
]


# ---------------------------------------------------------------------------
# Japanese notebook
# ---------------------------------------------------------------------------

JA_CELLS = [
    md("""\
# 多次元フィールド解析：ビームプロファイルと空間統計

`gwexpy.fields` モジュールは gwexpy を **4 次元時空間データ**
`(時間, x, y, z)` に拡張します。これにより、時間的・空間的両方の構造を持つ
現象を解析できます：

- **ミラー面上で測定された光学ビームプロファイル**
- **分散加速度計アレイによる地震波動場マップ**
- **検出器周辺の環境場マップ**（磁場、音響、温度）

**このチュートリアルで学ぶこと：**
1. `ScalarField` で合成ビームと波動場データを作成する
2. 空間統計：平均プロファイル、分散マップ、空間 PSD
3. コヒーレンスマップ：空間的にどれだけ相関があるか
4. 時間遅延マップ：信号伝播速度の推定
5. `VectorField` 演算：勾配、回転（カール）、ノルム
6. k 空間解析：空間 FFT から波数スペクトルへ

**`field_scalar_intro.ipynb` との関係：** 基礎チュートリアルは `ScalarField`
データ構造を導入します。本チュートリアルは**高度な空間解析**
（コヒーレンスマップ、時間遅延推定、波数スペクトル）に特化しています。
"""),

    md("## セットアップ"),

    code("""\
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from gwexpy.fields import ScalarField, VectorField
from gwexpy.fields.demo import make_demo_scalar_field, make_propagating_gaussian
from gwexpy.fields.signal import (
    spectral_density, coherence_map, time_delay_map, compute_xcorr,
)
"""),

    md("""\
## 1. 合成ビームプロファイルデータの生成

1 m × 1 m のミラー面上の 32×32 点で測定されたレーザービームの
光強度を表す 4 次元 `ScalarField` を作成します。
熱ドリフトによるビーム重心のゆっくりした動きをシミュレートします。
"""),

    code("""\
rng = np.random.default_rng(42)

nt = 100
nx = ny = 32
dt = 0.1 * u.s
dx = dy = 0.03 * u.m

cx = rng.normal(0, 0.05, size=nt).cumsum() * 0.002
cy = rng.normal(0, 0.05, size=nt).cumsum() * 0.002

x = (np.arange(nx) - nx//2) * dx.value
y = (np.arange(ny) - ny//2) * dy.value
XX, YY = np.meshgrid(x, y, indexing="ij")
beam_radius = 0.12

data = np.empty((nt, nx, ny, 1))
for i in range(nt):
    I = np.exp(-2 * ((XX - cx[i])**2 + (YY - cy[i])**2) / beam_radius**2)
    I += rng.normal(0, 0.02, size=(nx, ny))
    data[i, :, :, 0] = I

sf = ScalarField(
    data, unit=u.W / u.m**2,
    axis0=np.arange(nt) * dt.value,
    axis1=x, axis2=y, axis3=np.array([0.0]),
    axis_names=["time", "x", "y", "z"],
    axis0_domain="time", space_domain="real",
)

print(f"ScalarField 形状: {sf.data.shape}")
print(f"軸: t={nt}, x={nx}, y={ny}, z=1")
print(f"単位: {sf.unit}")
"""),

    md("## 2. 平均ビームプロファイルと分散マップ"),

    code("""\
mean_profile = sf.data[:, :, :, 0].mean(axis=0)
var_map      = sf.data[:, :, :, 0].var(axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

im1 = ax1.imshow(mean_profile.T, origin="lower",
                 extent=[x[0], x[-1], y[0], y[-1]], cmap="inferno")
ax1.set_title("平均ビームプロファイル")
ax1.set_xlabel("x [m]")
ax1.set_ylabel("y [m]")
plt.colorbar(im1, ax=ax1, label="平均強度 [W/m²]")

im2 = ax2.imshow(var_map.T, origin="lower",
                 extent=[x[0], x[-1], y[0], y[-1]], cmap="viridis")
ax2.set_title("強度分散マップ")
ax2.set_xlabel("x [m]")
plt.colorbar(im2, ax=ax2, label="分散 [(W/m²)²]")

plt.suptitle("ビームプロファイル統計（100 タイムステップ）")
plt.tight_layout()
plt.show()
print(f"ピーク平均強度: {mean_profile.max():.3f} W/m²")
"""),

    md("## 3. 空間 PSD — 波数スペクトル"),

    code("""\
kx = np.fft.rfftfreq(nx, dx.value)
spat_fft = np.fft.rfft(sf.data[:, :, ny//2, 0], axis=1)
spat_psd  = np.mean(np.abs(spat_fft)**2, axis=0) / (nx**2 * dx.value)

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(kx[1:], spat_psd[1:], color="darkorange", lw=1.5)
kx_model = kx[1:50]
psd_gauss = spat_psd[1] * np.exp(-(np.pi * beam_radius * kx_model)**2 / 2)
ax.semilogy(kx_model, psd_gauss, "--", color="tomato", lw=1.5,
            label=f"ガウシアンモデル (r={beam_radius} m)")
ax.set_xlabel("空間周波数 kx [m⁻¹]")
ax.set_ylabel("空間 PSD")
ax.set_title("波数スペクトル（y = 中心）")
ax.legend()
ax.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()
"""),

    md("""\
## 4. 波動場伝播と時間遅延マップ

分散加速度計アレイで測定した地震波動場シナリオです。
`time_delay_map()` は各センサーの到着時刻を参照センサーに対して推定し、
見かけの伝播速度を求めます。
"""),

    code("""\
wave = make_propagating_gaussian(
    nt=200, nx=16, ny=16, nz=1,
    dt=0.005 * u.s,
    dx=0.5 * u.m,
    speed=2.5,
    direction=(1.0, 0.5),
    amplitude=1.0, width=1.5, seed=7,
)
print(f"波動場形状: {wave.data.shape}")

td_map = time_delay_map(wave, ref_indices=(0, 0, 0), axis=0)
td_data = td_map.data[0, :, :, 0]

x_w = wave.axis1
y_w = wave.axis2
XX_w, YY_w = np.meshgrid(x_w, y_w, indexing="ij")
dist  = np.sqrt(XX_w**2 + YY_w**2).ravel()
delay = td_data.ravel()
valid = dist > 0.01
slope = np.polyfit(dist[valid], delay[valid], 1)[0]
speed_est = 1.0 / slope if abs(slope) > 1e-6 else np.nan

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

im1 = ax1.imshow(td_data.T, origin="lower", cmap="RdBu_r",
                 extent=[x_w[0], x_w[-1], y_w[0], y_w[-1]])
ax1.set_title("時間遅延マップ（参照: [0,0]）")
ax1.set_xlabel("x [m]")
ax1.set_ylabel("y [m]")
plt.colorbar(im1, ax=ax1, label="到着遅延 [s]")

d_line = np.linspace(0, dist.max(), 100)
ax2.scatter(dist[valid], delay[valid], s=20, alpha=0.6, color="steelblue")
ax2.plot(d_line, d_line * slope, color="tomato", lw=2,
         label=f"フィット: v = {speed_est:.2f} m/s")
ax2.set_xlabel("参照点からの距離 [m]")
ax2.set_ylabel("時間遅延 [s]")
ax2.set_title("遅延 vs. 距離 → 伝播速度")
ax2.legend()
ax2.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
print(f"推定伝播速度: {speed_est:.2f} m/s  （真値: 2.50 m/s）")
"""),

    md("## 5. コヒーレンスマップ"),

    code("""\
coh = coherence_map(wave, ref_indices=(0, 0, 0), axis=0,
                    fftlength=0.5, overlap=0.25)

freqs_coh = np.fft.rfftfreq(int(0.5 / 0.005), 0.005)
target_hz  = 2.0
bin_idx    = np.argmin(np.abs(freqs_coh - target_hz))
coh_slice  = np.abs(coh.data[bin_idx, :, :, 0])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(coh_slice.T, origin="lower", vmin=0, vmax=1,
               extent=[x_w[0], x_w[-1], y_w[0], y_w[-1]], cmap="viridis")
ax.set_title(f"{freqs_coh[bin_idx]:.2f} Hz のコヒーレンスマップ（参照: [0,0]）")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
plt.colorbar(im, ax=ax, label="コヒーレンス")
plt.tight_layout()
plt.show()
print(f"{freqs_coh[bin_idx]:.2f} Hz での平均コヒーレンス: {coh_slice.mean():.3f}")
"""),

    md("## 6. VectorField：勾配とノルム"),

    code("""\
grad_x = np.gradient(mean_profile, dx.value, axis=0)
grad_y = np.gradient(mean_profile, dy.value, axis=1)

def make_static_sf(arr2d, x_ax, y_ax, unit):
    d = arr2d[np.newaxis, :, :, np.newaxis]
    return ScalarField(d, unit=unit,
                       axis0=np.array([0.0]),
                       axis1=x_ax, axis2=y_ax, axis3=np.array([0.0]),
                       axis_names=["time", "x", "y", "z"])

sf_gx = make_static_sf(grad_x, x, y, u.W / u.m**3)
sf_gy = make_static_sf(grad_y, x, y, u.W / u.m**3)
vf = VectorField(components={"x": sf_gx, "y": sf_gy})
grad_norm = vf.norm().data[0, :, :, 0]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, arr, title, cmap in zip(axes,
    [grad_x, grad_y, grad_norm], ["dI/dx", "dI/dy", "‖∇I‖"],
    ["RdBu_r", "RdBu_r", "hot"]):
    im = ax.imshow(arr.T, origin="lower",
                   extent=[x[0], x[-1], y[0], y[-1]], cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    plt.colorbar(im, ax=ax)

axes[0].set_ylabel("y [m]")
plt.suptitle("ビーム強度勾配（VectorField）")
plt.tight_layout()
plt.show()
print("VectorField 成分:", list(vf.components.keys()))
print("勾配ノルム最大値:", f"{grad_norm.max():.3f} W/m³")
"""),

    md("""\
## まとめ

| 解析 | API | ユースケース |
|------|-----|------------|
| 平均・分散プロファイル | `sf.data.mean(axis=0)` | ビーム安定性の評価 |
| 時間 PSD | `spectral_density(sf, axis=0)` | 各ピクセルの周波数成分 |
| 波数スペクトル | `np.fft.rfft` + `ScalarField` | 空間周波数内容 |
| 時間遅延マップ | `time_delay_map(sf, ref_indices, axis)` | 伝播速度の推定 |
| コヒーレンスマップ | `coherence_map(sf, ref_indices, axis, fftlength)` | 相関振動の特定 |
| 勾配・ノルム | `VectorField(components={...}).norm()` | 波面センシング |

**KAGRA / LIGO での応用：**
- ビームプロファイル監視とポインティングドリフト補正
- 分散加速度計アレイによる地震波動場特性評価
- ノイズハンティングのための環境場マッピング
- 変位センサーアレイからのミラー表面変形解析
"""),
]


def write_nb(cells, path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"Written: {path}")


if __name__ == "__main__":
    root = Path(__file__).parents[2]
    write_nb(EN_CELLS, root / "docs/web/en/user_guide/tutorials/advanced_field_analysis.ipynb")
    write_nb(JA_CELLS, root / "docs/web/ja/user_guide/tutorials/advanced_field_analysis.ipynb")
    print("Done.")
