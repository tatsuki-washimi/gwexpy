"""Generate DTTXML calibration tutorial notebooks (EN + JA)."""

import base64
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helper: build a minimal DTT XML in memory
# ---------------------------------------------------------------------------

def _b64_float32(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


def _b64_complex64(arr: np.ndarray) -> str:
    """Interleave real/imag as float32 pairs (DTT floatComplex format)."""
    c = arr.astype(np.complex64)
    interleaved = np.empty(len(c) * 2, dtype=np.float32)
    interleaved[0::2] = c.real
    interleaved[1::2] = c.imag
    return base64.b64encode(interleaved.tobytes()).decode()


def make_synthetic_dttxml(path: str) -> None:
    """Write a minimal valid DTT XML file with synthetic TF/PSD/COH data."""
    N = 512
    f0 = 0.0
    df = 1.0
    freqs = np.arange(N) * df + f0  # 0–511 Hz

    # --- Synthetic resonant system (f_res=100 Hz, Q=30) ---
    f_res, Q, gain = 100.0, 30.0, 1.0
    tf = gain * f_res**2 / (f_res**2 - freqs**2 + 1j * freqs * f_res / Q)
    tf[0] = tf[1]  # avoid DC singularity

    # Coherence: high near resonance, decreasing elsewhere
    coh = np.exp(-((freqs - f_res) / 30.0) ** 2) * 0.98 + 0.01

    # PSD of input
    psd_input = np.ones(N, dtype=np.float32) * 1e-10

    def param(name, value, typ="string"):
        return f'      <Param Name="{name}" Type="{typ}">{value}</Param>'

    def psd_block(channel, data, f0_, df_, N_, t0=1000000000):
        enc = _b64_float32(data)
        return f"""  <LIGO_LW Type="Spectrum">
{param("ChannelA", channel)}
{param("Subtype", "1")}
{param("f0", str(f0_))}
{param("df", str(df_))}
{param("N", str(N_))}
{param("BUnit", "1/Hz")}
    <Time Name="t0">{t0}</Time>
    <Array Type="float">
      <Dim>{N_}</Dim>
      <Stream Encoding="LittleEndian,base64">{enc}</Stream>
    </Array>
  </LIGO_LW>"""

    def tf_block(ch_a, ch_b, data, f0_, df_, N_, t0=1000000000):
        enc = _b64_complex64(data)
        return f"""  <LIGO_LW Type="Spectrum">
{param("ChannelA", ch_a)}
{param("ChannelB", ch_b)}
{param("Subtype", "3")}
{param("f0", str(f0_))}
{param("df", str(df_))}
{param("N", str(N_))}
{param("BUnit", "1")}
    <Time Name="t0">{t0}</Time>
    <Array Type="floatComplex">
      <Dim>{N_}</Dim>
      <Stream Encoding="LittleEndian,base64">{enc}</Stream>
    </Array>
  </LIGO_LW>"""

    def coh_block(ch_a, ch_b, data, f0_, df_, N_, t0=1000000000):
        enc = _b64_float32(data)
        return f"""  <LIGO_LW Type="Spectrum">
{param("ChannelA", ch_a)}
{param("ChannelB", ch_b)}
{param("Subtype", "2")}
{param("f0", str(f0_))}
{param("df", str(df_))}
{param("N", str(N_))}
{param("BUnit", "1")}
    <Time Name="t0">{t0}</Time>
    <Array Type="float">
      <Dim>{N_}</Dim>
      <Stream Encoding="LittleEndian,base64">{enc}</Stream>
    </Array>
  </LIGO_LW>"""

    xml = "<?xml version='1.0' encoding='utf-8'?>\n<LIGO_LW>\n"
    xml += psd_block("K1:SUS-ITMX_EXCITATION", psd_input, f0, df, N)
    xml += "\n"
    xml += tf_block("K1:SUS-ITMX_DISP_DQ", "K1:SUS-ITMX_EXCITATION",
                    tf, f0, df, N)
    xml += "\n"
    xml += coh_block("K1:SUS-ITMX_DISP_DQ", "K1:SUS-ITMX_EXCITATION",
                     coh, f0, df, N)
    xml += "\n</LIGO_LW>\n"

    Path(path).write_text(xml)


# ---------------------------------------------------------------------------
# Notebook building helpers
# ---------------------------------------------------------------------------

def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"md_{abs(hash(source)) % 10**8:08x}",
        "metadata": {},
        "source": source,
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"cd_{abs(hash(source)) % 10**8:08x}",
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def notebook(cells: list, lang: str = "python") -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# English notebook
# ---------------------------------------------------------------------------

EN_CELLS = [
    md("""\
# Reading and Using DTT XML Files with gwexpy

[DTT (Diagnostic Tool for Transfer functions)](https://dtt.ligo.org/) is the
standard measurement tool used at gravitational-wave detector sites to acquire
transfer functions, power spectral densities, and coherence functions.
Measurement results are saved as **DTT XML** files.

This tutorial shows how to:
1. Inspect channels stored in a DTT XML file with `extract_xml_channels()`
2. Load all measurement products with `load_dttxml_products()`
3. Visualise a **Bode plot** (magnitude + phase) and **coherence**
4. Fit a resonant-system model to the transfer function

This notebook complements `case_transfer_function.ipynb`, which demonstrates
TF estimation from raw time series.  Here the data is already the *result* of
a measurement saved by DTT.
"""),

    md("## Setup"),

    code("""\
import numpy as np
import matplotlib.pyplot as plt

from gwexpy.io.dttxml_common import extract_xml_channels, load_dttxml_products
from gwexpy.frequencyseries import FrequencySeries
"""),

    md("""\
## 1. Create a Synthetic DTT XML File

A real workflow starts from an existing `.xml` file produced by DTT.
For reproducibility we generate a synthetic file that mimics the structure
of real KAGRA suspension transfer-function measurements.

The file contains three measurement products:
| Product | Type | Description |
|---------|------|-------------|
| PSD | real | Input excitation PSD |
| TF | complex | Displacement / Excitation transfer function |
| COH | real | Coherence between displacement and excitation |
"""),

    code("""\
import base64, struct, pathlib

def make_synthetic_dttxml(path):
    \"\"\"Generate a minimal valid DTT XML with synthetic resonant TF data.\"\"\"
    N, f0_hz, df = 512, 0.0, 1.0
    freqs = np.arange(N) * df + f0_hz

    # Resonant system: f_res=100 Hz, Q=30
    f_res, Q = 100.0, 30.0
    tf_data = f_res**2 / (f_res**2 - freqs**2 + 1j * freqs * f_res / Q)
    tf_data[0] = tf_data[1]

    coh_data = np.exp(-((freqs - f_res) / 30.0) ** 2) * 0.97 + 0.02
    psd_data = np.ones(N, dtype=np.float32) * 1e-10

    def b64f32(arr):
        return base64.b64encode(arr.astype(np.float32).tobytes()).decode()

    def b64c64(arr):
        c = arr.astype(np.complex64)
        v = np.empty(len(c) * 2, dtype=np.float32)
        v[0::2], v[1::2] = c.real, c.imag
        return base64.b64encode(v.tobytes()).decode()

    def p(name, val):
        return f'    <Param Name="{name}" Type="string">{val}</Param>'

    def block(attrs, data, dtype, f0_, df_, N_):
        lines = ['  <LIGO_LW Type="Spectrum">']
        for k, v in attrs.items():
            lines.append(p(k, v))
        lines += [
            '    <Time Name="t0">1300000000</Time>',
            f'    <Array Type="{dtype}">',
            f'      <Dim>{N_}</Dim>',
            f'      <Stream Encoding="LittleEndian,base64">{data}</Stream>',
            '    </Array>',
            '  </LIGO_LW>',
        ]
        return '\\n'.join(lines)

    xml = "<?xml version='1.0' encoding='utf-8'?>\\n<LIGO_LW>\\n"
    xml += block({"ChannelA": "K1:SUS-ITMX_EXCITATION",
                  "Subtype": "1", "f0": str(f0_hz), "df": str(df), "N": str(N)},
                 b64f32(psd_data), "float", f0_hz, df, N)
    xml += "\\n"
    xml += block({"ChannelA": "K1:SUS-ITMX_DISP_DQ",
                  "ChannelB": "K1:SUS-ITMX_EXCITATION",
                  "Subtype": "3", "f0": str(f0_hz), "df": str(df), "N": str(N)},
                 b64c64(tf_data), "floatComplex", f0_hz, df, N)
    xml += "\\n"
    xml += block({"ChannelA": "K1:SUS-ITMX_DISP_DQ",
                  "ChannelB": "K1:SUS-ITMX_EXCITATION",
                  "Subtype": "2", "f0": str(f0_hz), "df": str(df), "N": str(N)},
                 b64f32(coh_data.astype(np.float32)), "float", f0_hz, df, N)
    xml += "\\n</LIGO_LW>\\n"

    pathlib.Path(path).write_text(xml)
    print(f"Wrote {path}")

xml_path = "/tmp/kagra_sus_itmx.xml"
make_synthetic_dttxml(xml_path)
"""),

    md("""\
## 2. Inspect Channels

`extract_xml_channels()` returns a list of channel names and their active
status without loading the full data — useful for a quick scan of large files.
"""),

    code("""\
channels = extract_xml_channels(xml_path)
print(f"Found {len(channels)} channel entries:")
for ch in channels:
    status = "active" if ch["active"] else "inactive"
    print(f"  [{status}] {ch['name']}")
"""),

    md("""\
## 3. Load Measurement Products

`load_dttxml_products()` returns a dictionary keyed by product type.
Each product is itself a dict with numpy arrays for data and frequency axis.

```
{
  "PSD": [{"freq": ndarray, "data": ndarray, "channel_a": str, ...}],
  "TF":  [{"freq": ndarray, "data": ndarray (complex), ...}],
  "COH": [{"freq": ndarray, "data": ndarray, ...}],
  ...
}
```

Pass `native=True` to use gwexpy's built-in XML parser instead of the
`dttxml` package — this is recommended when the dttxml package is unavailable
or when you encounter the known phase-loss bug for complex responses.
"""),

    code("""\
products = load_dttxml_products(xml_path, native=True)

print("Product types found:", list(products.keys()))
for ptype, items in products.items():
    print(f"  {ptype}: {len(items)} measurement(s)")
    for item in items:
        print(f"    ChannelA={item.get('channel_a', '?')}"
              f"  N={len(item['freq'])}  df={item['freq'][1]-item['freq'][0]:.3f} Hz")
"""),

    md("""\
## 4. Bode Plot — Transfer Function

Extract the TF product and wrap the data in a `FrequencySeries` to access
gwexpy's plotting and fitting methods.
"""),

    code("""\
# Extract the first TF measurement
tf_prod = products["TF"][0]
freqs = tf_prod["freq"]           # frequency axis [Hz]
tf_data = tf_prod["data"]         # complex transfer function

# Build FrequencySeries (unit: dimensionless for displacement/force TF)
tf_fs = FrequencySeries(tf_data, frequencies=freqs, unit="m/N", name="ITMX TF")

# --- Bode plot ---
fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

ax_mag.loglog(freqs[1:], np.abs(tf_data[1:]), color="steelblue", lw=1.5)
ax_mag.set_ylabel("Magnitude [m/N]")
ax_mag.set_title("KAGRA ITMX Suspension: Transfer Function (DTT XML)")
ax_mag.grid(True, which="both", alpha=0.4)

phase_deg = np.angle(tf_data[1:], deg=True)
ax_ph.semilogx(freqs[1:], phase_deg, color="darkorange", lw=1.5)
ax_ph.set_ylabel("Phase [deg]")
ax_ph.set_xlabel("Frequency [Hz]")
ax_ph.set_ylim(-200, 200)
ax_ph.grid(True, which="both", alpha=0.4)

plt.tight_layout()
plt.show()
"""),

    md("""\
## 5. Coherence

The coherence indicates the signal-to-noise quality of the measurement.
Values close to 1 mean the output is well explained by the input;
values below ~0.9 suggest the measurement is unreliable at those frequencies.
"""),

    code("""\
coh_prod = products["COH"][0]
coh_freqs = coh_prod["freq"]
coh_data  = coh_prod["data"]

fig, ax = plt.subplots(figsize=(9, 3))
ax.semilogx(coh_freqs[1:], coh_data[1:], color="seagreen", lw=1.5)
ax.axhline(0.9, color="red", ls="--", lw=1, label="Coherence = 0.9")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Coherence")
ax.set_title("Coherence: ITMX Displacement vs. Excitation")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()
"""),

    md("""\
## 6. Fitting a Resonant-System Model

Transfer functions near a mechanical resonance follow a second-order response:

$$H(f) = \\frac{A \\, f_0^2}{f_0^2 - f^2 + i\\, f f_0 / Q}$$

We use `gwexpy.fitting` to extract $f_0$ (resonance frequency), $Q$ (quality
factor), and $A$ (gain) directly from the DTT XML data.
"""),

    code("""\
from gwexpy.fitting import fit_series

# Restrict to the frequency band where coherence > 0.9
good = coh_data > 0.9
fit_freqs = freqs[good]
fit_data  = tf_data[good]

def resonator_model(f, A, f_res, Q):
    return A * f_res**2 / (f_res**2 - f**2 + 1j * f * f_res / Q)

result = fit_series(
    fit_freqs, fit_data,
    model=resonator_model,
    p0=[1.0, 95.0, 25.0],
    method="complex_least_squares",
)
A_fit, f0_fit, Q_fit = result.params

print(f"Fitted resonance frequency : {f0_fit:.2f} Hz  (true: 100.0 Hz)")
print(f"Fitted quality factor      : {Q_fit:.1f}   (true: 30.0)")
print(f"Fitted gain                : {A_fit:.4f}")
"""),

    code("""\
# Overlay the fitted model on the Bode plot
f_model = np.linspace(1, 511, 2000)
tf_model = resonator_model(f_model, A_fit, f0_fit, Q_fit)

fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

ax_mag.loglog(freqs[1:], np.abs(tf_data[1:]),
              color="steelblue", lw=1.5, alpha=0.7, label="DTT data")
ax_mag.loglog(f_model, np.abs(tf_model),
              color="red", ls="--", lw=2, label=f"Fit: f₀={f0_fit:.1f} Hz, Q={Q_fit:.1f}")
ax_mag.set_ylabel("Magnitude [m/N]")
ax_mag.set_title("ITMX Transfer Function — DTT data vs. fitted model")
ax_mag.legend()
ax_mag.grid(True, which="both", alpha=0.4)

ax_ph.semilogx(freqs[1:], np.angle(tf_data[1:], deg=True),
               color="steelblue", lw=1.5, alpha=0.7, label="DTT data")
ax_ph.semilogx(f_model, np.angle(tf_model, deg=True),
               color="red", ls="--", lw=2, label="Fit")
ax_ph.set_ylabel("Phase [deg]")
ax_ph.set_xlabel("Frequency [Hz]")
ax_ph.set_ylim(-200, 200)
ax_ph.legend()
ax_ph.grid(True, which="both", alpha=0.4)

plt.tight_layout()
plt.show()
"""),

    md("""\
## Summary

| Step | gwexpy API | Output |
|------|-----------|--------|
| Inspect channels | `extract_xml_channels(path)` | list of `{"name", "active"}` |
| Load all products | `load_dttxml_products(path, native=True)` | dict of TF / PSD / COH arrays |
| Visualise TF | `FrequencySeries` + matplotlib | Bode plot |
| Fit resonance | `fit_series(freqs, data, model, p0)` | $f_0$, $Q$, $A$ |

**Tips for real measurements:**
- Pass `native=False` to use the `dttxml` package when installed (faster for large files).
- Use `native=True` if you observe phase flips in complex TF products.
- The `load_dttxml_products()` dict keys follow DTT's product naming:
  `"TF"`, `"STF"`, `"PSD"`, `"ASD"`, `"CSD"`, `"COH"`, `"TS"`.
"""),
]


# ---------------------------------------------------------------------------
# Japanese notebook
# ---------------------------------------------------------------------------

JA_CELLS = [
    md("""\
# gwexpy で DTT XML ファイルを読み込む

[DTT（Diagnostic Tool for Transfer functions）](https://dtt.ligo.org/)は、
重力波検出器サイトで標準的に使われる伝達関数測定ツールです。
測定結果は **DTT XML** ファイルとして保存されます。

このチュートリアルでは以下を学びます：
1. `extract_xml_channels()` でファイル内のチャネルを確認する
2. `load_dttxml_products()` で全測定量を読み込む
3. **Bode プロット**（振幅・位相）と**コヒーレンス**を可視化する
4. gwexpy のフィッティング機能で共振パラメータを推定する

`case_transfer_function.ipynb`（時系列から伝達関数を推定するチュートリアル）の
発展版として、**DTT が出力した測定済み結果**を直接扱うワークフローを示します。
"""),

    md("## セットアップ"),

    code("""\
import numpy as np
import matplotlib.pyplot as plt

from gwexpy.io.dttxml_common import extract_xml_channels, load_dttxml_products
from gwexpy.frequencyseries import FrequencySeries
"""),

    md("""\
## 1. 合成 DTT XML ファイルの作成

実際の運用では既存の `.xml` ファイルを使いますが、
ここでは KAGRA 懸架系の伝達関数測定を模した合成ファイルを生成します。

生成するファイルには以下の3種類の測定量が含まれます：

| 測定量 | 型 | 内容 |
|--------|---|------|
| PSD | 実数 | 励振入力の PSD |
| TF | 複素数 | 変位 / 励振の伝達関数 |
| COH | 実数 | 変位と励振間のコヒーレンス |
"""),

    code("""\
import base64, pathlib

def make_synthetic_dttxml(path):
    \"\"\"合成共振系データを持つ最小限の DTT XML を生成する。\"\"\"
    N, f0_hz, df = 512, 0.0, 1.0
    freqs = np.arange(N) * df + f0_hz

    # 共振系: f_res=100 Hz, Q=30
    f_res, Q = 100.0, 30.0
    tf_data = f_res**2 / (f_res**2 - freqs**2 + 1j * freqs * f_res / Q)
    tf_data[0] = tf_data[1]

    coh_data = np.exp(-((freqs - f_res) / 30.0) ** 2) * 0.97 + 0.02
    psd_data = np.ones(N, dtype=np.float32) * 1e-10

    def b64f32(arr):
        return base64.b64encode(arr.astype(np.float32).tobytes()).decode()

    def b64c64(arr):
        c = arr.astype(np.complex64)
        v = np.empty(len(c) * 2, dtype=np.float32)
        v[0::2], v[1::2] = c.real, c.imag
        return base64.b64encode(v.tobytes()).decode()

    def p(name, val):
        return f'    <Param Name="{name}" Type="string">{val}</Param>'

    def block(attrs, data, dtype, N_):
        lines = ['  <LIGO_LW Type="Spectrum">']
        for k, v in attrs.items():
            lines.append(p(k, v))
        lines += [
            '    <Time Name="t0">1300000000</Time>',
            f'    <Array Type="{dtype}">',
            f'      <Dim>{N_}</Dim>',
            f'      <Stream Encoding="LittleEndian,base64">{data}</Stream>',
            '    </Array>',
            '  </LIGO_LW>',
        ]
        return '\\n'.join(lines)

    xml = "<?xml version='1.0' encoding='utf-8'?>\\n<LIGO_LW>\\n"
    xml += block({"ChannelA": "K1:SUS-ITMX_EXCITATION",
                  "Subtype": "1", "f0": "0.0", "df": "1.0", "N": str(N)},
                 b64f32(psd_data), "float", N)
    xml += "\\n"
    xml += block({"ChannelA": "K1:SUS-ITMX_DISP_DQ",
                  "ChannelB": "K1:SUS-ITMX_EXCITATION",
                  "Subtype": "3", "f0": "0.0", "df": "1.0", "N": str(N)},
                 b64c64(tf_data), "floatComplex", N)
    xml += "\\n"
    xml += block({"ChannelA": "K1:SUS-ITMX_DISP_DQ",
                  "ChannelB": "K1:SUS-ITMX_EXCITATION",
                  "Subtype": "2", "f0": "0.0", "df": "1.0", "N": str(N)},
                 b64f32(coh_data.astype(np.float32)), "float", N)
    xml += "\\n</LIGO_LW>\\n"

    pathlib.Path(path).write_text(xml)
    print(f"書き込み完了: {path}")

xml_path = "/tmp/kagra_sus_itmx.xml"
make_synthetic_dttxml(xml_path)
"""),

    md("""\
## 2. チャネルの確認

`extract_xml_channels()` はデータ本体を読み込まずにチャネル名と
アクティブ状態だけを返します。大容量ファイルのクイックスキャンに便利です。
"""),

    code("""\
channels = extract_xml_channels(xml_path)
print(f"チャネル数: {len(channels)}")
for ch in channels:
    status = "active" if ch["active"] else "inactive"
    print(f"  [{status}] {ch['name']}")
"""),

    md("""\
## 3. 測定量の読み込み

`load_dttxml_products()` は測定量の種類をキーとする辞書を返します。

```python
{
  "PSD": [{"freq": ndarray, "data": ndarray, "channel_a": str, ...}],
  "TF":  [{"freq": ndarray, "data": ndarray (複素数), ...}],
  "COH": [{"freq": ndarray, "data": ndarray, ...}],
  ...
}
```

`native=True` を指定すると gwexpy 組み込みパーサーを使用します。
dttxml パッケージの複素数位相損失バグを回避できるため推奨です。
"""),

    code("""\
products = load_dttxml_products(xml_path, native=True)

print("取得した測定量:", list(products.keys()))
for ptype, items in products.items():
    print(f"  {ptype}: {len(items)} 件")
    for item in items:
        print(f"    ChannelA={item.get('channel_a', '?')}"
              f"  N={len(item['freq'])}  df={item['freq'][1]-item['freq'][0]:.3f} Hz")
"""),

    md("""\
## 4. Bode プロット — 伝達関数の可視化

TF データを取り出し、振幅と位相を周波数の関数としてプロットします。
"""),

    code("""\
tf_prod = products["TF"][0]
freqs   = tf_prod["freq"]
tf_data = tf_prod["data"]  # 複素数配列

fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

ax_mag.loglog(freqs[1:], np.abs(tf_data[1:]), color="steelblue", lw=1.5)
ax_mag.set_ylabel("振幅 [m/N]")
ax_mag.set_title("KAGRA ITMX 懸架系：伝達関数（DTT XML より）")
ax_mag.grid(True, which="both", alpha=0.4)

ax_ph.semilogx(freqs[1:], np.angle(tf_data[1:], deg=True),
               color="darkorange", lw=1.5)
ax_ph.set_ylabel("位相 [deg]")
ax_ph.set_xlabel("周波数 [Hz]")
ax_ph.set_ylim(-200, 200)
ax_ph.grid(True, which="both", alpha=0.4)

plt.tight_layout()
plt.show()
"""),

    md("""\
## 5. コヒーレンス

コヒーレンスは測定の信頼性指標です。1 に近いほど出力が入力で
よく説明されることを意味します。0.9 以下の周波数帯は
信頼性が低く、フィッティングの対象から外すことが推奨されます。
"""),

    code("""\
coh_prod = products["COH"][0]
coh_freqs = coh_prod["freq"]
coh_data  = coh_prod["data"]

fig, ax = plt.subplots(figsize=(9, 3))
ax.semilogx(coh_freqs[1:], coh_data[1:], color="seagreen", lw=1.5)
ax.axhline(0.9, color="red", ls="--", lw=1, label="コヒーレンス = 0.9")
ax.set_xlabel("周波数 [Hz]")
ax.set_ylabel("コヒーレンス")
ax.set_title("コヒーレンス：ITMX 変位 vs. 励振")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()
"""),

    md("""\
## 6. 共振モデルのフィッティング

機械的共振付近の伝達関数は2次系で近似できます：

$$H(f) = \\frac{A \\, f_0^2}{f_0^2 - f^2 + i\\, f f_0 / Q}$$

gwexpy のフィッティング機能を使って、
共振周波数 $f_0$、Q 値、ゲイン $A$ を DTT XML データから直接推定します。
"""),

    code("""\
from gwexpy.fitting import fit_series

# コヒーレンス > 0.9 の帯域のみでフィット
good = coh_data > 0.9
fit_freqs = freqs[good]
fit_data  = tf_data[good]

def resonator_model(f, A, f_res, Q):
    return A * f_res**2 / (f_res**2 - f**2 + 1j * f * f_res / Q)

result = fit_series(
    fit_freqs, fit_data,
    model=resonator_model,
    p0=[1.0, 95.0, 25.0],
    method="complex_least_squares",
)
A_fit, f0_fit, Q_fit = result.params

print(f"推定共振周波数: {f0_fit:.2f} Hz  （真値: 100.0 Hz）")
print(f"推定 Q 値     : {Q_fit:.1f}   （真値: 30.0）")
print(f"推定ゲイン    : {A_fit:.4f}")
"""),

    code("""\
f_model  = np.linspace(1, 511, 2000)
tf_model = resonator_model(f_model, A_fit, f0_fit, Q_fit)

fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

ax_mag.loglog(freqs[1:], np.abs(tf_data[1:]),
              color="steelblue", lw=1.5, alpha=0.7, label="DTT データ")
ax_mag.loglog(f_model, np.abs(tf_model), color="red", ls="--", lw=2,
              label=f"フィット: f₀={f0_fit:.1f} Hz, Q={Q_fit:.1f}")
ax_mag.set_ylabel("振幅 [m/N]")
ax_mag.set_title("伝達関数 — DTT データ vs. フィットモデル")
ax_mag.legend()
ax_mag.grid(True, which="both", alpha=0.4)

ax_ph.semilogx(freqs[1:], np.angle(tf_data[1:], deg=True),
               color="steelblue", lw=1.5, alpha=0.7, label="DTT データ")
ax_ph.semilogx(f_model, np.angle(tf_model, deg=True),
               color="red", ls="--", lw=2, label="フィット")
ax_ph.set_ylabel("位相 [deg]")
ax_ph.set_xlabel("周波数 [Hz]")
ax_ph.set_ylim(-200, 200)
ax_ph.legend()
ax_ph.grid(True, which="both", alpha=0.4)

plt.tight_layout()
plt.show()
"""),

    md("""\
## まとめ

| ステップ | gwexpy API | 出力 |
|---------|-----------|------|
| チャネル確認 | `extract_xml_channels(path)` | `{"name", "active"}` のリスト |
| 測定量の読み込み | `load_dttxml_products(path, native=True)` | TF / PSD / COH 辞書 |
| Bode プロット | `FrequencySeries` + matplotlib | 振幅・位相グラフ |
| 共振フィット | `fit_series(freqs, data, model, p0)` | $f_0$, $Q$, $A$ |

**実測ファイルを使うときの注意点：**
- dttxml パッケージがインストール済みなら `native=False`（デフォルト）が高速。
- 複素 TF で位相が飛ぶ場合は `native=True` を指定してください（既知バグの回避）。
- `load_dttxml_products()` の返却キー：`"TF"`, `"STF"`, `"PSD"`, `"ASD"`, `"CSD"`, `"COH"`, `"TS"`
"""),
]


# ---------------------------------------------------------------------------
# Write notebooks
# ---------------------------------------------------------------------------

def write_nb(cells, path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"Written: {path}")


if __name__ == "__main__":
    root = Path(__file__).parents[2]
    en_path = root / "docs/web/en/user_guide/tutorials/case_dttxml_calibration.ipynb"
    ja_path = root / "docs/web/ja/user_guide/tutorials/case_dttxml_calibration.ipynb"

    write_nb(EN_CELLS, en_path)
    write_nb(JA_CELLS, ja_path)
    print("Done.")
