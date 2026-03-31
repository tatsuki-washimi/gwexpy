"""Generate HDF5 provenance / reproducible metadata tutorial notebooks (EN + JA)."""

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
# Reproducible Metadata Management with HDF5

**Provenance** means recording *who did what, when, and how* so that any
analysis result can be reproduced exactly by someone else — or by yourself
six months later.

In gravitational-wave data analysis, provenance is critical because:
- Parameters (GPS times, FFT lengths, filter settings) change between runs
- Software versions affect numerical results
- Sharing results with collaborators requires a self-describing file

This tutorial shows how to use gwexpy's HDF5 interop layer together with
`h5py` attribute metadata to build **self-describing, provenance-rich
analysis archives**.

**What this tutorial covers:**
1. Saving a `TimeSeries` to HDF5 with basic metadata
2. Adding provenance attributes (software version, parameters, GPS time)
3. Storing derived products (ASD, spectrogram) with full lineage
4. Reading back and verifying provenance
5. Building a provenance-aware analysis pipeline helper
"""),

    md("## Setup"),

    code("""\
import datetime
import json
import os
import platform
import tempfile

import h5py
import numpy as np

import gwexpy
from gwexpy.timeseries import TimeSeries
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.interop.hdf5_ import to_hdf5, from_hdf5
"""),

    md("""\
## 1. Synthetic Data

We create a short DARM-like time series with a known injection for later
verification.
"""),

    code("""\
fs   = 4096.0
T    = 64.0
N    = int(T * fs)
t0   = 1_300_000_000   # GPS epoch
rng  = np.random.default_rng(0)

# Coloured noise + 50 Hz sine injection
t = np.arange(N) / fs
freqs_n = np.fft.rfftfreq(N, 1.0/fs)[1:]
asd_model = np.where(freqs_n < 50, (50/freqs_n)**3, 1.0)
fft = asd_model * np.exp(1j * rng.uniform(0, 2*np.pi, size=len(freqs_n)))
fft = np.concatenate([[0.0], fft])
noise = np.fft.irfft(fft, n=N)
inj   = 5.0 * np.sin(2*np.pi * 50.0 * t)

ts = TimeSeries(noise + inj, t0=t0, sample_rate=fs,
                name="K1:LSC-DARM_OUT_DQ", unit="ct")
print(f"TimeSeries: {len(ts)} samples @ {fs} Hz, t0={t0}")
"""),

    md("""\
## 2. Writing to HDF5 with Provenance Metadata

`to_hdf5()` saves the data array and basic metadata (t0, dt, unit, name).
We then add **provenance attributes** directly on the HDF5 dataset:
- Software versions
- Analysis parameters
- Timestamp of the run
- Host information
"""),

    code("""\
def build_provenance(params: dict) -> dict:
    return {
        "gwexpy_version":  gwexpy.__version__,
        "numpy_version":   np.__version__,
        "h5py_version":    h5py.__version__,
        "python_version":  platform.python_version(),
        "hostname":        platform.node(),
        "created_utc":     datetime.datetime.utcnow().isoformat(),
        "params_json":     json.dumps(params),   # analysis parameters as JSON
    }

# Analysis parameters (these would vary between runs)
analysis_params = {
    "fftlength":  4.0,
    "overlap":    0.5,
    "method":     "median",
    "freq_range": [10.0, 2000.0],
}

hdf5_path = tempfile.mktemp(suffix=".h5")

with h5py.File(hdf5_path, "w") as f:
    # --- Raw TimeSeries ---
    raw_grp = f.require_group("raw")
    to_hdf5(ts, raw_grp, "darm")

    dset = raw_grp["darm"]
    prov = build_provenance(analysis_params)
    for key, val in prov.items():
        dset.attrs[key] = val
    dset.attrs["description"] = "Raw DARM output, uncalibrated"

    print("Written dataset attributes:")
    for k, v in dset.attrs.items():
        print(f"  {k}: {str(v)[:60]}")

print(f"\\nHDF5 file: {hdf5_path}  ({os.path.getsize(hdf5_path)/1024:.1f} kB)")
"""),

    md("""\
## 3. Storing Derived Products with Full Lineage

Each derived product (ASD, spectrogram, filtered series) should reference
the parent dataset so the full processing chain is traceable.
"""),

    code("""\
# Compute ASD
asd = ts.asd(fftlength=analysis_params["fftlength"],
             overlap=analysis_params["overlap"],
             method=analysis_params["method"])

with h5py.File(hdf5_path, "a") as f:
    proc_grp = f.require_group("processed")

    # Save ASD frequencies and values as separate datasets
    freq_dset = proc_grp.create_dataset("asd_frequencies", data=asd.frequencies.value)
    freq_dset.attrs["unit"] = "Hz"

    asd_dset = proc_grp.create_dataset("asd_values", data=asd.value)
    asd_dset.attrs["unit"] = str(asd.unit)
    asd_dset.attrs["name"] = asd.name

    # Provenance: link back to parent + record parameters
    prov_asd = build_provenance(analysis_params)
    prov_asd["parent_dataset"] = "/raw/darm"
    prov_asd["processing_step"] = "ASD via Welch/median"
    for key, val in prov_asd.items():
        asd_dset.attrs[key] = val

    # Print structure
    def print_tree(name, obj):
        indent = "  " * name.count("/")
        attrs_summary = f"  [{len(obj.attrs)} attrs]" if hasattr(obj, "attrs") else ""
        print(f"{indent}{name.split('/')[-1]}{attrs_summary}")

    print("HDF5 file structure:")
    f.visititems(print_tree)
"""),

    md("""\
## 4. Reading Back and Verifying Provenance

A collaborator (or future-you) can open the file and immediately understand
what analysis produced each dataset, using what parameters and software.
"""),

    code("""\
with h5py.File(hdf5_path, "r") as f:
    # Reconstruct TimeSeries
    ts_loaded = from_hdf5(TimeSeries, f["raw"], "darm")

    # Read provenance from the raw dataset
    dset_attrs = dict(f["raw/darm"].attrs)
    params_loaded = json.loads(dset_attrs["params_json"])

    # Read back ASD
    freqs_loaded = f["processed/asd_frequencies"][:]
    asd_loaded   = f["processed/asd_values"][:]
    asd_prov     = dict(f["processed/asd_values"].attrs)

print("=== Reconstructed TimeSeries ===")
print(f"  name   : {ts_loaded.name}")
print(f"  t0     : {ts_loaded.t0.value}")
print(f"  samples: {len(ts_loaded)}")
print(f"  unit   : {ts_loaded.unit}")

print("\\n=== Provenance (raw/darm) ===")
for key in ["gwexpy_version", "created_utc", "hostname"]:
    print(f"  {key}: {dset_attrs[key]}")

print("\\n=== Analysis parameters ===")
for key, val in params_loaded.items():
    print(f"  {key}: {val}")

print("\\n=== ASD parent reference ===")
print(f"  parent_dataset  : {asd_prov['parent_dataset']}")
print(f"  processing_step : {asd_prov['processing_step']}")
"""),

    md("""\
## 5. Provenance-Aware Pipeline Helper

For repeated use, wrap the provenance pattern in a reusable context manager
that automatically attaches metadata to every dataset written inside it.
"""),

    code("""\
import contextlib

@contextlib.contextmanager
def provenance_file(path, params: dict, mode: str = "w"):
    with h5py.File(path, mode) as f:
        def save_ts(ts_obj, group_path: str, name: str, description: str = ""):
            grp = f.require_group(group_path)
            to_hdf5(ts_obj, grp, name)
            dset = grp[name]
            prov = build_provenance(params)
            for k, v in prov.items():
                dset.attrs[k] = v
            if description:
                dset.attrs["description"] = description
            return dset

        def save_array(arr, path_in_file: str, unit: str = "",
                       description: str = "", parent: str = ""):
            parts = path_in_file.rsplit("/", 1)
            grp_path, name = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
            grp = f.require_group(grp_path) if grp_path else f
            dset = grp.create_dataset(name, data=arr)
            if unit:        dset.attrs["unit"]        = unit
            if description: dset.attrs["description"] = description
            if parent:      dset.attrs["parent"]      = parent
            prov = build_provenance(params)
            for k, v in prov.items():
                dset.attrs[k] = v
            return dset

        yield f, save_ts, save_array

# --- Use the helper ---
pipeline_path = tempfile.mktemp(suffix="_pipeline.h5")
run_params = {"fftlength": 8.0, "overlap": 0.75, "method": "median", "run_id": "R001"}

with provenance_file(pipeline_path, run_params) as (f, save_ts, save_array):
    save_ts(ts, "input", "darm", description="Raw DARM before processing")

    asd2 = ts.asd(fftlength=run_params["fftlength"], overlap=run_params["overlap"])
    save_array(asd2.value,            "products/asd",  unit="ct/rtHz",
               description="ASD", parent="/input/darm")
    save_array(asd2.frequencies.value,"products/freqs", unit="Hz")

print(f"Pipeline archive: {pipeline_path}  ({os.path.getsize(pipeline_path)/1024:.1f} kB)")

# Verify run_id was stored
with h5py.File(pipeline_path, "r") as f:
    params_back = json.loads(f["input/darm"].attrs["params_json"])
    print(f"Verified run_id: {params_back['run_id']}")
"""),

    md("""\
## 6. Visualise the Stored ASD (Reproducibility Check)

Verify that the ASD loaded from the archive matches the in-memory result.
"""),

    code("""\
import matplotlib.pyplot as plt

with h5py.File(pipeline_path, "r") as f:
    freqs_ar = f["products/freqs"][:]
    asd_ar   = f["products/asd"][:]

fig, ax = plt.subplots(figsize=(9, 4))
ax.loglog(asd2.frequencies.value[1:], asd2.value[1:],
          color="steelblue", lw=2, label="In-memory ASD")
ax.loglog(freqs_ar[1:], asd_ar[1:],
          color="tomato", ls="--", lw=1.5, label="Loaded from HDF5")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("ASD [ct/√Hz]")
ax.set_title("Reproducibility check: in-memory vs. HDF5 archive")
ax.legend()
ax.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()

max_diff = np.max(np.abs(asd2.value[1:] - asd_ar[1:]) / (asd2.value[1:] + 1e-300))
print(f"Max relative difference: {max_diff:.2e}  (should be ~machine epsilon)")
"""),

    md("""\
## Summary

| Step | API / Pattern | Purpose |
|------|--------------|---------|
| Write TimeSeries | `to_hdf5(ts, group, path)` | Save data array + basic metadata |
| Read TimeSeries | `from_hdf5(TimeSeries, group, path)` | Reconstruct from archive |
| Add provenance | `dset.attrs[key] = value` | Software version, params, timestamp |
| Structured pipeline | `provenance_file(path, params)` context manager | Auto-attach provenance to all writes |

**Recommended provenance attributes:**

| Attribute | Content |
|-----------|---------|
| `gwexpy_version` | `gwexpy.__version__` |
| `created_utc` | `datetime.datetime.utcnow().isoformat()` |
| `params_json` | `json.dumps(analysis_params)` |
| `parent_dataset` | HDF5 path to the input dataset |
| `processing_step` | Human-readable description of what was done |

**Tips:**
- Use `h5py.File(..., "a")` to append to existing archives without overwriting.
- Store `params_json` as a JSON string so arbitrary nested parameters are preserved.
- For long pipelines, add a `pipeline_version` attribute tied to your analysis code's git hash.
"""),
]


# ---------------------------------------------------------------------------
# Japanese notebook
# ---------------------------------------------------------------------------

JA_CELLS = [
    md("""\
# HDF5 による再現可能なメタデータ管理

**プロバナンス（来歴）**とは、「誰が・何を・いつ・どのように」行ったかを記録し、
解析結果を他者（または半年後の自分）が完全に再現できるようにする概念です。

重力波データ解析においてプロバナンスが重要な理由：
- GPS 時刻・FFT 長・フィルタ設定などのパラメータがランごとに変化する
- ソフトウェアのバージョンが数値結果に影響する
- 共同研究者との結果共有には自己記述型ファイルが必要

このチュートリアルでは、gwexpy の HDF5 相互運用層と
`h5py` 属性メタデータを使って**自己記述型・プロバナンス付き解析アーカイブ**を
構築する方法を示します。

**このチュートリアルで学ぶこと：**
1. `TimeSeries` を基本メタデータ付きで HDF5 に保存する
2. プロバナンス属性（ソフトウェアバージョン、パラメータ、GPS 時刻）を追加する
3. 派生量（ASD、スペクトログラム）を完全なリネージ情報付きで保存する
4. 読み戻してプロバナンスを検証する
5. プロバナンス対応解析パイプラインヘルパーを構築する
"""),

    md("## セットアップ"),

    code("""\
import datetime
import json
import os
import platform
import tempfile

import h5py
import numpy as np

import gwexpy
from gwexpy.timeseries import TimeSeries
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.interop.hdf5_ import to_hdf5, from_hdf5
"""),

    md("## 1. 合成データの生成"),

    code("""\
fs   = 4096.0
T    = 64.0
N    = int(T * fs)
t0   = 1_300_000_000
rng  = np.random.default_rng(0)

t = np.arange(N) / fs
freqs_n = np.fft.rfftfreq(N, 1.0/fs)[1:]
asd_model = np.where(freqs_n < 50, (50/freqs_n)**3, 1.0)
fft = asd_model * np.exp(1j * rng.uniform(0, 2*np.pi, size=len(freqs_n)))
fft = np.concatenate([[0.0], fft])
noise = np.fft.irfft(fft, n=N)
inj   = 5.0 * np.sin(2*np.pi * 50.0 * t)

ts = TimeSeries(noise + inj, t0=t0, sample_rate=fs,
                name="K1:LSC-DARM_OUT_DQ", unit="ct")
print(f"TimeSeries: {len(ts)} サンプル @ {fs} Hz, t0={t0}")
"""),

    md("""\
## 2. プロバナンス属性付きで HDF5 に書き込む

`to_hdf5()` はデータ配列と基本メタデータ（t0、dt、unit、name）を保存します。
その後、HDF5 データセットに直接**プロバナンス属性**を追加します：
- ソフトウェアバージョン
- 解析パラメータ
- 実行タイムスタンプ
- ホスト情報
"""),

    code("""\
def build_provenance(params: dict) -> dict:
    return {
        "gwexpy_version":  gwexpy.__version__,
        "numpy_version":   np.__version__,
        "h5py_version":    h5py.__version__,
        "python_version":  platform.python_version(),
        "hostname":        platform.node(),
        "created_utc":     datetime.datetime.utcnow().isoformat(),
        "params_json":     json.dumps(params),
    }

analysis_params = {
    "fftlength":  4.0,
    "overlap":    0.5,
    "method":     "median",
    "freq_range": [10.0, 2000.0],
}

hdf5_path = tempfile.mktemp(suffix=".h5")

with h5py.File(hdf5_path, "w") as f:
    raw_grp = f.require_group("raw")
    to_hdf5(ts, raw_grp, "darm")

    dset = raw_grp["darm"]
    prov = build_provenance(analysis_params)
    for key, val in prov.items():
        dset.attrs[key] = val
    dset.attrs["description"] = "生 DARM 出力、未キャリブレーション"

    print("書き込んだデータセット属性:")
    for k, v in dset.attrs.items():
        print(f"  {k}: {str(v)[:60]}")

print(f"\\nHDF5 ファイル: {hdf5_path}  ({os.path.getsize(hdf5_path)/1024:.1f} kB)")
"""),

    md("""\
## 3. 派生量をリネージ情報付きで保存する

各派生量（ASD、スペクトログラム、フィルタ済み時系列）には
親データセットへの参照を含め、完全な処理チェーンを追跡できるようにします。
"""),

    code("""\
asd = ts.asd(fftlength=analysis_params["fftlength"],
             overlap=analysis_params["overlap"],
             method=analysis_params["method"])

with h5py.File(hdf5_path, "a") as f:
    proc_grp = f.require_group("processed")

    freq_dset = proc_grp.create_dataset("asd_frequencies", data=asd.frequencies.value)
    freq_dset.attrs["unit"] = "Hz"

    asd_dset = proc_grp.create_dataset("asd_values", data=asd.value)
    asd_dset.attrs["unit"] = str(asd.unit)
    asd_dset.attrs["name"] = asd.name

    prov_asd = build_provenance(analysis_params)
    prov_asd["parent_dataset"] = "/raw/darm"
    prov_asd["processing_step"] = "Welch/median による ASD 計算"
    for key, val in prov_asd.items():
        asd_dset.attrs[key] = val

    def print_tree(name, obj):
        indent = "  " * name.count("/")
        attrs_summary = f"  [{len(obj.attrs)} 属性]" if hasattr(obj, "attrs") else ""
        print(f"{indent}{name.split('/')[-1]}{attrs_summary}")

    print("HDF5 ファイル構造:")
    f.visititems(print_tree)
"""),

    md("## 4. 読み戻してプロバナンスを検証する"),

    code("""\
with h5py.File(hdf5_path, "r") as f:
    ts_loaded = from_hdf5(TimeSeries, f["raw"], "darm")
    dset_attrs = dict(f["raw/darm"].attrs)
    params_loaded = json.loads(dset_attrs["params_json"])
    freqs_loaded = f["processed/asd_frequencies"][:]
    asd_loaded   = f["processed/asd_values"][:]
    asd_prov     = dict(f["processed/asd_values"].attrs)

print("=== 復元された TimeSeries ===")
print(f"  name   : {ts_loaded.name}")
print(f"  t0     : {ts_loaded.t0.value}")
print(f"  samples: {len(ts_loaded)}")

print("\\n=== プロバナンス (raw/darm) ===")
for key in ["gwexpy_version", "created_utc", "hostname"]:
    print(f"  {key}: {dset_attrs[key]}")

print("\\n=== 解析パラメータ ===")
for key, val in params_loaded.items():
    print(f"  {key}: {val}")

print("\\n=== ASD 親参照 ===")
print(f"  parent_dataset  : {asd_prov['parent_dataset']}")
print(f"  processing_step : {asd_prov['processing_step']}")
"""),

    md("## 5. プロバナンス対応パイプラインヘルパー"),

    code("""\
import contextlib

@contextlib.contextmanager
def provenance_file(path, params: dict, mode: str = "w"):
    with h5py.File(path, mode) as f:
        def save_ts(ts_obj, group_path: str, name: str, description: str = ""):
            grp = f.require_group(group_path)
            to_hdf5(ts_obj, grp, name)
            dset = grp[name]
            prov = build_provenance(params)
            for k, v in prov.items():
                dset.attrs[k] = v
            if description:
                dset.attrs["description"] = description
            return dset

        def save_array(arr, path_in_file: str, unit: str = "",
                       description: str = "", parent: str = ""):
            parts = path_in_file.rsplit("/", 1)
            grp_path, name = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
            grp = f.require_group(grp_path) if grp_path else f
            dset = grp.create_dataset(name, data=arr)
            if unit:        dset.attrs["unit"]        = unit
            if description: dset.attrs["description"] = description
            if parent:      dset.attrs["parent"]      = parent
            prov = build_provenance(params)
            for k, v in prov.items():
                dset.attrs[k] = v
            return dset

        yield f, save_ts, save_array

pipeline_path = tempfile.mktemp(suffix="_pipeline.h5")
run_params = {"fftlength": 8.0, "overlap": 0.75, "method": "median", "run_id": "R001"}

with provenance_file(pipeline_path, run_params) as (f, save_ts, save_array):
    save_ts(ts, "input", "darm", description="処理前の生 DARM データ")
    asd2 = ts.asd(fftlength=run_params["fftlength"], overlap=run_params["overlap"])
    save_array(asd2.value,             "products/asd",   unit="ct/rtHz",
               description="ASD",     parent="/input/darm")
    save_array(asd2.frequencies.value, "products/freqs", unit="Hz")

print(f"パイプラインアーカイブ: {pipeline_path}  ({os.path.getsize(pipeline_path)/1024:.1f} kB)")
with h5py.File(pipeline_path, "r") as f:
    params_back = json.loads(f["input/darm"].attrs["params_json"])
    print(f"run_id 検証: {params_back['run_id']}")
"""),

    md("## 6. 再現性確認：保存済み ASD の可視化"),

    code("""\
import matplotlib.pyplot as plt

with h5py.File(pipeline_path, "r") as f:
    freqs_ar = f["products/freqs"][:]
    asd_ar   = f["products/asd"][:]

fig, ax = plt.subplots(figsize=(9, 4))
ax.loglog(asd2.frequencies.value[1:], asd2.value[1:],
          color="steelblue", lw=2, label="メモリ内 ASD")
ax.loglog(freqs_ar[1:], asd_ar[1:],
          color="tomato", ls="--", lw=1.5, label="HDF5 から読み込んだ ASD")
ax.set_xlabel("周波数 [Hz]")
ax.set_ylabel("ASD [ct/√Hz]")
ax.set_title("再現性確認：メモリ内 vs. HDF5 アーカイブ")
ax.legend()
ax.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()

max_diff = np.max(np.abs(asd2.value[1:] - asd_ar[1:]) / (asd2.value[1:] + 1e-300))
print(f"最大相対誤差: {max_diff:.2e}  （マシンイプシロン程度であるべき）")
"""),

    md("""\
## まとめ

| ステップ | API / パターン | 目的 |
|---------|--------------|------|
| TimeSeries の書き込み | `to_hdf5(ts, group, path)` | データ配列 + 基本メタデータの保存 |
| TimeSeries の読み込み | `from_hdf5(TimeSeries, group, path)` | アーカイブから復元 |
| プロバナンスの追加 | `dset.attrs[key] = value` | バージョン、パラメータ、タイムスタンプ |
| 構造化パイプライン | `provenance_file(path, params)` コンテキストマネージャ | 全書き込みに自動でプロバナンスを付与 |

**推奨プロバナンス属性：**

| 属性 | 内容 |
|-----|------|
| `gwexpy_version` | `gwexpy.__version__` |
| `created_utc` | `datetime.datetime.utcnow().isoformat()` |
| `params_json` | `json.dumps(analysis_params)` |
| `parent_dataset` | 入力データセットの HDF5 パス |
| `processing_step` | 実行した処理の説明 |
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
    write_nb(EN_CELLS, root / "docs/web/en/user_guide/tutorials/case_hdf5_provenance.ipynb")
    write_nb(JA_CELLS, root / "docs/web/ja/user_guide/tutorials/case_hdf5_provenance.ipynb")
    print("Done.")
