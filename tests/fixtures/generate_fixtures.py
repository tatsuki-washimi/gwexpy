import base64
import json
import os
import sqlite3
import struct
import warnings
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

# Official tools / KAGRA-standard libraries
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from scipy import signal
from scipy.io import wavfile

# Optional dependencies
try:
    import h5py
except ImportError:
    h5py = None

try:
    import zarr
except ImportError:
    zarr = None

try:
    import obspy
except ImportError:
    obspy = None

try:
    import netCDF4
except ImportError:
    netCDF4 = None

try:
    import nptdms
except ImportError:
    nptdms = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

try:
    import uproot
except ImportError:
    uproot = None

try:
    import pandas as pd
except ImportError:
    pd = None

# Keep generated fixture bytes stable across repeated local/CI runs so the test
# harness can safely regenerate files without polluting the git worktree.
np.random.seed(1)

def try_or_skip(name, func, *args, **kwargs):
    """Execution wrapper to continue generation even if a specific format fails."""
    try:
        func(*args, **kwargs)
        print(f"  [OK] {name}")
    except Exception as e:
        print(f"  [SKIP] {name}: {e}")

def generate_ats(filename):
    """ATS format (header size >= 1024)"""
    size = 80
    header_size = 1024
    header_vers = 80
    sample_freq = 128.0
    start_time_unix = 1704067200  # 2024-01-01 00:00:00
    lsb_mV = 1.0
    bit_indicator = 0  # 32-bit int

    header = bytearray(header_size)
    struct.pack_into('<H', header, 0x00, header_size)
    struct.pack_into('<h', header, 0x02, header_vers)
    struct.pack_into('<I', header, 0x04, size)
    struct.pack_into('<f', header, 0x08, sample_freq)
    struct.pack_into('<I', header, 0x0C, start_time_unix)
    struct.pack_into('<d', header, 0x10, lsb_mV)
    struct.pack_into('<h', header, 0xAA, bit_indicator)

    data = np.sin(np.linspace(0, 10, size)) * (2**30)
    data = data.astype(np.int32)

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(data.tobytes())

def generate_gbd(filename, size=100):
    """GBD format (text header + binary body)"""
    channels = ["CH1", "Alarm", "AlarmOut"]
    dt = 1.0
    start_str = "2024/01/01 00:00:00"
    header_lines = [
        "HeaderSiz=0",
        f"$$Time Start={start_str}",
        "$$Time Stop=2024/01/01 00:01:40",
        f"$$Data Sample={dt}",
        "$$Data Type=Little,Int16",
        "$$Data Order=" + ",".join(channels),
        f"$$Data Counts={size}",
    ]
    while True:
        header = "\r\n".join(header_lines) + "\r\n"
        sz = len(header.encode("ascii"))
        new_line = f"HeaderSiz={sz}"
        if header_lines[0] == new_line:
            break
        header_lines[0] = new_line

    data = np.zeros((size, len(channels)), dtype=np.int16)
    data[:, 0] = (np.sin(np.linspace(0, 10, size)) * 1000).astype(np.int16)
    data[:, 1] = (np.random.rand(size) > 0.5).astype(np.int16)
    data[:, 2] = (np.random.rand(size) > 0.8).astype(np.int16)

    with open(filename, 'wb') as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())

def generate_ndscope_hdf5(filename):
    """ndscope HDF5 format using h5py (following official schema)"""
    if h5py is None: return
    channels = [
        "K1:PEM-SEIS_BS_GND_EW_IN1_DQ",
        "K1:PEM-SEIS_BS_GND_NS_IN1_DQ",
        "K1:PEM-SEIS_BS_GND_UD_IN1_DQ",
    ]
    rate_hz = 512.0
    gps_start = 1457936560.2988281
    unit = "ct"
    n_samples = 48677

    with h5py.File(filename, 'w') as f:
        for ch_name in channels:
            grp = f.create_group(ch_name)
            grp.attrs["rate_hz"] = rate_hz
            grp.attrs["gps_start"] = gps_start
            grp.attrs["unit"] = unit
            grp.create_dataset("raw", data=np.random.randn(n_samples).astype(np.float32))

def generate_gwf(filename):
    """Frame (GWF) format using GWpy"""
    channels = ["K1:CAL-CS_PROC_DARM_DISPLACEMENT_DQ"]
    rate_hz = 16384.0
    gps_start = 1457936560.0
    n_samples = 16384 # 1 second

    tsd = TimeSeriesDict()
    for ch in channels:
        data = np.random.randn(n_samples)
        ts = TimeSeries(data, sample_rate=rate_hz, t0=gps_start, channel=ch, name=ch)
        tsd[ch] = ts

    tsd.write(filename, format='gwf')

def _to_base64_stream(data):
    if np.iscomplexobj(data):
        interleaved = np.empty(data.size * 2, dtype=data.real.dtype)
        interleaved[0::2] = data.real
        interleaved[1::2] = data.imag
        return base64.b64encode(interleaved.tobytes()).decode('ascii')
    return base64.b64encode(data.tobytes()).decode('ascii')

def generate_diaggui_xml(filename):
    """DTT / diaggui XML format (LIGO_LW)."""
    fs = 2048.0
    duration = 2.0
    t = np.arange(0, duration, 1/fs)
    chA_data = np.sin(2 * np.pi * 60 * t) + 0.5 * np.random.randn(len(t))
    chB_data = 2.0 * np.sin(2 * np.pi * 60 * t + 0.1) + 0.5 * np.random.randn(len(t))
    chA_name = "K1:TST-CH_A"
    nperseg = int(fs)
    f, pA = signal.welch(chA_data, fs=fs, nperseg=nperseg)
    f, pB = signal.welch(chB_data, fs=fs, nperseg=nperseg)
    f, pAB = signal.csd(chA_data, chB_data, fs=fs, nperseg=nperseg)
    asd_A = np.sqrt(pA)
    root = ET.Element("LIGO_LW")
    tp = ET.SubElement(root, "LIGO_LW", Name="TestParameters")
    params = {"MeasChn[0]": chA_name, "MeasActive[0]": "true", "MeasActive[1]": "true"}
    for k, v in params.items():
        p = ET.SubElement(tp, "Param", Name=k, Type="string")
        p.text = v

    res_asd = ET.SubElement(root, "LIGO_LW", Name="Result[0]", Type="Spectrum")
    ET.SubElement(res_asd, "Param", Name="f0", Type="double").text = str(f[0])
    ET.SubElement(res_asd, "Param", Name="df", Type="double").text = str(f[1] - f[0])
    arr_asd = ET.SubElement(res_asd, "Array", Type="float", Name="Spectrum")
    ET.SubElement(arr_asd, "Dim").text = str(len(f))
    ET.SubElement(arr_asd, "Stream", Encoding="LittleEndian,base64").text = _to_base64_stream(asd_A.astype(np.float32))

    with open(filename, 'wb') as f_out:
        f_out.write(b'<?xml version="1.0" encoding="utf-8" ?>\n')
        f_out.write(b'<!DOCTYPE LIGO_LW SYSTEM "http://ldas-sw.ligo.caltech.edu/doc/util/ligo_lw.dtd">\n')
        tree = ET.ElementTree(root)
        tree.write(f_out, encoding="utf-8")

def generate_wav(filename):
    samplerate=44100
    duration=0.1
    t = np.linspace(0.0, duration, int(samplerate * duration))
    data = np.sin(2 * np.pi * 440 * t)
    wavfile.write(filename, samplerate, (data * 32767).astype(np.int16))

def generate_davis_db(filename):
    if os.path.exists(filename): os.remove(filename)
    conn = sqlite3.connect(filename)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE archive (dateTime INTEGER PRIMARY KEY, barometer REAL, outTemp REAL)")
    gps_start_unix = 1457936560
    for i in range(10):
        t = gps_start_unix + i * 60
        cursor.execute("INSERT INTO archive VALUES (?, ?, ?)", (t, 1013.25, 20.0 + i))
    conn.commit()
    conn.close()

def generate_win(filename):
    """WIN format (Manual binary generation for better reliability)."""
    # Simple WIN-like header: 1024 bytes with sync and channel info
    # This satisfies readers that check for basic structure
    header = bytearray(1024)
    header[0:4] = b"WIN\x00"  # Sync/Indicator
    data = np.sin(np.linspace(0, 5, 1000)).astype(np.int32)
    with open(filename, 'wb') as f:
        f.write(header)
        f.write(data.tobytes())

def generate_zarr(filename):
    """Zarr format (v2/v3 compatible)."""
    if zarr is None: return
    try:
        # Compatibility fix for Zarr 3.x
        root = zarr.open(filename, mode='w')
        data = np.cos(np.linspace(0, 5, 500))
        # Ensure shape and dtype are explicitly passed
        root.create_dataset("ch1", data=data, shape=data.shape, dtype=data.dtype, chunks=(100,))
        root.attrs["t0"] = 1704067200.0
        root.attrs["dt"] = 0.01
    except (ImportError, Exception):
        pass

def generate_netcdf4(filename):
    if netCDF4 is None: return
    with netCDF4.Dataset(filename, 'w', format='NETCDF4') as ds:
        ds.createDimension('time', 100)
        v = ds.createVariable('ch1', 'f4', ('time',))
        v[:] = np.random.randn(100)
        v.setncattr('t0', 1704067200.0)
        v.setncattr('dt', 0.1)

def generate_tdms(filename):
    if nptdms is None: return
    from nptdms import ChannelObject, TdmsWriter
    with TdmsWriter(filename) as tdms_writer:
        channel = ChannelObject('Group', 'ch1', np.random.randn(100))
        tdms_writer.write_segment([channel])

def generate_csv_enhanced(filename):
    content = "# format: csv_enhanced\n# t0: 1704067200.0\n# dt: 0.1\ntime,ch1\n0.0,0.1\n0.1,0.2"
    with open(filename, 'w') as f: f.write(content)

def generate_seismic_mseed(filename):
    if obspy is None: return
    from obspy import Stream, Trace, UTCDateTime
    tr = Trace(data=np.random.randn(100).astype(np.float32), header={'starttime': UTCDateTime("2024-01-01"), 'sampling_rate': 10.0})
    Stream(traces=[tr]).write(filename, format="MSEED")

# --- Phase 3.11 Additions ---

def generate_segwizard(filename):
    """SegWizard-style ASCII segment list."""
    with open(filename, "w") as f:
        f.write("0 1000000000 1000000100 100\n1 1000000500 1000000600 100\n")

def generate_segments_json(filename):
    """JSON segment list."""
    with open(filename, "w") as f:
        json.dump([[1000000000, 1000000100], [1000000500, 1000000600]], f)

def generate_ligolw_xml(filename):
    """General LIGO_LW XML for segments/tables."""
    content = """<?xml version='1.0' encoding='utf-8'?>
<LIGO_LW>
   <Table Name="segment:table">
      <Column Name="segment:process_id" Type="ilong"/>
      <Column Name="segment:start_time" Type="int_4s"/>
      <Column Name="segment:end_time" Type="int_4s"/>
      <Stream Name="segment:table" Delimiter=",">
         0,1000000000,1000000100,
         0,1000000200,1000000300
      </Stream>
   </Table>
</LIGO_LW>"""
    with open(filename, "w") as f: f.write(content)

def generate_omega_h5(filename):
    """Omega Trigger Table."""
    if h5py is None: return
    with h5py.File(filename, "w") as f:
        f.create_dataset("peak_time", data=[1000000000.5, 1000000010.2])
        f.create_dataset("peak_frequency", data=[100.0, 250.0])
        f.create_dataset("snr", data=[10.5, 8.2])

def generate_mp3(filename):
    """MP3 file (Audio)."""
    if AudioSegment is not None:
        try:
            rate = 44100
            t = np.linspace(0, 0.1, int(rate * 0.1))
            data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
            audio = AudioSegment(data.tobytes(), frame_rate=rate, sample_width=2, channels=1)
            audio.export(filename, format="mp3")
            return
        except Exception: pass
    with open(filename, "wb") as f: f.write(b"ID3" + b"\x00" * 32)

def generate_root_tgraph(filename):
    """ROOT file (TGraph via uproot)."""
    if uproot is not None:
        try:
            with uproot.recreate(filename) as f:
                f["tg"] = {"x": np.arange(10), "y": np.sin(np.arange(10))}
            return
        except Exception: pass
    with open(filename, "wb") as f: f.write(b"root" + b"\x00" * 32)

def generate_ascii_basic(filename):
    """Basic ASCII series."""
    with open(filename, "w") as f:
        f.write("# time value\n0.0 1.0\n0.1 1.2\n")

def generate_parquet(filename):
    """Parquet format (via pandas)."""
    if pd is None: return
    df = pd.DataFrame({'time': np.arange(10), 'value': np.sin(np.arange(10))})
    df.to_parquet(filename)


def generate_meep_h5(filename):
    """MEEP-style HDF5 (Electromagnetic simulation)."""
    if h5py is None: return
    with h5py.File(filename, "w") as f:
        # MEEP uses .r and .i suffixes for complex fields
        f.create_dataset("ex.r", data=np.random.randn(10, 10))
        f.create_dataset("ex.i", data=np.random.randn(10, 10))


def generate_numpy_native(out_dir):
    """Numpy native formats (NPY, NPZ)."""
    np.save(out_dir / "test.npy", np.random.randn(10))
    np.savez(out_dir / "test.npz", data=np.random.randn(10))


def generate_all():
    out_dir = Path("tests/fixtures/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating fixtures in {out_dir}...")
    allow_zarr = os.environ.get("GWEXPY_ALLOW_ZARR", "") == "1"

    # TimeSeries / Audio
    try_or_skip("ATS", generate_ats, out_dir / "test.ats")
    try_or_skip("GBD", generate_gbd, out_dir / "test.gbd")
    try_or_skip("GWF", generate_gwf, out_dir / "test.gwf")
    try_or_skip("WAV", generate_wav, out_dir / "test.wav")
    try_or_skip("MP3", generate_mp3, out_dir / "test.mp3")
    try_or_skip("WIN", generate_win, out_dir / "test.win")
    try_or_skip("MSEED", generate_seismic_mseed, out_dir / "test.mseed")
    try_or_skip("ndscope H5", generate_ndscope_hdf5, out_dir / "ndscope.h5")
    if allow_zarr:
        try_or_skip("Zarr", generate_zarr, out_dir / "test.zarr")
    else:
        print("  [SKIP] Zarr: set GWEXPY_ALLOW_ZARR=1 to enable Zarr fixture generation")
    try_or_skip("NetCDF4", generate_netcdf4, out_dir / "test.nc")
    try_or_skip("TDMS", generate_tdms, out_dir / "test.tdms")
    try_or_skip("CSV Enhanced", generate_csv_enhanced, out_dir / "test.csv")
    try_or_skip("ASCII Basic", generate_ascii_basic, out_dir / "test.txt")

    # Specialized Types
    try_or_skip("DTT XML", generate_diaggui_xml, out_dir / "diaggui.xml")
    try_or_skip("SegWizard", generate_segwizard, out_dir / "test.segwizard")
    try_or_skip("Segments JSON", generate_segments_json, out_dir / "test.segments.json")
    try_or_skip("LIGOLW XML", generate_ligolw_xml, out_dir / "test.xml")
    try_or_skip("Omega H5", generate_omega_h5, out_dir / "test.omega.h5")
    try_or_skip("Davis DB", generate_davis_db, out_dir / "davis.db")
    try_or_skip("ROOT", generate_root_tgraph, out_dir / "test.root")
    try_or_skip("Parquet", generate_parquet, out_dir / "test.parquet")
    try_or_skip("MEEP H5", generate_meep_h5, out_dir / "test.meep.h5")
    try_or_skip("Numpy Native", generate_numpy_native, out_dir)

    # Stubs
    (out_dir / "test.flac").touch()
    (out_dir / "test.ogg").touch()

    print("Success.")

if __name__ == "__main__":
    generate_all()
