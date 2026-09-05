"""Save synthetic ndscope data, analyze a segment, and record the settings."""

import json
import platform
from importlib.metadata import version
from pathlib import Path

from gwexpy.noise.wave import gaussian, sine
from gwexpy.timeseries import TimeSeriesDict

# settings-begin
output = Path("commissioner-output")
output.mkdir(exist_ok=True)
parameters = {
    "sample_rate_hz": 512,
    "t0_gps_s": 1400000000,
    "duration_s": 32,
    "unit": "V",
    "tone_hz": 40,
    "noise_std_v": {"X1:REFERENCE": 0.3, "X1:SENSOR": 0.8},
    "noise_seeds": {"X1:REFERENCE": 10, "X1:SENSOR": 20},
    "crop_offset_s": [4, 28],
    "fftlength_s": 2,
    "overlap_s": 1,
    "window": "hann",
    "asd_method": "welch",
    "reference_channel": "X1:REFERENCE",
    "sensor_channel": "X1:SENSOR",
}
# settings-end

# data-begin
settings = dict(
    duration=parameters["duration_s"],
    sample_rate=parameters["sample_rate_hz"],
    t0=parameters["t0_gps_s"],
    unit=parameters["unit"],
)
tone = sine(frequency=parameters["tone_hz"], **settings)
channels = TimeSeriesDict(
    {
        name: tone
        + gaussian(std=parameters["noise_std_v"][name], seed=seed, **settings)
        for name, seed in parameters["noise_seeds"].items()
    }
)
data_path = output / "channels.hdf5"
channels.write(data_path, format="hdf.ndscope", overwrite=True)
loaded = TimeSeriesDict.read(data_path, format="hdf.ndscope")
print("Loaded channels:", list(loaded))
# data-end

# analysis-begin
t0 = parameters["t0_gps_s"]
start = t0 + parameters["crop_offset_s"][0]
end = t0 + parameters["crop_offset_s"][1]
segment = loaded.copy().crop(start, end)
spectral_settings = dict(
    fftlength=parameters["fftlength_s"],
    overlap=parameters["overlap_s"],
    window=parameters["window"],
)
spectra = segment.asd(method=parameters["asd_method"], **spectral_settings)
asd_plot = spectra.plot(xlim=(1, 256), ylabel=r"ASD [V/$\sqrt{\mathrm{Hz}}$]")
asd_plot.gca().legend()
asd_plot.savefig(output / "asd.png")

reference = segment[parameters["reference_channel"]]
sensor = segment[parameters["sensor_channel"]]
coherence = sensor.coherence(reference, **spectral_settings)
coherence_plot = coherence.plot(
    xlim=(1, 256), ylim=(0, 1), yscale="linear", ylabel="Magnitude-squared coherence"
)
coherence_plot.savefig(output / "coherence.png")
# analysis-end

# save-begin
parameters["crop_start_gps_s"] = start
parameters["crop_end_gps_s"] = end
parameters["input_file"] = str(data_path)
parameters["versions"] = {
    package: version(package)
    for package in ("gwexpy", "gwpy", "numpy", "scipy", "astropy", "h5py")
}
parameters["versions"]["python"] = platform.python_version()
(output / "analysis-parameters.json").write_text(
    json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
)
print("Saved data, figures, and analysis-parameters.json in", output)
# save-end
