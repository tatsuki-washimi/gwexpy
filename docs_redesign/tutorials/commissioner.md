# Commissioner: from GUI to a saved analysis

Reproduce a familiar commissioning workflow in Python: load channels, select a time span, compare ASD and coherence, and save figures with the analysis settings.
This lesson is for users of DiagGUI, ndscope, or Virgo dataDisplay who already choose channels and spectral settings in a GUI.
No Python expertise is assumed beyond following the commands below; [First analysis](first_analysis.md#names-functions-and-channels) explains the syntax if needed.

Prerequisites: [GWexpy installed](installation.md).
The main example uses synthetic data and core dependencies, including HDF5 support; it requires no detector connection.
Study goal: 20–30 minutes. Script runtime goal: seconds on a laptop, depending on the environment.

## Map GUI choices to code

| GUI action or setting | Script equivalent |
| --- | --- |
| Select channels | Keys in a `TimeSeriesDict` |
| Load an ndscope recording | `TimeSeriesDict.read(path, format="hdf.ndscope")` |
| Choose a time interval | `channels.copy().crop(start, end)`, with GPS seconds for recorded GW data |
| Choose an ASD FFT length and overlap | `channels.asd(fftlength=2, overlap=1, window="hann", method="welch")` |
| Choose a reference channel | `sensor.coherence(reference, fftlength=2, overlap=1, window="hann")` |
| Save traces as a figure | `plot.savefig("asd.png")` |
| Record the analysis setup | A JSON file of channel names, times, and spectral settings |

Virgo dataDisplay is included as a familiar workflow background.
This tutorial demonstrates the explicit ndscope and DiagGUI formats below; it does not define a direct dataDisplay reader.

## Run a complete local workflow

Download {download}`commissioner.py <../_static/downloads/commissioner.py>` into a working folder.
In a terminal, activate the environment from the installation guide and run:

```bash
python commissioner.py
```

The script creates `commissioner-output/` with `channels.hdf5`, `asd.png`, `coherence.png`, and `analysis-parameters.json`.
Rerunning it replaces these tutorial outputs.
Open the PNG files in an image viewer and the JSON file in a text editor.

```{literalinclude} ../_static/downloads/commissioner.py
:language: python
```

## Inspect the saved data and time selection

The script makes two 32-second voltage channels sampled at 512 Hz and beginning at GPS 1400000000.
Both contain a 40 Hz sine wave plus independently seeded Gaussian noise.
The channel labels are synthetic examples.

`channels.write(..., format="hdf.ndscope")` creates an ndscope-format HDF5 file.
`TimeSeriesDict.read(..., format="hdf.ndscope")` loads the channel values and their metadata.
The printed list contains `X1:REFERENCE` and `X1:SENSOR`.
These public calls load their required I/O handler on demand.

The crop selects offsets 4 through 28 seconds relative to the start: GPS 1400000004 up to, but excluding, GPS 1400000028.
The resulting 24-second segment is shared by both channels.
The script copies the collection before cropping because `TimeSeriesDict.crop()` updates its collection.
For your own recording, inspect channel names, start times, sample rates, and units before choosing the interval and reference.

## Interpret the two figures

The ASD figure shows a line near 40 Hz in both channels and a higher broadband floor in `X1:SENSOR`.
Its unit is V per square root Hz.
The script uses a Hann window, 2-second FFT segments, 1-second overlap, and Welch averaging; the frequency-bin spacing is 0.5 Hz.

The second figure shows magnitude-squared coherence, a dimensionless measure between zero and one of linear association at each frequency.
Coherence should rise near the shared 40 Hz tone.
Away from that tone, the independent noise produces a smaller, fluctuating estimate; finite averaging does not give exactly zero.
A high value identifies shared spectral content, but does not by itself establish the direction of a physical coupling.

`analysis-parameters.json` records the source file, channel choices, absolute crop bounds, FFT settings, the seeds used for the synthetic data, and the Python and package versions.
Keep this file with the figures so that the calculation can be repeated.

## Read a DiagGUI time-series export

Install the optional `dttxml` dependency for this section in the same environment:

```bash
python -m pip install dttxml
```

Download the small synthetic {download}`commissioner.xml <../_static/downloads/commissioner.xml>` sample into your working folder.
Save the following code as `read_diaggui.py` alongside it and run `python read_diaggui.py`:

```python
from gwexpy.timeseries import TimeSeriesDict

diaggui = TimeSeriesDict.read(
    "commissioner.xml", format="xml.diaggui", products="TS", unit="V"
)
print(list(diaggui))
for name, channel in diaggui.items():
    print(name, channel.t0, channel.sample_rate, channel.unit)
plot = diaggui.plot()
plot.savefig("diaggui-timeseries.png")
```

`products="TS"` selects saved time-series data.
The synthetic sample contains four voltage samples at 4 Hz under `TEST:SYNTHETIC_INPUT`.
The example passes `unit="V"` explicitly because this time-series adapter does not recover the physical unit from the XML; this unit is a known calibration assumption for the supplied sample.
For your own export, supply the unit justified by its calibration.
DiagGUI can also save frequency-domain products; those are separate spectra and use the corresponding frequency-series readers.
An export must contain time-series data for this example.
See [I/O formats](../reference/io_capabilities.md) for the product-specific entry points when loading your own saved spectral results.

## Further reading

- [First analysis](first_analysis.md): Python syntax and plot interpretation.
- [TimeSeriesMatrix basics](matrix_timeseries.ipynb): work with a larger aligned channel set.
- [Case studies](../how-to/case-studies/index.md): adapt a complete analysis to a measurement.
