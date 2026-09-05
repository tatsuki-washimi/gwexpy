# Your first analysis

Run Python code that generates two signals, then save a plot of the signals and a plot of their amplitude spectral densities (ASDs).
This lesson assumes no Python or spectral-analysis knowledge. First complete [Installation](installation.md).
Study goal: 20–30 minutes; the small synthetic dataset is intended to run locally within seconds, depending on your environment.

If you already work with detector channels and spectra, jump to [familiar concepts in Python](#for-gw-experimentalists), then run the same example.

## Run Python in a file

A Python script is a text file whose name ends in `.py`.
Python executes its statements from top to bottom.
Download {download}`quickstart.py <../_static/downloads/quickstart.py>` and save it in a new working folder.
Open a terminal in that folder, activate the environment from the installation guide, and run:

```bash
python quickstart.py
```

This command belongs in the terminal. The lines in `quickstart.py` are Python code.
When the command finishes, open `asd.png` from the same folder with an image viewer.
The [Quickstart](quickstart.md#read-the-result) shows the expected figure.

A notebook provides another way to run Python: paste the script into a code cell and run it.
Run later cells in order, because they use names created by earlier cells.
The saved figure goes into the notebook's working directory.

## Names, functions, and channels

Read the downloaded script alongside this explanation:

```{literalinclude} ../_static/downloads/quickstart.py
:language: python
:start-after: quickstart-begin
:end-before: quickstart-end
```

A line beginning with `from ... import ...` makes library tools available by name.
An assignment such as `tone = sine(...)` calls a function and stores the returned object under the name `tone`.
Parentheses contain the function's inputs: `frequency=40` asks for a 40 Hz sine wave.
A variable can hold a number, text, or an object containing both measured values and metadata.

The `settings` variable is a dictionary: a collection of names and values.
Here it records a 16-second duration, a 512 Hz sample rate, a start time of zero, and volts as the unit.
`**settings` passes these named values into each signal generator.
All generated channels therefore use the same time axis and unit.

`sine(...)` produces a repeating oscillation. `gaussian(...)` produces random noise.
The `seed` fixes the random sequence so that rerunning the script reproduces the same samples.
The `+` operator adds the sine and noise sample by sample, making each simulated sensor signal.

A `TimeSeries` contains one channel's values, sampling information, start time, and unit.
A `TimeSeriesDict` groups channels under names: `channels["Sensor A"]` selects one channel.
`channels.asd(...)` calculates an ASD for every channel and returns a collection of spectra.
A dot, as in `channels.asd`, selects an operation or attribute belonging to that object.

## Plot the signal against time

Append this code to the end of `quickstart.py`, then run `python quickstart.py` again:

```python
first_second = channels.copy().crop(0, 1)
time_plot = first_second.plot(ylabel="Voltage [V]")
time_plot.gca().legend()
time_plot.savefig("timeseries.png")
print(channels["Sensor A"].sample_rate)
print(channels["Sensor A"].unit)
```

Open `timeseries.png`. Its horizontal axis represents time and its vertical axis shows voltage.
You should see an oscillation mixed with noise, with larger fluctuations in Sensor B.
`.crop(0, 1)` selects the interval from zero up to, but not including, one second.
`.crop()` updates the collection it acts on. Here `.copy()` first creates another collection, so `channels` keeps the complete data.

The two `print(...)` statements display the sample rate and unit in the terminal.
They should report 512 Hz and V. These values travel with the channel; they are not only labels on a figure.

## Read the ASD axes

An ASD describes fluctuation amplitude per square root of frequency bandwidth.
Its horizontal axis is frequency in hertz (cycles per second); its vertical axis is volts per square root hertz.
A tall feature near 40 Hz indicates the oscillation that appears in both channels.
The surrounding broadband floor comes from the noise. Sensor B uses a larger noise standard deviation and has a higher floor.

The script uses logarithmic spectral axes: equal distances represent equal ratios.
For example, moving from 10 Hz to 100 Hz spans the same distance as moving from 1 Hz to 10 Hz.
The time-series figure answers when a fluctuation happened; the ASD identifies its frequency content.

The ASD estimate uses 2-second segments (`fftlength=2`), a Hann window, Welch averaging, and 1 second of overlap.
The segment length gives a frequency-bin spacing of 1 / 2 = 0.5 Hz.
Windowing reduces spectral leakage from segment boundaries; overlap reuses samples between adjacent segments.
The ASD peak height depends on these settings, so do not read it as the original sine amplitude in volts.

(for-gw-experimentalists)=
## Familiar concepts in Python

This table connects detector-analysis terms to the runnable script above.
If Python is new, the short sections on [running a file](#run-python-in-a-file) and [names and functions](#names-functions-and-channels) explain the syntax needed here.

| Experimental concept | Python representation in this lesson |
| --- | --- |
| One channel with sampled values | A `TimeSeries` returned by `sine` or `gaussian` |
| Channel list | `TimeSeriesDict` with keys `"Sensor A"` and `"Sensor B"` |
| Sample rate and segment start | `sample_rate=512` and `t0=0` |
| Engineering unit | `unit="V"` attached to every channel |
| Time selection | `channels.copy().crop(0, 1)` |
| FFT length and overlap | `fftlength=2`, `overlap=1`, both in seconds |
| ASD traces | `channels.asd(...).plot()` |
| Save plot | `plot.savefig("asd.png")` |

Here `t0=0` is a convenient synthetic origin.
For recorded GW data, `t0` usually represents a GPS timestamp in seconds.
Crop bounds use the same coordinate as the channel's start time; for a channel starting at GPS `t0`, select its first second with `.crop(t0, t0 + 1)`.

## Try a controlled change

Change the sine frequency from 40 Hz to 70 Hz, rerun the file, and inspect `asd.png`.
The shared spectral feature should move to 70 Hz.
Restore 40 Hz, then change only Sensor B's `std` from 0.8 to 0.3.
The two broadband floors should become comparable, while the independently seeded noise traces remain different.

## Further reading

- [Commissioner workflow](commissioner.md): read saved channels and compare their ASD and coherence.
- [TimeSeries basics](intro_timeseries.ipynb): practice more time-series operations.
- [Start Here](getting_started.md): choose another route.
