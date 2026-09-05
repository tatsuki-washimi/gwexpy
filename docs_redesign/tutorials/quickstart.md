---
myst:
  html_meta:
    description: "Generate two reproducible synthetic channels with GWexpy, calculate their amplitude spectral densities, and save a first analysis figure."
---

# Quickstart

Create two synthetic channels, calculate their amplitude spectral densities (ASDs), and save `asd.png`.
Prerequisites: Python 3.11 or later and GWexpy installed; no detector files or optional packages are needed.
Study goal: about 5 minutes after installation. Script runtime goal: seconds on a laptop; this is not a benchmark.

(en-quickstart-install-command)=
(quick-install)=
## Install GWexpy

In a terminal with your chosen Python environment active, run:

```bash
python -m pip install gwexpy
```

For environment creation and Conda instructions, use [Installation](installation.md).

(en-quick-demo)=
(4-line-quickstart)=
(multi-channel-analysis-example)=
## Run the complete example

Download {download}`quickstart.py <../_static/downloads/quickstart.py>` into a working folder and run the command below from that folder.
A terminal command starts Python; the downloaded file contains the Python statements it will execute in order.

```bash
python quickstart.py
```

```{literalinclude} ../_static/downloads/quickstart.py
:language: python
```

The `sine` and `gaussian` functions create `TimeSeries` objects with the same 16-second duration, 512 Hz sample rate, start time, and volt unit.
Explicit seeds make both noise sequences repeatable.
`TimeSeriesDict` keeps the two named channels together, and `.asd()` applies the same spectral settings to each one.
The final line saves the figure in the folder where the command runs; open `asd.png` with an image viewer.

## Read the result

```{figure} ../_static/images/quickstart-asd.png
:alt: Two voltage amplitude spectral densities with a shared peak near 40 Hz and different noise floors.
:width: 720px

Expected output from the downloadable example: a 40 Hz line in both channels and a higher broadband noise floor in Sensor B.
```

(core-concepts)=
The horizontal axis is frequency in hertz; the vertical axis is ASD in volts per square root hertz.
An ASD shows how fluctuation amplitude is distributed over frequency.
Both channels contain the same 40 Hz sine wave, while Sensor B has larger Gaussian noise.
The peak height also depends on the spectral settings; it is not the sine wave's amplitude in volts.

`fftlength=2` uses 2-second segments, giving a frequency-bin spacing of 0.5 Hz.
`overlap=1` overlaps adjacent segments by 1 second.
The example explicitly selects a Hann window and Welch averaging so the analysis choices are visible.

(30-min-hands-on-interactive-tutorial)=
(gwexpy-basic-hands-on)=
## Change one parameter

Change `frequency=40` to `frequency=70` and run the script again.
The two peaks should move to 70 Hz. Then restore 40 Hz and try changing one noise standard deviation (`std`) to see how the broadband floor responds.

(need-help)=
## If the script does not run

`ModuleNotFoundError: No module named 'gwexpy'` means the Python running the script cannot find GWexpy.
Activate the environment used for installation, then rerun `python quickstart.py`.
For other symptoms, use [Troubleshooting](../how-to/troubleshooting.md).

<a id="next-to-read"></a>
<a id="next-steps"></a>

## Further reading

- [First analysis](first_analysis.md): learn Python variables, plots, and ASD step by step.
- [Start Here](getting_started.md): choose the next lesson for your background.
- [TimeSeries basics](intro_timeseries.ipynb): continue with time-domain and frequency-domain operations.
