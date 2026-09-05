# Tutorials

Choose a lesson by the result you want to produce.
[Start Here](getting_started.md) recommends an entry point for six backgrounds, including readers new to Python.
Study times are planning goals, excluding installation.

## Start here

| Lesson | Prerequisites | What you will produce | Study goal |
| --- | --- | --- | --- |
| [Installation](installation.md) | A computer with Python or Conda | An environment that can import GWexpy | Follow the setup steps |
| [Quickstart](quickstart.md) | GWexpy installed; able to run a terminal command | A saved ASD comparison of two synthetic channels | 5 minutes |
| [First analysis](first_analysis.md) | GWexpy installed; no Python or spectral-analysis knowledge assumed | Time-series and ASD figures, with the code and axes explained | 20–30 minutes |
| [Commissioner workflow](commissioner.md) | Familiar with channels, time spans, and GUI spectral settings | A saved-data analysis with ASD, coherence, and recorded settings | 20–30 minutes |
| [Scientific Python to GWexpy](scientific_python.md) | NumPy arrays, dictionaries, and basic plotting | Arrays carrying time and unit metadata; spectra computed for a collection | 10–15 minutes |

GW experimentalists can start at [familiar concepts in Python](first_analysis.md#familiar-concepts-in-python).
GWpy users can use the [migration guide](../explanation/gwexpy_for_gwpy_users.md).
Returning GWexpy users can browse [task recipes](../how-to/index.md).

## Core lessons

These notebooks extend the first analysis to another container or technique.
Open a notebook in Jupyter and run its cells from top to bottom.
Use the installation instructions for any additional packages named by a lesson.

| Lesson | Prerequisites | Learning outcome |
| --- | --- | --- |
| [TimeSeries basics](intro_timeseries.ipynb) | First analysis or equivalent Python experience | Filter a channel, compute spectra, and use time-series operations |
| [FrequencySeries basics](intro_frequencyseries.ipynb) | TimeSeries and frequency-domain concepts | Work with spectra and transfer functions |
| [Spectrogram basics](intro_spectrogram.ipynb) | TimeSeries and ASD | Build and interpret a time-frequency representation |
| [Plotting basics](intro_plotting.ipynb) | A TimeSeries or spectrum to plot | Customize axes, labels, and saved figures |
| [Fitting basics](intro_fitting.ipynb) | Arrays, plotting, and a model to fit | Fit a model and inspect its parameters |
| [Noise generation basics](intro_noise.ipynb) | TimeSeries and ASD | Generate synthetic waveforms and noise with specified parameters |
| [TimeSeriesMatrix basics](matrix_timeseries.ipynb) | TimeSeriesDict and multi-channel analysis | Organize aligned channels in a matrix container |

After a lesson, adapt a [case study](../how-to/case-studies/index.md) or consult the [API reference](../reference/index.md) for parameter details.

```{toctree}
:hidden:
:caption: Start here

getting_started
installation
quickstart
first_analysis
commissioner
scientific_python
```

```{toctree}
:hidden:
:caption: Core lessons

intro_timeseries
intro_frequencyseries
intro_spectrogram
intro_plotting
intro_fitting
intro_noise
matrix_timeseries
```
