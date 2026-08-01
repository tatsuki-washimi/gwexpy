---
myst:
  html_meta:
    description: "How GWexpy relates to GWpy, spicypy, GWDama, and the wider gravitational-wave Python ecosystem, and which problems it deliberately leaves to other packages."
---

# Where GWexpy Sits in the GW Python Ecosystem

GWexpy is an extension library built on top of [GWpy](https://gwpy.github.io/). It is not an
official component of the GWpy project, and it is not a search, parameter-estimation, or
detector-operations pipeline. This page explains which layer of the gravitational-wave Python
stack GWexpy occupies, how it relates to neighbouring packages, and which problems it leaves
to other tools on purpose.

If your question is "how do I convert this object into another library?", see the
[Interop / Conversion Guide](../how-to/interop). If it is "which file format can I read?", see
the [File I/O Supported Formats Guide](../how-to/io_formats). This page is about scope and
positioning, not about API selection.

## How to Read This Page

Each project below is placed in one of five categories. The categories describe the
*relationship* to GWexpy, not the quality or maturity of the other project. The order within
any table is thematic and is not a ranking by usage statistics.

| Category | Meaning |
| --- | --- |
| Foundation | GWexpy is built on it and extends its object model |
| Complementary | Solves an adjacent problem; useful alongside GWexpy without overlap |
| Reference | GWexpy learns from its API design or data semantics, without reusing code |
| Interop candidate | A conversion or reader/writer bridge is planned or worth building |
| Out of scope | Deliberately left to that project; GWexpy does not aim to replace it |

## Layers of the Gravitational-Wave Python Stack

| Layer | Packages that primarily occupy it |
| --- | --- |
| Standard GW data objects and base API | GWpy |
| Matrix, multichannel, and experiment-oriented analysis containers | **GWexpy** |
| Signal processing and control systems | spicypy |
| HDF5 data preparation, archiving, and ML-ready datasets | GWDama |
| Detector characterization, summary pages, and veto workflows | gwdetchar, gwsumm, gwvet, hveto |
| Trigger generation, search, and parameter estimation | pyomicron, PyCBC, bilby, pygwb, pycWB |
| Data discovery and access | gwosc, gwdatafind |
| Interferometer simulation and design | Finesse, pygwinc, Differometor |

GWexpy sits one layer above GWpy and one layer below the workflow and pipeline packages. It
supplies containers, I/O, and analysis primitives that those workflows can consume, rather
than orchestrating the workflows itself.

## GWpy, GWexpy, spicypy, and GWDama Compared

These four packages are compared together because they are the ones most often mistaken for
each other: all four are Python, all four handle gravitational-wave time-series data, and
three of them relate to GWpy.

| | GWpy | GWexpy | spicypy | GWDama |
| --- | --- | --- | --- | --- |
| In one line | Base library for GW detector data analysis | GWpy-compatible extension for experimental and multichannel analysis | GWpy plus signal-processing and control-systems methods | HDF5-first data manager and data-preparation package |
| Primary layer | Base API and standard containers | Analysis containers, workflow primitives, interop | Signal, spectral, and control methods | Data acquisition, storage, and preparation |
| Central data structures | `TimeSeries`, `FrequencySeries`, `Spectrogram`, segments, `EventTable` | `TimeSeriesMatrix`, `FrequencySeriesMatrix`, `SpectrogramMatrix`, `SegmentTable`, `ScalarField`, typed result objects | GWpy series objects extended with signal and control methods | `GwDataManager` and an h5py-derived `Dataset` |
| Design approach | A standard Python interface for GW detector data | Keeps the GWpy object model, adds matrix, batch, I/O, and experiment workflows | Bridges GWpy objects and python-control style workflows | Organises raw and processed data into HDF5 groups with attributes |
| Persistence | Reads the standard GW data formats | Broad multi-format I/O and interop; no single canonical backend | I/O is secondary to the signal and control examples | HDF5 is the canonical format |
| Relationship to GWexpy | Foundation | — | Complementary, and a reference for API design | Partly overlapping; a reference and an interop candidate |
| License | GPL-3.0 | MIT | Apache-2.0 | MIT |

## How GWexpy Relates to Each Project

| Project | Category | Why |
| --- | --- | --- |
| [GWpy](https://gwpy.github.io/) | Foundation | GWexpy subclasses and extends its containers, and aims to stay readable by GWpy itself |
| [spicypy](https://gitlab.com/pyda-group/spicypy) | Complementary, Reference | Also a GWpy extension, but centred on signal processing and control systems rather than containers and I/O. Its LPSD, Daniell's method, and huddle-test APIs are useful design references |
| [GWDama](https://gwnoisehunt.gitlab.io/gwdama/) | Reference, Interop candidate | Overlaps with GWexpy in multichannel I/O and preprocessing, but its scope is data preparation and HDF5 archiving. Its hierarchical group and attribute conventions are a useful reference, and reading its HDF5 products is a reasonable bridge |
| [gwdetchar](https://gwdetchar.readthedocs.io/en/stable/), [gwsumm](https://gwsumm.readthedocs.io/en/latest/), gwvet, [hveto](https://hveto.readthedocs.io/en/stable/) | Out of scope | Operational detector-characterization and reporting workflows. GWexpy supplies analysis primitives these tools could use, but does not aim to reproduce their operator-facing workflows |
| [pyomicron](https://pyomicron.readthedocs.io/en/latest/) | Out of scope, Interop candidate | Trigger generation and HTCondor orchestration are out of scope; reading and writing the resulting trigger tables is not |
| [PyCBC](https://pycbc.org/), [bilby](https://lscsoft.docs.ligo.org/bilby/), pygwb, pycWB | Out of scope, Interop candidate | Search and inference pipelines. GWexpy connects to the data products they produce and consume, and already provides PyCBC and LAL series conversion |
| [gwosc](https://gwosc.org/), [gwdatafind](https://gwdatafind.readthedocs.io/en/stable/) | Complementary | Data discovery and access. Used through GWpy and in examples rather than wrapped by GWexpy |
| [pemcoupling](https://git.ligo.org/pem/pemcoupling) | Reference | A command-line PEM coupling-function generator. Its coupling product schema and measurement status flags are useful domain references, but no code is reused; see the licensing note below |
| Finesse, pygwinc, [Differometor](https://github.com/artificial-scientist-lab/Differometor) | Complementary, Interop candidate | Interferometer simulation and design tools. GWexpy converts their frequency-domain outputs into its own containers rather than reimplementing the simulators |

## What Makes GWexpy Different

- **GWpy-compatible by construction.** New containers stay readable as GWpy objects wherever
  possible, so code can move between the two without a rewrite.
- **Matrix-native analysis.** `TimeSeriesMatrix`, `FrequencySeriesMatrix`, and
  `SpectrogramMatrix` treat multichannel data as a first-class shape rather than as a loop over
  a dictionary.
- **Typed analysis results.** Coupling, response, and fitting workflows return dedicated result
  objects instead of bare arrays or ad-hoc dictionaries.
- **Broad I/O without a single canonical backend.** HDF5, NetCDF4, Zarr, GWF, DTT XML, ndscope
  HDF5, and instrument-specific logger formats are all supported paths; none of them is
  privileged as *the* storage format.
- **Generic source-to-target coupling.** Coupling and projection are expressed in terms of a
  source and a target, not in terms of a specific detector's naming conventions.
- **Library-first and notebook-friendly.** The public surface is Python objects and functions;
  the command-line interface is a thin convenience layer, not the primary product.

## What GWexpy Deliberately Does Not Do

Keeping these outside the core is what allows the container, I/O, and interop layers to stay
general. GWexpy does not aim to provide:

- detector- or site-specific operational pipelines
- operator-facing HTML reports and summary pages
- HTCondor or site-wide job orchestration
- veto production or search pipelines
- trigger generators
- APIs whose names hard-code one detector's channel conventions

Where a workflow needs these, the intended pattern is to use GWexpy for the analysis
primitives and a dedicated tool for the workflow around them.

## Interoperability Status

The [Interop / Conversion Guide](../how-to/interop) is the authoritative catalogue of
conversion APIs, and the [File I/O Supported Formats Guide](../how-to/io_formats) is the
authoritative list of read and write formats. The summary below only records which
ecosystem-level bridges exist today.

| Status | Projects |
| --- | --- |
| Implemented | GWpy, LALSuite, PyCBC, Finesse, pygwinc, PySpice, ObsPy, python-control, SimPEG, MTH5, scikit-rf, ROOT, ndscope HDF5, DTT / DiagGUI XML, and the other targets listed in the interop guide |
| Planned | GWDama HDF5 products, Differometor design and sensitivity outputs |
| Documentation only | spicypy, gwdetchar, gwsumm, gwvet, hveto, pyomicron, pemcoupling |

"Documentation only" means the relationship is described here and no conversion API is
planned; where those tools speak GWpy objects, GWpy is already the shared language and no
adapter is needed.

## Third-Party Code and Licensing

GWexpy takes design ideas, product semantics, and API conventions from several of the projects
above, but does not copy their code. Licences are verified by reading the LICENSE file in the
project's own repository, not by trusting package classifiers or README badges. Where a project
ships no LICENSE file, GWexpy treats it as all rights reserved and reuses nothing from it.

The per-project policy is recorded for contributors in
`docs/developers/LICENSES_THIRD_PARTY.md` in the repository.

## Related Pages

- [GWexpy for GWpy Users](gwexpy_for_gwpy_users) for the practical migration differences
- [Architecture & data flow](architecture) for how the internal layers fit together
- [Interop / Conversion Guide](../how-to/interop) for the conversion API catalogue
- [File I/O Supported Formats Guide](../how-to/io_formats) for read and write formats
- [Roadmap](roadmap) for where the project is heading
