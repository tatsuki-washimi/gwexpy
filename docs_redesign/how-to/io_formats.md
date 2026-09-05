# Read and write data

Use a reader that matches the file's schema and the class of data it contains.
The [I/O capability catalogue](../reference/io_capabilities.md) lists supported
classes, formats, optional dependencies, time interpretation, and metadata retention.

## Read an ndscope file

```python
from gwexpy.timeseries import TimeSeriesDict

data = TimeSeriesDict.read("measurement.h5", format="hdf.ndscope")
```

Follow the [Commissioner tutorial](../tutorials/commissioner.md) for a downloadable
sample, interval selection, ASD/coherence, and saved analysis conditions.

## Read DiagGUI time-series products

```python
data = TimeSeriesDict.read("measurement.xml", format="xml.diaggui", products="TS")
```

Specify `products` explicitly. This example requires saved time-series data;
a saved spectral or transfer-function result is a different product. See the
[DTT workflow](case-studies/case_dttxml_calibration.ipynb) for response data.

## Preserve data and metadata

```python
data.write("analysis.h5", format="hdf5", overwrite=True)
restored = TimeSeriesDict.read("analysis.h5", format="hdf5")
```

Choose [HDF5 provenance](case-studies/case_hdf5_provenance.ipynb) when you need
an example of recording origin and processing history. Check the capabilities
of the specific writer before choosing an exchange format.

## Detailed capabilities

The former catalogue sections remain available through these links.

:::{dropdown} Catalogue sections

:::{raw} html
<span id="file-i-o-supported-formats-guide"></span>
:::

- <a href="../reference/io_capabilities.html#file-i-o-supported-formats-guide">File I/O Supported Formats Guide</a>

:::{raw} html
<span id="first-decision-rules"></span>
:::

- <a href="../reference/io_capabilities.html#first-decision-rules">First: Decision Rules</a>

:::{raw} html
<span id="jump-links"></span>
:::

- <a href="../reference/io_capabilities.html#jump-links">Jump Links</a>

:::{raw} html
<span id="quick-selection-table"></span>
:::

- <a href="../reference/io_capabilities.html#quick-selection-table">Quick Selection Table</a>

:::{raw} html
<span id="basic-read-write-fetch-usage"></span>
:::

- <a href="../reference/io_capabilities.html#basic-read-write-fetch-usage">Basic `.read()` / `.write()` / `fetch()` Usage</a>

:::{raw} html
<span id="supported-classes-at-a-glance"></span>
:::

- <a href="../reference/io_capabilities.html#supported-classes-at-a-glance">Supported Classes at a Glance</a>

:::{raw} html
<span id="optional-dependency-matrix"></span>
:::

- <a href="../reference/io_capabilities.html#optional-dependency-matrix">Optional Dependency Matrix</a>

:::{raw} html
<span id="a-gw-standards"></span>
:::

- <a href="../reference/io_capabilities.html#a-gw-standards">A. GW Standards</a>

:::{raw} html
<span id="b-seismic-and-geophysical-observation"></span>
:::

- <a href="../reference/io_capabilities.html#b-seismic-and-geophysical-observation">B. Seismic and Geophysical Observation</a>

:::{raw} html
<span id="c-general-analysis-and-exchange"></span>
:::

- <a href="../reference/io_capabilities.html#c-general-analysis-and-exchange">C. General Analysis and Exchange</a>

:::{raw} html
<span id="d-loggers-and-instrument-formats"></span>
:::

- <a href="../reference/io_capabilities.html#d-loggers-and-instrument-formats">D. Loggers and Instrument Formats</a>

:::{raw} html
<span id="developer-notes"></span>
:::

- <a href="../reference/io_capabilities.html#developer-notes">Developer Notes</a>

:::{raw} html
<span id="managed-in-design-but-not-prominent-in-the-public-page"></span>
:::

- <a href="../reference/io_capabilities.html#managed-in-design-but-not-prominent-in-the-public-page">Managed in design, but not prominent in the public page</a>

:::{raw} html
<span id="planned-format-tokens"></span>
:::

- <a href="../reference/io_capabilities.html#planned-format-tokens">Planned Format Tokens</a>

:::{raw} html
<span id="timeseries-stubs"></span>
:::

- <a href="../reference/io_capabilities.html#timeseries-stubs">TimeSeries stubs</a>

:::{raw} html
<span id="frequencyseries-stubs"></span>
:::

- <a href="../reference/io_capabilities.html#frequencyseries-stubs">FrequencySeries stubs</a>

:::{raw} html
<span id="related-pages"></span>
:::

- <a href="../reference/io_capabilities.html#related-pages">Related Pages</a>

:::{raw} html
<span id="next-to-read"></span>
:::

- <a href="../reference/io_capabilities.html#next-to-read">Next to Read</a>

:::{raw} html
<span id="page-end-navigation"></span>
:::

- <a href="../reference/io_capabilities.html#page-end-navigation">Page-End Navigation</a>

:::{raw} html
<span id="io-formats-en-supported-classes"></span>
:::

- <a href="../reference/io_capabilities.html#io-formats-en-supported-classes">io-formats-en-supported-classes</a>

:::{raw} html
<span id="io-formats-en-top"></span>
<span id="io-formats-ja-top"></span>
:::

- <a href="../reference/io_capabilities.html#io-formats-en-top">File I/O Supported Formats Guide</a>

:::{raw} html
<span id="io-formats-en-quick"></span>
<span id="io-formats-ja-quick"></span>
:::

- <a href="../reference/io_capabilities.html#io-formats-en-quick">Quick Selection Table</a>

:::{raw} html
<span id="io-formats-en-basic"></span>
<span id="io-formats-ja-basic"></span>
:::

- <a href="../reference/io_capabilities.html#io-formats-en-basic">Basic read / write / fetch usage</a>

:::{raw} html
<span id="io-formats-en-a"></span>
<span id="io-formats-ja-a"></span>
:::

- <a href="../reference/io_capabilities.html#io-formats-en-a">GW Standards</a>

:::{raw} html
<span id="io-formats-en-b"></span>
<span id="io-formats-ja-b"></span>
:::

- <a href="../reference/io_capabilities.html#io-formats-en-b">Seismic and Geophysical Observation</a>

:::{raw} html
<span id="io-formats-en-c"></span>
<span id="io-formats-ja-c"></span>
:::

- <a href="../reference/io_capabilities.html#io-formats-en-c">General Analysis and Exchange</a>

:::{raw} html
<span id="io-formats-en-d"></span>
<span id="io-formats-ja-d"></span>
:::

- <a href="../reference/io_capabilities.html#io-formats-en-d">Loggers and Instrument Formats</a>

:::{raw} html
<span id="io-formats-en-dev"></span>
<span id="io-formats-ja-dev"></span>
:::

- <a href="../reference/io_capabilities.html#io-formats-en-dev">Developer Notes</a>

:::
