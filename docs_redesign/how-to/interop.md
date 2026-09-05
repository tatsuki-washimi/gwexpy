# Convert scientific Python objects

Use `to_*()` and `from_*()` when converting an object already in memory.
Use [file I/O](io_formats.md) when reading or writing a local file.
The [conversion catalogue](../reference/interop_capabilities.md) records the
supported classes, directions, optional dependencies, and metadata boundaries.

## Start with an array

```python
from gwexpy.timeseries import TimeSeries

signal = TimeSeries([0.0, 1.0, 0.0, -1.0], sample_rate=4, t0=0, unit="V")
values = signal.value
```

The array contains values; keep `signal.sample_rate`, `signal.t0`, and `signal.unit`
when exporting to a representation that cannot preserve them. The
[Scientific Python tutorial](../tutorials/scientific_python.md) shows the complete
array-to-container workflow without requiring GWpy knowledge.

## Use a library-specific bridge

Follow the [interoperability tutorial](interop/intro_interop.ipynb) for worked
pandas, xarray, and external-library conversions. Install the dependencies listed
for the selected bridge and check which metadata survives the round trip.

For control-system plots, use a measured or modelled **complex transfer response**
with the correct input/output units. An ASD or PSD describes noise amplitude or
power; converting its container does not turn it into a system response.

## Detailed capabilities

The former catalogue sections remain available through these links.

:::{dropdown} Catalogue sections

:::{raw} html
<span id="interop-conversion-guide"></span>
:::

- <a href="../reference/interop_capabilities.html#interop-conversion-guide">Interop / Conversion Guide</a>

:::{raw} html
<span id="direct-i-o-names"></span>
:::

- <a href="../reference/interop_capabilities.html#direct-i-o-names">Direct I/O Names</a>

:::{raw} html
<span id="jump-links"></span>
:::

- <a href="../reference/interop_capabilities.html#jump-links">Jump Links</a>

:::{raw} html
<span id="how-to-read-this-page"></span>
:::

- <a href="../reference/interop_capabilities.html#how-to-read-this-page">How to Read This Page</a>

:::{raw} html
<span id="s-foundation-layer"></span>
:::

- <a href="../reference/interop_capabilities.html#s-foundation-layer">S. Foundation Layer</a>

:::{raw} html
<span id="status-labels"></span>
:::

- <a href="../reference/interop_capabilities.html#status-labels">Status Labels</a>

:::{raw} html
<span id="optional-dependency-policy"></span>
:::

- <a href="../reference/interop_capabilities.html#optional-dependency-policy">Optional Dependency Policy</a>

:::{raw} html
<span id="a-storage-formats-and-container-conversion"></span>
:::

- <a href="../reference/interop_capabilities.html#a-storage-formats-and-container-conversion">A. Storage Formats and Container Conversion</a>

:::{raw} html
<span id="b-analysis-library-and-object-conversion"></span>
:::

- <a href="../reference/interop_capabilities.html#b-analysis-library-and-object-conversion">B. Analysis Library and Object Conversion</a>

:::{raw} html
<span id="c-scientific-computing-signal-processing-machine-learning-and-array-backends"></span>
:::

- <a href="../reference/interop_capabilities.html#c-scientific-computing-signal-processing-machine-learning-and-array-backends">C. Scientific Computing, Signal Processing, Machine Learning, and Array Backends</a>

:::{raw} html
<span id="d-physics-and-domain-specific-libraries"></span>
:::

- <a href="../reference/interop_capabilities.html#d-physics-and-domain-specific-libraries">D. Physics and Domain-Specific Libraries</a>

:::{raw} html
<span id="related-pages"></span>
:::

- <a href="../reference/interop_capabilities.html#related-pages">Related Pages</a>

:::{raw} html
<span id="next-to-read"></span>
:::

- <a href="../reference/interop_capabilities.html#next-to-read">Next to Read</a>

:::{raw} html
<span id="interop-en-how-to-read"></span>
:::

- <a href="../reference/interop_capabilities.html#interop-en-how-to-read">interop-en-how-to-read</a>

:::{raw} html
<span id="interop-en-foundation-layer"></span>
:::

- <a href="../reference/interop_capabilities.html#interop-en-foundation-layer">interop-en-foundation-layer</a>

:::{raw} html
<span id="interop-en-status-labels"></span>
:::

- <a href="../reference/interop_capabilities.html#interop-en-status-labels">interop-en-status-labels</a>

:::{raw} html
<span id="interop-en-storage-conversion"></span>
:::

- <a href="../reference/interop_capabilities.html#interop-en-storage-conversion">interop-en-storage-conversion</a>

:::{raw} html
<span id="interop-en-analysis-conversion"></span>
:::

- <a href="../reference/interop_capabilities.html#interop-en-analysis-conversion">interop-en-analysis-conversion</a>

:::{raw} html
<span id="interop-en-ml-conversion"></span>
:::

- <a href="../reference/interop_capabilities.html#interop-en-ml-conversion">interop-en-ml-conversion</a>

:::{raw} html
<span id="interop-en-domain-conversion"></span>
:::

- <a href="../reference/interop_capabilities.html#interop-en-domain-conversion">interop-en-domain-conversion</a>

:::

```{toctree}
:hidden:

interop/intro_interop
```
