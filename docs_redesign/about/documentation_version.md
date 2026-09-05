# Documentation version

This site is built from the development branch. Its API reference describes
the package version shown above each page, which may include unreleased changes.
The currently released package is **{{ stable_release }}**. The introductory
downloadable examples are also checked against **{{ intro_examples_release }}**.

Check the package in the environment where you run your analysis:

```python
import gwexpy
print(gwexpy.__version__)
```

The commit link identifies the source used to build these pages. A local preview
with uncommitted edits is labelled "local changes". Build information in JSON format
provides the package version, source revision, and introductory example target.

:::{raw} html
<p><a href="../build-info.json">build-info.json</a></p>
:::

Use the [changelog](changelog.md) to identify changes after your installed release,
and [Known limitations](known_limitations.md) for supported boundaries. API
reference generation does not establish compatibility with every installed version.

## Reproducing a result

Keep the script or notebook, package versions, input data, channel names,
time interval, sample rate, units, and spectral settings with the exported figure.
The [Commissioner tutorial](../tutorials/commissioner.md) saves these conditions
alongside its plots. A saved image alone does not record the analysis settings.
