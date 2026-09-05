---
sd_hide_title: true
---

# GWexpy

::::{div} sd-text-center sd-my-3

:::{div} sd-fs-1 sd-font-weight-bolder
GWexpy
:::

```{rubric} Analyze experimental data with units, timestamps, and channel names.
```

Compare multiple channels and save figures with the code and settings needed to reproduce them.
GWexpy extends [GWpy](https://gwpy.github.io/) with matrix and field containers,
noise generation, fitting, and connections to scientific Python libraries.

[Start Here →](tutorials/getting_started){.sd-btn .sd-btn-primary .sd-shadow-sm .sd-px-4 .sd-fs-5}
::::

## Start from your background

Choose a route that uses what you already know. Each first lesson states its prerequisites and the result you will produce.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Beginner
:link: tutorials/first_analysis
:link-type: doc

New to Python or signal analysis? Run a script, understand its variables, and read your first time-series and ASD plots.
:::

:::{grid-item-card} GW Experimentalist
:link: for-gw-experimentalists
:link-type: ref

Know channels, sampling, and spectra? Map familiar experimental concepts to Python objects, then analyze a pair of channels.
:::

:::{grid-item-card} Commissioner (DiagGUI · ndscope · Virgo dataDisplay)
:link: tutorials/commissioner
:link-type: doc

Bring an interactive analysis workflow into a reproducible script: read channels, select a time span, plot ASD and coherence, and save the settings.
:::

:::{grid-item-card} Scientific Python User
:link: tutorials/scientific_python
:link-type: doc

Start with NumPy arrays and dictionaries. Attach sampling, start time, and units, then calculate spectra for all channels together.
:::

:::{grid-item-card} GWpy User
:link: explanation/gwexpy_for_gwpy_users
:link-type: doc

Use familiar GWpy concepts and explore the added containers, analysis methods, and I/O through concrete migration examples.
:::

:::{grid-item-card} GWexpy User
:link: how-to/index
:link-type: doc

Go directly to a task recipe, a case study, or the API reference for your next analysis.
:::
::::

(install-and-try-it)=
## Try a multi-channel ASD

After [installation](tutorials/installation.md), this complete example generates two synthetic channels and saves `asd.png`. It needs no data download or optional packages.

```{literalinclude} _static/downloads/quickstart.py
:language: python
:start-after: quickstart-begin
:end-before: quickstart-end
```

[Run the example and see the expected figure](tutorials/quickstart.md).

(find-your-path)=
## Browse the documentation

The documentation follows [Diátaxis](https://diataxis.fr/): lessons for learning, recipes for tasks, reference for lookup, and explanation for context.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Follow a guided lesson and produce a working analysis.
:::

:::{grid-item-card} How-to guides
:link: how-to/index
:link-type: doc

Solve a specific analysis problem or adapt a case study.
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc

Look up containers, methods, parameters, and supported file formats.
:::

:::{grid-item-card} Explanation
:link: explanation/index
:link-type: doc

Understand the data model, analysis conventions, and design decisions.
:::
::::

For corresponding GWpy APIs, default finite numerical results, sample
selection, axis information, and successful completion remain GWpy-compatible.
Intentional divergence from these guarantees requires explicit user opt-in, except for a
named, human-approved safety exception satisfying all policy gates. See the
[GWpy compatibility policy](explanation/gwpy_compatibility_policy).

(highlights)=
For contribution, testing, and release information, use the [Developer guide](about/developer.md).

```{toctree}
:hidden:
:caption: Documentation

tutorials/index
how-to/index
reference/index
explanation/index
about/index
```
