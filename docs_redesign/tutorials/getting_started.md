---
myst:
  html_meta:
    description: "Choose a GWexpy first lesson for beginners, GW experimentalists, commissioners, scientific Python users, GWpy users, or returning GWexpy users."
---

(getting-started)=
# Start Here

Choose your background to find a first lesson and a concrete result to work toward.
You can start with synthetic data on your own computer; detector access is not a prerequisite.
The suggested study times below are planning goals, excluding installation.

(choose-your-path)=
## Choose a first lesson

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Beginner
:link: first_analysis
:link-type: doc

Prerequisites: no Python or spectral-analysis experience assumed; install GWexpy first.

Outcome: run Python code and explain a time-series plot and an ASD plot. Study goal: 20–30 minutes.
:::

:::{grid-item-card} GW Experimentalist
:link: for-gw-experimentalists
:link-type: ref

Prerequisites: familiar with channels, sample rates, and spectra; Python may be new.

Outcome: connect those concepts to `TimeSeries`, `TimeSeriesDict`, and a saved ASD figure. Study goal: 10–20 minutes.
:::

:::{grid-item-card} Commissioner (DiagGUI · ndscope · Virgo dataDisplay)
:link: commissioner
:link-type: doc

Prerequisites: comfortable choosing channels, time spans, FFT lengths, and reference channels in a GUI.

Outcome: reproduce a saved-data workflow with ASD, coherence, and recorded analysis settings. Study goal: 20–30 minutes.
:::

:::{grid-item-card} Scientific Python User
:link: scientific_python
:link-type: doc

Prerequisites: NumPy arrays, dictionaries, and basic plotting; no GWpy knowledge needed.

Outcome: attach time and unit metadata to arrays and replace a per-channel spectral loop with a collection method. Study goal: 10–15 minutes.
:::

:::{grid-item-card} GWpy User
:link: ../explanation/gwexpy_for_gwpy_users
:link-type: doc

Prerequisites: existing GWpy scripts or familiarity with its containers.

Outcome: identify useful GWexpy additions and adapt an existing analysis with the migration examples.
:::

:::{grid-item-card} GWexpy User
:link: ../how-to/index
:link-type: doc

Prerequisites: a working GWexpy environment and a specific analysis task.

Outcome: find a task recipe or case study; use the [API reference](../reference/index.md) for exact parameters.
:::
::::

(en-learning-path)=
(learning-path)=
## Prepare and run

(1-preparation)=
1. Follow [Installation](installation.md) if GWexpy is not installed yet.
2. Open the lesson matching your background. Each includes the concepts needed for its first result.
3. For a short environment check, run the [Quickstart](quickstart.md): two synthetic channels produce a saved ASD figure.

(5-min-quick-start)=
(30-min-hands-on)=
(for-gwpy-users)=
(2-core-data-structures)=
(3-advanced-analysis)=
(4-practical-applications)=
## Continue with your result

Use [core lessons](index.md#core-lessons) to learn another container, or choose a [case study](../how-to/case-studies/index.md) that resembles your measurement.
Detailed [FFT, GPS-time, and compatibility conventions](../explanation/prerequisites_and_conventions.md) are available when you need them.

<a id="next-to-read"></a>
<a id="next-steps"></a>

## Further reading

- [All tutorials](index.md)
- [GWpy Difference API Index](../reference/gwpy_added_api.md)
- [Developer guide](../about/developer.md)
