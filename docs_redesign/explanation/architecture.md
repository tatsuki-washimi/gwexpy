# Architecture

GWexpy is organised as three cooperating layers built on top of GWpy. This
page explains the responsibilities of each layer and why the boundaries are
drawn where they are.

:::{admonition} In a nutshell
:class: important

GWexpy keeps *data representation*, *analysis*, and *interoperability* as
separate, composable layers so that each can evolve independently.
:::

## The three layers

The system is best understood as a stack. Data flows upward from raw detector
formats into rich containers, then into analysis pipelines:

```text
            ┌─────────────────────────────────────────────┐
            │  Analysis layer                              │
            │  noise budgets · ARIMA · fitting / MCMC      │
            └───────────────▲─────────────────────────────┘
                            │ operates on
            ┌───────────────┴─────────────────────────────┐
            │  Container layer                             │
            │  TimeSeriesMatrix · FrequencySeriesMatrix    │
            │  Scalar / Vector / TensorField               │
            └───────────────▲─────────────────────────────┘
                            │ populated by
            ┌───────────────┴─────────────────────────────┐
            │  Interoperability & I/O layer (~40 modules)  │
            │  multi-format readers/writers · detector glue │
            └─────────────────────────────────────────────┘
```

### Container layer

The containers generalise GWpy's one-dimensional `TimeSeries` and
`FrequencySeries` to matrices (many synchronous channels) and fields (values
defined over a spatial grid). A single, consistent container API means that an
analysis written for one channel scales to many without rewrites.

### Analysis layer

Analysis tools consume containers and never touch raw formats directly. This
keeps the BrUCo noise budget, ARIMA modelling, and fitting/MCMC code free of
format-specific concerns.

### Interoperability layer

Roughly forty interoperability modules translate detector- and format-specific
data into the common container types. New formats can be added here without
disturbing the layers above.

## Why this matters

Because the layers only depend downward, you can adopt GWexpy incrementally:
read data with the I/O layer today, and reach for the analysis layer when you
need it.
