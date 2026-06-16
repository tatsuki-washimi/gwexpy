# Architecture

GWexpy is built in three loosely-coupled layers. Understanding the boundaries
between them makes the rest of the documentation easier to navigate.

## The three layers

**Containers** sit at the base. They generalise GWpy's `TimeSeries` and
`FrequencySeries` to higher dimensions: `TimeSeriesMatrix` and
`FrequencySeriesMatrix` for multi-channel data, and `ScalarField`,
`VectorField`, and `TensorField` for spatial data. Containers know about their
own shape and sampling but contain no analysis logic.

**Analysis methods** operate on containers. BrUCo (broadband coherence), ARIMA
modelling, curve fitting, and MCMC sampling all take a container in and return
a container or a result object out. Because they only depend on the container
interface, a new container type works with every analysis method for free.

**Interop modules** connect GWexpy to the wider ecosystem — roughly forty
adapters covering multiple file formats and neighbouring libraries. They are
isolated so that an optional dependency never blocks the core.

## Why a matrix, not a list?

Representing multi-channel data as a single matrix (rather than a list of
single-channel series) keeps the sampling rate and time axis in one place and
lets analysis code lean on vectorised array operations. The container enforces
that every channel shares one time base, which removes a whole class of
alignment bugs.

## Data flow

A typical session flows in one direction:

```text
   raw data  ──▶  container  ──▶  analysis method  ──▶  result/plot
                    ▲                                       │
                    └────────────  interop I/O  ◀───────────┘
```

This separation is what lets the {doc}`reference <../reference/index>` stay
small: most pages document a container or a method, and the two compose
predictably.
