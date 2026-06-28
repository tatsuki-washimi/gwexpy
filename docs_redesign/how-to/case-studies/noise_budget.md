# How to build a noise budget

This recipe shows how to combine measured noise contributions and compare
their quadrature sum against a target spectrum, using the BrUCo-style
`noise_budget` helper.

:::{admonition} Goal
:class: note

Given a target channel and a set of projected contributions, produce a budget
that reveals how much of the target is explained and what remains.
:::

## 1. Project each contribution

Bring every contribution into the same `FrequencySeriesMatrix` layout as your
target measurement:

```python
from gwexpy.timeseries import TimeSeriesMatrix
from gwexpy.analysis import noise_budget

target = TimeSeriesMatrix.read("target.gwf", channels=["X1:STRAIN"]).fft()
contributions = {
    name: TimeSeriesMatrix.read(f"{name}.gwf").fft()
    for name in ("seismic", "thermal", "shot")
}
```

## 2. Compute the budget

```python
budget = noise_budget(target, contributions, normalize=True)
total = budget["sum"]
residual = budget["residual"]
```

## 3. Interpret the residual

A flat, low residual means your contributions explain the target. A structured
residual points at an unmodelled coupling.

:::{seealso}
For the underlying container behaviour, see the
[container reference](../../reference/index.md).
:::
