# SeriesMatrix B1 decision

**D21 selects B0 for v0.2.0.**  B1 is deferred: `adopted: false`.

The release retains the reviewed 318-cell B0 contract and does not implement general direct NumPy ufunc composition; no candidate runtime is adopted.
Unsupported direct ufuncs therefore continue to fail explicitly rather than
silently returning an ndarray or Quantity.

No target version is assigned for B1.  Reconsideration requires a separate
design review, an updated declarative contract, physics/data-model review, and
a clean reproducible benchmark comparison.
