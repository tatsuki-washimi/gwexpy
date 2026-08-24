# SeriesMatrix B1 decision

**D21 selects B0 for v0.2.0.**  B1 is deferred: `adopted: false`.

The release retains the reviewed 472-cell B0 contract.  It supersedes the
prior 318-cell candidate (missing scalar add/sub coverage), the later 390-cell
candidate (incomplete nested metadata/attrs and exponent-boundary coverage),
and the subsequent 453-cell candidate (missing supported SpectrogramMatrix
slice and independent axes/epoch coverage), the later 454-cell candidate
(incomplete equivalent-selector and 3-D batch-slice coverage), and the later
460-cell candidate (missing scalar Spectrogram epoch and negative integer
sample-index coverage).  Physics/data-model review rejected those candidates.
B1 does not implement general direct NumPy ufunc composition, and no candidate runtime is adopted.
Unsupported direct ufuncs therefore continue to fail explicitly rather than
silently returning an ndarray or Quantity.

No target version is assigned for B1.  Reconsideration requires a separate
design review, an updated declarative contract, physics/data-model review, and
a clean reproducible benchmark comparison.
