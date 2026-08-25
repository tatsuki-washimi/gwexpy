# SeriesMatrix B1 decision

**Proposed D21 decision: B0 for v0.2.0.** B1 remains deferred pending explicit
human D21/data-model sign-off: `adopted: false`.

The proposed v0.2.0 decision retains the reviewed 480-cell B0 contract. It
supersedes the
prior 318-cell candidate (missing scalar add/sub coverage), the later 390-cell
candidate (incomplete nested metadata/attrs and exponent-boundary coverage),
and the subsequent 453-cell candidate (missing supported SpectrogramMatrix
slice and independent axes/epoch coverage), the later 454-cell candidate
(incomplete equivalent-selector and 3-D batch-slice coverage), and the later
460-cell candidate (missing scalar Spectrogram epoch and negative integer
sample-index coverage), and the 472-cell candidate (missing fail-closed
conflicting epoch/time scalar extraction and complete selected-cell metadata
preservation), the 474-cell candidate (reserved selected-cell metadata carrier
collision overwrote user attrs), and the 478-cell candidate (collision
refusals omitted two supported 4-D negative row/column selectors).
Physics/data-model review rejected those candidates. Explicit human
D21/data-model sign-off before merge or release remains pending; AI review is
advisory and cannot provide it.
B1 does not implement general direct NumPy ufunc composition, and no candidate runtime is adopted.
Unsupported direct ufuncs therefore continue to fail explicitly rather than
silently returning an ndarray or Quantity.

No target version is assigned for B1.  Reconsideration requires a separate
design review, an updated declarative contract, physics/data-model review, and
a clean reproducible benchmark comparison.
