# GWpy Behavioral Compatibility Policy

GWexpy extends GWpy without silently changing the default scientific result of
an existing GWpy API.

For APIs corresponding to existing GWpy APIs, when GWpy returns normally with
finite numerical results, replacing the corresponding GWpy import with GWexpy
under default options must preserve numerical values, shape and selected
samples, axis information, and successful completion. Any intentional
numerical divergence requires an explicit user opt-in through a GWexpy-specific
API or option. Internal implementations may differ only when scientific
behavior is preserved and no material performance or resource regression is
introduced.

## What must match

Apply this policy when all of the following are true:

1. The operation corresponds to an existing public GWpy API.
2. The call uses default behavior rather than a GWexpy-specific opt-in.
3. GWpy accepts the input and returns normally with finite numerical results.

Compare the observable result, not the implementation:

| Area | Required comparison |
| --- | --- |
| Numerical result | `values` and units exposed by the corresponding API |
| Selection | `shape`, selected samples, and boundary behavior |
| Axes | `t0`, `dt`, `times`, `span`, and equivalent axis information |
| Completion | successful completion in GWpy must not become a GWexpy-only exception |

A difference in any required comparison is a release blocker unless the user
selected an explicit GWexpy-only API or option.

## Where extension is allowed

GWexpy may add containers, metadata, precision-preserving state, I/O formats,
and analysis methods that GWpy does not provide. Those additions must remain
separate from the default behavior of corresponding GWpy APIs. Merely importing
GWexpy is not an opt-in to different numerical semantics.

Invalid or contradictory GWexpy-specific metadata may continue to fail closed.
Such malformed extension metadata is outside the normal finite-result case, and
must have explicit validation tests.

## Internal changes and resource use

Internal structure may change when the public scientific behavior remains the
same. Changes to performance-sensitive bootstrap, dispatch, I/O, or numerical
kernel paths require performance/resource non-regression evidence proportionate
to their risk. A material regression blocks the change unless maintainers
explicitly accept and document the trade-off.

Documentation-only changes may record resource evidence as not applicable.

## Review checklist

1. Is this an existing GWpy API?
2. Does GWpy return a normal finite result for the case under review?
3. Do GWpy and GWexpy defaults match in values, shape and selected samples,
   axes, and completion behavior?
4. If they differ, is the difference behind an explicit GWexpy-only opt-in?
5. If the change is internal, is proportionate performance and resource
   evidence attached?

If step 3 fails without satisfying step 4, the review verdict is **BLOCK**.
