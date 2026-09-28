# Candidate v7 WAL semantic comparison

Result: **14/14 cases stable; 4 equal; 10 differences are inside the previously selected broad concurrent-WAL snapshot exception.** This evidence requires S2 disclosure and human scientific/data-model approval. No timing is represented here.

B0 is frozen old-R baseline `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`; B1 is candidate-v7 `74cba639fb71234a1e7f326a597557488d39db5c`. Harness digest: `ef827e3e1c3fce41575fc369aa473720e2b2200d8710e2034a88530bafa49bbd`; fixture manifest SHA-256: `a1995b43e921f7dd5664f05e69872d15ff9b8c974871f9f3d1deb3448e378837`; 5 samples/arm/case, 140 worker samples total. All triggers fired, writers committed, stderr was empty, and both arms were stable in every case.

| Case | B0 vs B1 public difference | Status |
|---|---|---|
| `schema__all` | B0 sees renamed schema after validation: quoted key `"outTemp"`, all NaNs, and one nonnumeric warning; B1 uses snapshot schema and normal `outTemp` values. | Declared WAL exception |
| `schema__selected` | B0 sees renamed schema after validation: quoted key `"outTemp"`, all NaNs, and one nonnumeric warning; B1 uses snapshot schema and normal `outTemp` values. | Declared WAL exception |
| `selected_malformed__all` | B0 sees malformed selected `outTemp`, warns once and returns NaN; B1 retains pre-update numeric value without warning. | Declared WAL exception |
| `selected_malformed__selected` | B0 sees malformed selected `outTemp`, warns once and returns NaN; B1 retains pre-update numeric value without warning. | Declared WAL exception |
| `selected_value__all` | B0 sees post-validation selected `outTemp` update; B1 retains snapshot value. | Declared WAL exception |
| `selected_value__selected` | B0 sees post-validation selected `outTemp` update; B1 retains snapshot value. | Declared WAL exception |
| `timestamp__all` | B0 errors: `SDB timestamp gap at index 1: expected cadence 300, got 299`; B1 returns timestamp/value data from the earlier read snapshot. | Declared WAL exception |
| `timestamp__selected` | B0 errors: `SDB timestamp gap at index 1: expected cadence 300, got 299`; B1 returns timestamp/value data from the earlier read snapshot. | Declared WAL exception |
| `unselected_malformed__all` | All-columns read: B0 sees malformed `outHumidity`, warns once and returns NaN; B1 retains pre-update value. | Declared WAL exception |
| `unselected_malformed__selected` | Same public fingerprint, outcome, warnings, and selected values. | Equal |
| `unselected_value__all` | All-columns read: B0 sees post-validation `outHumidity` update; B1 retains snapshot value. | Declared WAL exception |
| `unselected_value__selected` | Same public fingerprint, outcome, warnings, and selected values. | Equal |
| `usunits__all` | Same public fingerprint, outcome, warnings, and selected values. | Equal |
| `usunits__selected` | Same public fingerprint, outcome, warnings, and selected values. | Equal |

Equal cases: `unselected_value__selected`, `unselected_malformed__selected`, `usunits__all`, and `usunits__selected`. In particular, the usUnits mutation produced no public difference in this trigger schedule.

The differences reflect a selected consistency-model exception: B0 can mix post-validation writer state into later payload/schema/timestamp operations; B1 holds one SQLite read snapshot across validation and payload selection. The full per-case warning lists, outcomes, key order, member value hashes, sample raw files, and normalized summary are in this directory.
