# B-F2-SDB WAL concurrency baseline-v2

This append-only fault matrix records public `TimeSeriesDict.read(..., format="sdb")`
behavior from the published v0.2.4 wheel (B0) and the old-R v0.2.5 wheel
(B1, source `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c`). It supplements
the frozen F2 baseline-v1; it does not replace the baseline-v1 CSV, SDB
static-source timing/structure, or TDMS evidence.

The 64-row fixture is deterministic and copied into a fresh scratch database
for every worker. On the first payload `SELECT` containing `outTemp` and
`dateTime`, a second SQLite WAL connection executes `BEGIN IMMEDIATE`, one
mutation, then `COMMIT`, before the payload query runs. Each raw sample records
the metadata SQL executed before the trigger, exact trigger SQL, update SQL and
parameters, commit, WAL mode, first-row values and schema before/after, public
fingerprint, warnings, logs, and any error. The runner asserts that metadata
validation ran before the trigger and that the writer committed. B0 and B1
use the same wheel-only install mode and dependency environment. Each case has
five samples per arm in ABBA order; all 14 cases were stable within each arm.

| Mutation after metadata validation | Selected `outTemp` B1 | All columns B1 |
| --- | --- | --- |
| `outTemp` 60→99 | returns post-commit value | returns post-commit value |
| `outHumidity` 40→99 | unchanged selected value | returns post-commit humidity |
| `outTemp` 60→`not-a-number` | returns NaN and `UserWarning` | returns NaN and `UserWarning` |
| `outHumidity` 40→`not-a-number` | no payload warning | returns NaN and `UserWarning` |
| first `dateTime` +1 second | `ValueError`: cadence 300, got 299 | same `ValueError` |
| first `usUnits` 1→2 | returns; validation saw pre-commit units | returns; validation saw pre-commit units |
| rename `outTemp` to `outTempRenamed` | returns a literal `"outTemp"` column of NaNs with `UserWarning` | same renamed/literal column plus other columns |

The raw fingerprints, including the common dependency `DeprecationWarning`,
are authoritative; the table summarizes them. B0 and B1 were equal on all
14 scheduled cases. The selected-value case reproduces old-R's 37.22222222222222
°C first result after the commit. A single pinned read snapshot would instead
see the fixture's pre-commit 60 °F (15.555555555555555 °C); that candidate
result is an inference, not a measured candidate result.

The user chose a scoped SDB concurrent-WAL exception covering values,
warnings, and errors caused by a writer commit after snapshot pinning.
Candidate SDB behavior must be compared using the frozen harness bytes from
the commit containing this evidence, with an equivalent trigger if its SQL
route changes. Static-source behavior remains B1-equivalent. Separate
technical review, disclosure, and human S2 scientific/data-model approval
remain required before claiming this exception or merging the SDB runtime
change. This matrix makes no performance claim and does not freeze any
additional CSV/TDMS route.
