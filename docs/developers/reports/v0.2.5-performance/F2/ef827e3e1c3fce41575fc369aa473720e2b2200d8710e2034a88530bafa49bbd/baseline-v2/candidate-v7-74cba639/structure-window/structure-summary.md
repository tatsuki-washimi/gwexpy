# Candidate v7 SDB structure gate

**PASS:** all five B0 samples materialized 4096 source payload rows in the pandas DataFrame; all five candidate-v7 samples materialized the 512 requested-window rows. `sdb_dataframe_probe_covered` was true throughout, and both arms returned normally with empty stderr.

The exact harness digest is `e8422a63694d54ba4c8d349f8fb2da7f5df3cda63387d2e036c1af4cba338c72` (frozen commit `6f0a3d4ea14858f667abe4de2395f4359927c03d`). Fixture inventory SHA-256 is `2b29c7bccedfc8335beb5c59580b26aa1a211bd3959e142dcb2e99a7dbcab1f8`. Candidate source/wheel SHA-256: `74cba639fb71234a1e7f326a597557488d39db5c` / `692b8c48a7f8439b2a5d82cf0cb9c1e262711fbce7822c7ca590c230afab65c0`; B0 source/wheel SHA-256: `1eb2cd62c365a7ed3252ed1b1fe82f9265c5ef1c` / `473fa52f1f94d61f6b785220a082928b7a7bee4b5d215ced29b7d5d80692c498`.

| Arm | DataFrame payload rows per sample |
|---|---:|
| B0 frozen old-R | 4096, 4096, 4096, 4096, 4096 |
| Candidate v7 | 512, 512, 512, 512, 512 |

The probe counts `len(DataFrame)` returned by `pd.read_sql_query` for the payload query. It does not count all SQLite cursor rows. Validation still scans the full source in O(N) time through a bounded cursor; the TEXT conversion preflight uses bounded chunks no larger than 256 rows. The structure result makes no claim that full-source validation work was removed. Timing and PSS are captured separately.
