# Candidate v7 SDB timing and memory evidence

**Disposition: HOLD pending diagnosis of slower selected full-range reads.** No timing or PSS improvement claim is made.

Frozen harness digest `e8422a63694d54ba4c8d349f8fb2da7f5df3cda63387d2e036c1af4cba338c72`; fixture inventory SHA-256 `2b29c7bccedfc8335beb5c59580b26aa1a211bd3959e142dcb2e99a7dbcab1f8`. Timing used independent warm 9/arm batches in the harness ABBA order; PSS used a separate cold 5/arm run with 1 ms sampling.

## Warm medians and noise-aware gate

| Route / batch | Metric | B0 median (MAD) | Candidate median (MAD) | Candidate change | 3× summed relative MAD | Gate crossed? |
|---|---|---:|---:|---:|---:|---|
| window | wall | 11.209 ms (0.225) | 10.018 ms (0.440) | -10.63% | 19.20% | no |
| window | CPU | 11.190 ms (0.225) | 10.002 ms (0.440) | -10.62% | 19.23% | no |
| selected #1 | wall | 8.411 ms (0.288) | 9.289 ms (1.292) | +10.44% | 51.98% | no |
| selected #1 | CPU | 8.403 ms (0.286) | 9.268 ms (1.277) | +10.29% | 51.56% | no |
| selected #2 | wall | 9.816 ms (0.535) | 11.055 ms (0.413) | +12.62% | 27.57% | no |
| selected #2 | CPU | 9.776 ms (0.505) | 11.043 ms (0.413) | +12.96% | 26.73% | no |
| all | wall | 11.792 ms (0.205) | 11.673 ms (0.304) | -1.01% | 13.01% | no |
| all | CPU | 11.783 ms (0.223) | 11.666 ms (0.303) | -0.99% | 13.49% | no |

The formal gate does not cross in any route. Both independent selected full-range batches nonetheless show candidate medians more than 10% slower with the same direction; relative MAD makes the formal threshold much higher. The candidate remains HOLD pending a narrow no-window path diagnosis. Per-sample paired ABBA values appear in `performance-summary.json`.

## PSS / RSS

- B0: peak-tree PSS median 304095 KiB (MAD 443); peak-tree RSS median 317368 KiB (MAD 332).
- candidate-v7: peak-tree PSS median 303617 KiB (MAD 132); peak-tree RSS median 316660 KiB (MAD 156).

Memory is supporting evidence only. No PSS/RSS improvement claim is made. The source change’s structural DataFrame-row reduction remains the primary result; validation still traverses all source timestamps in O(N) time.
