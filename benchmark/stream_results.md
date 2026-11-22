# Streaming benchmark results

Run `python benchmark/run_stream_benchmarks.py --repeats 5 --out /tmp/numcompute-stream`.
Environment python 3.11.13, numpy 2.4.6, os Darwin 27.0.0, machine arm64, processor arm, cpu_count 8. Seed 42.

## Gini root split

The Python and NumPy implementations sort each feature once and sweep class counts.
Both score the same duplicate-value boundaries and NaN routes over four features.
The split feature, threshold, route and gain are checked before timing every case.
Data generation and list conversion happen before timing. Best of 5 timed
calls after one warm-up. The CSV also records mean and standard deviation.

| Rows | Python loop (ms) | Production NumPy (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 500 | 2.142 | 0.331 | 6.47× |
| 2,500 | 9.483 | 1.164 | 8.14× |
| 10,000 | 37.748 | 4.564 | 8.27× |

## Buffer size on a synthetic stream with drift

The 4,096 rows arrive in chunks of 64 with 2% missing features and
3% label noise. The rule reverses after row 2,048. Every pipeline uses
the same stream, imputer, scaler, seed and depth-five tree. Only the threshold
changes (128, 512, 1024 rows). Accuracy uses predictions made before learning from
each chunk. Coverage counts all rows predicted by each model. The accuracy columns
use the common 3,072 rows starting at chunk
17. Early post-drift covers the first
1,024 rows after the reversal.

| Buffer | Coverage | Common accuracy | Before drift | Early after drift | All after drift | Retrains | Median ms per retrain | Median total ms | Peak logged KiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 96.9% | 83.5% | 85.8% | 76.5% | 82.3% | 32 | 1.770 | 56.621 | 3.2 |
| 512 | 87.5% | 77.9% | 91.0% | 51.3% | 71.4% | 8 | 4.059 | 32.786 | 18.4 |
| 1024 | 75.0% | 65.1% | 91.5% | 9.9% | 51.9% | 4 | 5.082 | 20.145 | 38.6 |

Times include the entire training update on triggering chunks, excluding
prediction and metric logging. Timings summarise 5 complete replays
after one warm-up. The retained array estimate is the highest post-update
reading from the first timed replay. It excludes temporary training allocation,
Python overhead and metric state. The per-chunk CSV comes from that same replay.
The synthetic experiment measures a known change and does not establish expected
accuracy on a real weather stream. Timings depend on the hardware and workload.
