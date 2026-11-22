"""Measure root split search and buffer-size effects on a seeded stream."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from math import isnan
from pathlib import Path

import numpy as np

from numcompute.benchmarking import SEED, benchmark, environment_info
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier, _best_split, _GAIN_TOLERANCE


def loop_split(rows, labels):
    """Return (feature, threshold, NaN-left, gain) using a Python Gini sweep."""
    total = Counter(labels)
    size = len(labels)
    parent_term = sum(count * count for count in total.values()) / size
    winners = []
    for feature, column in enumerate(zip(*rows)):
        observed, missing = [], Counter()
        for value, label in zip(column, labels):
            if isnan(value):
                missing[label] += 1
            else:
                observed.append((value, label))
        observed.sort(key=lambda pair: pair[0])
        if not observed:
            continue
        missing_size = size - len(observed)
        candidates = []
        for missing_left in (False, True) if missing_size else (False,):
            left_counts = missing.copy() if missing_left else Counter()
            right_counts = total - left_counts
            left_term = sum(count * count for count in left_counts.values())
            right_term = sum(count * count for count in right_counts.values())
            left_size = missing_size if missing_left else 0
            for i, (value, label) in enumerate(observed):
                # Moving one label changes one squared count on each side.
                left_term += 2 * left_counts[label] + 1
                right_term += 1 - 2 * right_counts[label]
                left_counts[label] += 1
                right_counts[label] -= 1
                left_size += 1
                if i + 1 == len(observed):
                    if not missing_size or missing_left:
                        continue
                    value = float("inf")
                elif value == observed[i + 1][0]:
                    continue
                right_size = size - left_size
                gain = (left_term / left_size + right_term / right_size - parent_term) / size
                if gain > _GAIN_TOLERANCE:
                    candidates.append((feature, value, missing_left, gain))
        if candidates:
            best_gain = max(candidate[-1] for candidate in candidates)
            eligible = (entry for entry in candidates if entry[-1] >= best_gain - _GAIN_TOLERANCE)
            winners.append(min(eligible, key=lambda entry: (entry[1], entry[2])))
    if not winners:
        return None
    best_gain = max(entry[-1] for entry in winners)
    return next(entry for entry in winners if entry[-1] >= best_gain - _GAIN_TOLERANCE)


def vectorised_split(X, codes):
    """Run the production root split kernel with nonnegative class codes."""
    split = _best_split(X, np.arange(len(X)), codes, np.bincount(codes),
                        np.arange(X.shape[1]), "gini")
    if split is None:
        return None
    return split.feature, split.threshold, split.missing_left, split.gain


def _split_cases(sizes, repeats):
    rng = np.random.default_rng(SEED)
    results = []
    for size in sizes:
        X = rng.normal(size=(size, 4)).round(2)
        codes = np.digitize(X[:, 0] + 0.3 * X[:, 1], [-0.5, 0.5])
        X[rng.random(X.shape) < 0.05] = np.nan
        rows, labels = X.tolist(), codes.tolist()
        loop = loop_split(rows, labels)
        vector = vectorised_split(X, codes)
        if loop is None or vector is None or loop[:3] != vector[:3]:
            raise AssertionError(f"split mismatch for {size} rows: {loop!r}, {vector!r}")
        np.testing.assert_allclose(loop[3], vector[3], rtol=0, atol=1e-12)
        loop_time = benchmark(loop_split, rows, labels, repeats=repeats, number=1, warmup=1)
        vector_time = benchmark(vectorised_split, X, codes, repeats=repeats, number=1, warmup=1)
        results.append({
            "rows": size, "features": X.shape[1], "gain": vector[3],
            "loop_best_ms": loop_time["min"] * 1000,
            "vector_best_ms": vector_time["min"] * 1000,
            "loop_mean_ms": loop_time["mean"] * 1000,
            "vector_mean_ms": vector_time["mean"] * 1000,
            "loop_std_ms": loop_time["std"] * 1000,
            "vector_std_ms": vector_time["std"] * 1000,
            "best_speedup": loop_time["min"] / vector_time["min"],
        })
    return results


def _drift_stream(rows, chunk_size):
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(rows, 4))
    # The feature distribution stays fixed. Halfway through, the label
    # rule reverses. A little noise prevents trivial perfect accuracy.
    target = X[:, 0] + 0.35 * X[:, 1] > 0.15
    target[rows // 2:] ^= True
    target ^= rng.random(rows) < 0.03
    X[rng.random(X.shape) < 0.02] = np.nan
    return tuple((X[start:start + chunk_size], target[start:start + chunk_size].astype(int))
                 for start in range(0, rows, chunk_size))


def _replay(chunks, buffer_size):
    model = Pipeline([
        ("impute", Imputer()), ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=buffer_size, max_depth=5, random_state=SEED)),
    ])
    # The experiment times retraining and compares accuracy. Exact AUC is
    # disabled here because it is unrelated to the buffer-size comparison.
    trainer = StreamTrainer(model, track_auc=False)
    for X, y in chunks:
        trainer.fit_chunk(X, y)
    return trainer.logs


def _weighted_accuracy(entries):
    rows = sum(entry.rows for entry in entries)
    return sum(entry.rows * entry.chunk_accuracy for entry in entries) / rows if rows else np.nan


def _buffer_study(rows, chunk_size, buffers, repeats):
    chunks = _drift_stream(rows, chunk_size)
    runs = {}
    for size in buffers:
        _replay(chunks, size)  # untimed warm-up replay
        runs[size] = [_replay(chunks, size) for _ in range(repeats)]
        reference = runs[size][0]
        for run in runs[size][1:]:
            np.testing.assert_allclose([log.chunk_accuracy for log in run],
                                       [log.chunk_accuracy for log in reference])
            if [log.retrained for log in run] != [log.retrained for log in reference]:
                raise AssertionError("retrain schedule changed between identical replays")
    traces = {size: runs[size][0] for size in buffers}
    starts = [next(entry.chunk for entry in log if np.isfinite(entry.chunk_accuracy))
              for log in traces.values()]
    first_common = max(starts)
    early_end = rows // 2 + max(buffers)
    tables = []
    for size, trace in traces.items():
        shared = trace[first_common - 1:]
        assert all(np.isfinite(log.chunk_accuracy) for log in shared)
        pre = [log for log in shared if log.rows_seen <= rows // 2]
        post = [log for log in shared if log.rows_seen > rows // 2]
        early = [log for log in post if log.rows_seen <= early_end]
        events = [[log.retrain_seconds for log in run if log.retrained] for run in runs[size]]
        tables.append({
            "buffer_size": size, "scored_rows": trace[-1].scored_samples,
            "coverage": trace[-1].scored_samples / rows,
            "first_common_chunk": first_common, "common_rows": sum(log.rows for log in shared),
            "common_accuracy": _weighted_accuracy(shared), "pre_drift_accuracy": _weighted_accuracy(pre),
            "early_post_drift_accuracy": _weighted_accuracy(early),
            "post_drift_accuracy": _weighted_accuracy(post),
            "retrains": len(events[0]),
            "median_retrain_ms": float(np.median(events)) * 1000,
            "median_total_retrain_ms": float(np.median([sum(event) for event in events])) * 1000,
            "peak_logged_array_kib": max(log.model_array_bytes for log in trace) / 1024,
        })
    trace_rows = [{"buffer_size": size, **asdict(log)} for size, trace in traces.items() for log in trace]
    return tables, trace_rows


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path, split, buffer, rows, chunk_size, repeats, quick):
    info = environment_info()
    environment = ", ".join(f"{key} {value}" for key, value in info.items())
    sizes = ", ".join(str(row["buffer_size"]) for row in buffer)
    split_lines = [
        f"| {row['rows']:,} | {row['loop_best_ms']:.3f} | {row['vector_best_ms']:.3f} "
        f"| {row['best_speedup']:.2f}× |" for row in split
    ]
    buffer_lines = [
        f"| {row['buffer_size']} | {row['coverage']:.1%} | {row['common_accuracy']:.1%} "
        f"| {row['pre_drift_accuracy']:.1%} | {row['early_post_drift_accuracy']:.1%} "
        f"| {row['post_drift_accuracy']:.1%} | {row['retrains']} "
        f"| {row['median_retrain_ms']:.3f} | {row['median_total_retrain_ms']:.3f} "
        f"| {row['peak_logged_array_kib']:.1f} |" for row in buffer
    ]
    mode = " --quick" if quick else ""
    text = f"""# Streaming benchmark results

Run `python benchmark/run_stream_benchmarks.py{mode} --repeats {repeats} --out /tmp/numcompute-stream`.
Environment {environment}. Seed {SEED}.

## Gini root split

The Python and NumPy implementations sort each feature once and sweep class counts.
Both score the same duplicate-value boundaries and NaN routes over four features.
The split feature, threshold, route and gain are checked before timing every case.
Data generation and list conversion happen before timing. Best of {repeats} timed
calls after one warm-up. The CSV also records mean and standard deviation.

| Rows | Python loop (ms) | Production NumPy (ms) | Speedup |
| ---: | ---: | ---: | ---: |
{chr(10).join(split_lines)}

## Buffer size on a synthetic stream with drift

The {rows:,} rows arrive in chunks of {chunk_size} with 2% missing features and
3% label noise. The rule reverses after row {rows // 2:,}. Every pipeline uses
the same stream, imputer, scaler, seed and depth-five tree. Only the threshold
changes ({sizes} rows). Accuracy uses predictions made before learning from
each chunk. Coverage counts all rows predicted by each model. The accuracy columns
use the common {buffer[0]['common_rows']:,} rows starting at chunk
{buffer[0]['first_common_chunk']}. Early post-drift covers the first
{max(row['buffer_size'] for row in buffer):,} rows after the reversal.

| Buffer | Coverage | Common accuracy | Before drift | Early after drift | All after drift | Retrains | Median ms per retrain | Median total ms | Peak logged KiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(buffer_lines)}

Times include the entire training update on triggering chunks, excluding
prediction and metric logging. Timings summarise {repeats} complete replays
after one warm-up. The retained array estimate is the highest post-update
reading from the first timed replay. It excludes temporary training allocation,
Python overhead and metric state. The per-chunk CSV comes from that same replay.
The synthetic experiment measures a known change and does not establish expected
accuracy on a real weather stream. Timings depend on the hardware and workload.
"""
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use smaller inputs for CI")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("stream-benchmark-output"))
    options = parser.parse_args()
    if options.repeats < 1:
        parser.error("--repeats must be at least one")
    sizes = (128, 512) if options.quick else (500, 2_500, 10_000)
    rows, chunk, buffers = (512, 16, (32, 64, 128)) if options.quick else (
        4_096, 64, (128, 512, 1_024),
    )
    split = _split_cases(sizes, options.repeats)
    buffer, trace = _buffer_study(rows, chunk, buffers, options.repeats)
    options.out.mkdir(parents=True, exist_ok=True)
    _write_csv(options.out / "stream_split.csv", split)
    _write_csv(options.out / "stream_buffers.csv", buffer)
    _write_csv(options.out / "stream_chunks.csv", trace)
    report = options.out / "stream_results.md"
    _write_report(report, split, buffer, rows, chunk, options.repeats, options.quick)
    print(f"Verified split agreement and repeated stream results. Report: {report}")


if __name__ == "__main__":
    main()
