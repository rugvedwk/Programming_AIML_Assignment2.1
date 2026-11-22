"""Reference and boundary checks for row-stream statistics."""

import numpy as np
import pytest

from numcompute.pipeline import NotFittedError
from numcompute_stream.stats import StreamingStats, update_stats


@pytest.mark.parametrize("chunk_size", [1, 4, 23])
def test_cumulative_results_match_independent_batch_reducers(chunk_size):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(23, 3))
    X[::4, 0] = np.nan
    X[:, 1] = 3
    edges = [-2, -0.5, 0, 1, 3]
    state = StreamingStats(bins=edges)
    for start in range(0, len(X), chunk_size):
        assert update_stats(X[start:start + chunk_size], stats=state) is state
    np.testing.assert_array_equal(state.count, (~np.isnan(X)).sum(axis=0))
    np.testing.assert_allclose(state.mean, np.nanmean(X, axis=0))
    np.testing.assert_allclose(state.variance, np.nanvar(X, axis=0), rtol=1e-13)
    np.testing.assert_allclose(state.std, np.nanstd(X, axis=0), rtol=1e-13)
    q = [0, 0.25, 0.5, 1]
    np.testing.assert_allclose(state.quantile(q), np.nanquantile(X, q, axis=0))
    np.testing.assert_allclose(state.quantile(0.5), np.nanmedian(X, axis=0))
    counts, found_edges = state.histogram()
    np.testing.assert_array_equal(counts, [np.histogram(col, bins=edges)[0] for col in X.T])
    np.testing.assert_array_equal(found_edges, edges)
    assert state.retained_rows == len(X)


def test_histogram_boundaries_and_bounded_cumulative_storage():
    edges = np.array([0., 1., 2.])
    state = StreamingStats(bins=edges, track_quantiles=False)
    edges[:] = -1
    state.update([[-1, np.nan], [0, 0], [1, 2]])
    state.update([[2, 1], [3, np.nan]])
    counts, returned_edges = state.histogram()
    np.testing.assert_array_equal(counts, [[1, 2], [1, 2]])
    np.testing.assert_array_equal(returned_edges, [0, 1, 2])
    counts[:] = -1
    returned_edges[:] = -1
    np.testing.assert_array_equal(state.histogram()[0], [[1, 2], [1, 2]])
    assert state.retained_rows == 0
    with pytest.raises(ValueError, match="disabled"):
        state.quantile(0.5)


@pytest.mark.parametrize("size", [1, 5, 9])
def test_windows_match_last_rows_after_partial_and_whole_evictions(size):
    X = np.arange(30.0).reshape(15, 2)
    X[2::4, 0] = np.nan
    edges = [-1, 5, 15, 30]
    state = StreamingStats(window_size=size, bins=edges)
    for start, end in zip([0, 3, 6, 8], [3, 6, 8, 15]):
        state.update(X[start:end])
        expected = X[max(0, end - size):end]
        np.testing.assert_array_equal(state.count, (~np.isnan(expected)).sum(axis=0))
        for j, col in enumerate(expected.T):
            valid = col[~np.isnan(col)]
            mean, var, med = (np.mean(valid), np.var(valid), np.median(valid)) if len(valid) else (
                np.nan, np.nan, np.nan
            )
            np.testing.assert_allclose(
                [state.mean[j], state.variance[j], state.quantile(0.5)[j]], [mean, var, med],
            )
            np.testing.assert_array_equal(
                state.histogram()[0][j], np.histogram(valid, bins=edges)[0],
            )
        assert state.retained_rows == min(end, size)


def test_all_missing_then_valid_data_and_result_ownership():
    X = np.array([[2, np.nan], [4, np.nan]])
    state = update_stats(X)
    X[:] = -99
    np.testing.assert_array_equal(state.count, [2, 0])
    np.testing.assert_allclose(state.mean, [3, np.nan])
    np.testing.assert_allclose(state.variance, [1, np.nan])
    np.testing.assert_allclose(state.quantile([0.25, 0.75]), [[2.5, np.nan], [3.5, np.nan]])
    assert state.quantile([]).shape == (0, 2)
    state.count[:] = -1
    state.update([[np.nan, 6]])
    np.testing.assert_array_equal(state.count, [2, 1])
    np.testing.assert_allclose(state.mean, [3, 6])


def test_opposite_sign_batch_means_preserve_small_residual():
    # Subtracting the two centers first rounds away the +2 and gives 0.
    state = StreamingStats(track_quantiles=False).update([[1e16]]).update([[-1e16 + 2]])
    np.testing.assert_array_equal(state.mean, [1])


def test_extreme_spread_has_finite_std_even_when_variance_overflows():
    limit = np.finfo(float).max
    state = StreamingStats(track_quantiles=False).update([[-limit]]).update([[limit]])
    np.testing.assert_array_equal(state.mean, [0])
    np.testing.assert_array_equal(state.std, [limit])
    np.testing.assert_array_equal(state.variance, [np.inf])


def test_empty_and_invalid_updates_preserve_cumulative_state():
    state = StreamingStats().update(np.empty((0, 4)))
    with pytest.raises(NotFittedError):
        _ = state.mean
    state.update([[5, 6]])
    state.update(np.empty((0, 2)))
    for bad in ([[np.inf, 1]], [[1]], [1, 2]):
        with pytest.raises(ValueError):
            state.update(bad)
    np.testing.assert_array_equal(state.count, [1, 1])
    np.testing.assert_array_equal(state.mean, [5, 6])
    assert state.retained_rows == 1
    with pytest.raises(ValueError, match="bins"):
        state.histogram()


@pytest.mark.parametrize("kwargs,error", [
    ({"bins": [0, 0]}, ValueError),
    ({"bins": [0, np.inf]}, ValueError),
    ({"bins": [[0, 1]]}, ValueError),
    ({"window_size": 0}, ValueError),
    ({"window_size": True}, TypeError),
    ({"track_quantiles": "yes"}, TypeError),
])
def test_stats_reject_invalid_configuration(kwargs, error):
    with pytest.raises(error):
        StreamingStats(**kwargs)
