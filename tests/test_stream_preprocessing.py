"""Incremental preprocessing compared with batch and known outputs."""

from copy import deepcopy

import numpy as np
import pytest

from numcompute.pipeline import NotFittedError
from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler


@pytest.mark.parametrize("chunk_size", [1, 3, 14])
def test_scaler_matches_batch_across_chunk_boundaries(chunk_size):
    X = np.arange(42.0).reshape(14, 3)[::-1]
    X[::2, 0] = np.nan
    X[:, 1] = 0.1
    X.setflags(write=False)
    scaler = StandardScaler()
    for start in range(0, len(X), chunk_size):
        assert scaler.partial_fit(X[start:start + chunk_size]) is scaler
    batch = StandardScaler().fit(X)
    np.testing.assert_allclose(scaler.mean_, np.nanmean(X, axis=0))
    np.testing.assert_allclose(scaler.var_, np.nanvar(X, axis=0), atol=1e-28)
    np.testing.assert_array_equal(scaler.n_samples_seen_, [7, 14, 14])
    np.testing.assert_allclose(scaler.transform(X), batch.transform(X))
    np.testing.assert_array_equal(scaler.transform(X)[:, 1], np.zeros(14))


@pytest.mark.parametrize("config,expected", [
    ({"with_mean": False}, [[1, np.nan], [3, 2]]),
    ({"with_std": False}, [[-1, np.nan], [1, 0]]),
])
def test_scaler_flags_and_late_observations(config, expected):
    scaler = StandardScaler(**config).partial_fit([[1, np.nan]]).partial_fit([[3, 2]])
    np.testing.assert_allclose(scaler.transform([[1, np.nan], [3, 2]]), expected)


def test_scaler_preserves_finite_standardisation_at_float_limits():
    limit = np.finfo(float).max
    scaler = StandardScaler().partial_fit([[-limit]]).partial_fit([[limit]])
    np.testing.assert_array_equal(scaler.scale_, [limit])
    np.testing.assert_array_equal(scaler.var_, [np.inf])
    np.testing.assert_array_equal(scaler.transform([[-limit], [limit]]), [[-1], [1]])


@pytest.mark.parametrize("chunk_size", [1, 3, 5])
def test_imputer_matches_batch_even_with_unequal_valid_counts(chunk_size):
    X = np.array([[1, np.nan], [np.nan, 10], [3, np.nan], [5, 30], [7, 50]])
    imputer = Imputer()
    for start in range(0, len(X), chunk_size):
        imputer.partial_fit(X[start:start + chunk_size])
    np.testing.assert_array_equal(imputer.n_samples_seen_, [4, 3])
    np.testing.assert_allclose(imputer.statistics_, [4, 30])
    np.testing.assert_allclose(imputer.transform(X), Imputer().fit_transform(X))
    np.testing.assert_allclose(imputer.transform([[np.nan, 9]]), [[4, 9]])


def test_imputer_fallback_and_fit_reset():
    imputer = Imputer(fill_value=-5).partial_fit([[np.nan, 1]])
    np.testing.assert_array_equal(imputer.transform([[np.nan, np.nan]]), [[-5, 1]])
    imputer.partial_fit([[3, np.nan]])
    np.testing.assert_array_equal(imputer.transform([[np.nan, np.nan]]), [[3, 1]])
    imputer.fit([[np.nan, 8]])
    np.testing.assert_array_equal(imputer.transform([[np.nan, np.nan]]), [[-5, 8]])


@pytest.mark.parametrize("factory,X", [
    (StandardScaler, [[0, 1]]),
    (Imputer, [[0, np.nan]]),
    (OneHotEncoder, [["a", "b"]]),
])
def test_transformer_shape_empty_chunks_and_failed_update(factory, X):
    transformer = factory().partial_fit(np.empty((0, 2)))
    with pytest.raises(NotFittedError):
        transformer.transform(X)
    transformer.partial_fit(X)
    expected = transformer.transform(X)
    transformer.partial_fit(np.empty((0, 2)))
    assert transformer.transform(np.empty((0, 2))).shape == (0, expected.shape[1])
    with pytest.raises(ValueError, match="features"):
        transformer.partial_fit([[1]])
    with pytest.raises(ValueError, match="infinite"):
        transformer.partial_fit([[np.inf, 1]])
    np.testing.assert_array_equal(transformer.transform(X), expected)


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_encoder_output_ids_are_chunk_invariant(chunk_size):
    X = np.array([["B", 10], ["A", 20], ["C", 30], ["A", 10]], dtype=object)
    X.setflags(write=False)
    streamed = OneHotEncoder()
    for start in range(0, len(X), chunk_size):
        streamed.partial_fit(X[start:start + chunk_size])
    expected_pairs = ((0, "B"), (1, 10), (0, "A"), (1, 20), (0, "C"), (1, 30))
    assert streamed.categories_ == expected_pairs
    expected = [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1],
                [0, 1, 1, 0, 0, 0]]
    np.testing.assert_array_equal(streamed.transform(X), expected)
    np.testing.assert_array_equal(OneHotEncoder().fit_transform(X), expected)


def test_encoder_appends_nans_and_unseen_values_without_shifting_old_ids():
    X = np.array([["a", 10], ["b", 20]], dtype=object)
    encoder = OneHotEncoder().partial_fit(X)
    frozen = deepcopy(encoder)
    added = np.array([["c", np.nan]], dtype=object)
    np.testing.assert_array_equal(encoder.transform(added), [[0, 0, 0, 0]])
    encoder.partial_fit(added).partial_fit(added)
    assert len(encoder.categories_) == 6
    np.testing.assert_array_equal(encoder.transform(added), [[0, 0, 0, 0, 1, 1]])
    np.testing.assert_array_equal(encoder.transform(X)[:, :4], frozen.transform(X))
    np.testing.assert_array_equal(frozen.transform(added), [[0, 0, 0, 0]])


def test_encoder_handles_unknown_errors_and_preserves_state_on_invalid_update():
    encoder = OneHotEncoder(handle_unknown="error").fit(np.array([["a", 1]], dtype=object))
    with pytest.raises(ValueError, match="unseen"):
        encoder.transform(np.array([["b", 1]], dtype=object))
    with pytest.raises(TypeError, match="kind"):
        encoder.partial_fit(np.array([["b", "invalid"]], dtype=object))
    assert encoder.categories_ == ((0, "a"), (1, 1))
    encoder.fit([["fresh"]])
    assert encoder.categories_ == ((0, "fresh"),)


def test_encoder_distinguishes_large_integer_categories_from_rounded_floats():
    x = np.array([[np.uint64(2**53 + 1)], [np.float64(2**53)]], dtype=object)
    encoder = OneHotEncoder().fit(x)
    np.testing.assert_array_equal(encoder.transform(x), np.eye(2))
    streamed = OneHotEncoder().partial_fit(x[:1]).partial_fit(x[1:])
    np.testing.assert_array_equal(streamed.transform(x), np.eye(2))
