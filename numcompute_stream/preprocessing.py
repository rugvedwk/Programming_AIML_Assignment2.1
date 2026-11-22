"""Incremental transformers for numeric and categorical feature matrices."""

from numbers import Real

import numpy as np

from numcompute.pipeline import BaseTransformer
from numcompute.preprocessing import _check_flag, _missing_mask, _standardise
from numcompute.utils import check_categories

from ._validation import _discrete_kind, _numeric_matrix
from .stats import _Moments

__all__ = ["StandardScaler", "Imputer", "OneHotEncoder"]


class _MomentTransformer(BaseTransformer):
    def fit(self, X, y=None):
        """Replace learned state from a nonempty (n_samples, n_features) matrix.

        Ignores y, returns self and omits NaNs column by column. Raises
        TypeError for nonnumeric input and ValueError for an invalid shape,
        an empty batch or infinity. Time and scratch space are O(X.size).
        The input is unchanged.
        """
        X = _numeric_matrix(X)
        if not len(X):
            raise ValueError("fit requires at least one sample")
        self._moments = _Moments.from_batch(X)
        self.n_features_in_ = X.shape[1]
        return self

    def partial_fit(self, X, y=None):
        """Update per-feature moments from a chunk and return self.

        Accepts the same matrix and y as fit. Empty valid chunks are no-ops.
        Raises ValueError for changes in feature count and the invalid values
        listed in fit. Time and scratch space are O(X.size + n_features).
        Retained state is O(n_features). The input is unchanged.
        """
        X = _numeric_matrix(X)
        if hasattr(self, "_moments"):
            self._check_n_features(X)
        if len(X):
            batch = _Moments.from_batch(X)
            self._moments = self._moments.merge(batch) if hasattr(self, "_moments") else batch
            self.n_features_in_ = X.shape[1]
        return self

    @property
    def n_samples_seen_(self):
        """Return a copy of non-NaN counts with shape (n_features,)."""
        self._check_is_fitted("_moments")
        return self._moments.count.copy()

    def _transform_input(self, X):
        self._check_is_fitted("_moments")
        X = _numeric_matrix(X)
        self._check_n_features(X)
        return X


class StandardScaler(_MomentTransformer):
    """Learn population moments and standardise columns incrementally.

    with_mean and with_std are booleans controlling centering and scaling.
    mean_, var_ and scale_ are arrays of shape (n_features,). Empty columns
    have a zero center and unit scale. Constant columns have unit scale.
    All calculations use float64. NaNs survive transform and infinity in
    inputs is rejected. var_ may be infinite while scale_ is still finite.
    """

    def __init__(self, with_mean=True, with_std=True):
        _check_flag(with_mean, "with_mean")
        _check_flag(with_std, "with_std")
        self.with_mean = with_mean
        self.with_std = with_std

    @property
    def mean_(self):
        """Observed centers after fit, using zero for unobserved columns."""
        self._check_is_fitted("_moments")
        return self._moments.mean.copy()

    @property
    def var_(self):
        """Population variances after fit, with NaN for unobserved columns."""
        self._check_is_fitted("_moments")
        return np.where(self._moments.count, self._moments.variance(), np.nan)

    @property
    def scale_(self):
        """Standard deviation or unit fallback for each column, after fit."""
        self._check_is_fitted("_moments")
        if not self.with_std:
            return np.ones(self.n_features_in_)
        return np.where(self._moments.std > 0, self._moments.std, 1.0)

    def transform(self, X):
        """Return a new float64 array of the same shape, preserving NaNs.

        Input must have shape (n_samples, n_features) and match fitted width.
        Raises NotFittedError before a nonempty training chunk and TypeError
        or ValueError for invalid inputs. Time and space are O(X.size + d).
        An unrepresentable scaled value is returned as signed infinity.
        """
        X = self._transform_input(X)
        center = self._moments.mean if self.with_mean else 0.0
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            return _standardise(X, center, self.scale_)


class Imputer(_MomentTransformer):
    """Fill numeric NaNs with the running column mean.

    fill_value is a finite real fallback, used until a column has a valid
    value. Its default is zero. All calculations and outputs use float64.
    """

    def __init__(self, fill_value=0.0):
        if not isinstance(fill_value, Real):
            raise TypeError("fill_value must be a real number")
        try:
            finite = np.isfinite(float(fill_value))
        except OverflowError as exc:
            raise ValueError("fill_value must be representable as a finite float64") from exc
        if not finite:
            raise ValueError("fill_value must be finite")
        self.fill_value = fill_value

    @property
    def statistics_(self):
        """Return current fill values with shape (n_features,), after fit."""
        self._check_is_fitted("_moments")
        return np.where(self._moments.count, self._moments.mean, float(self.fill_value))

    def transform(self, X):
        """Replace NaNs in a new float64 array with the same matrix shape.

        Raises NotFittedError before a nonempty training chunk and TypeError
        or ValueError for invalid dtype, shape, feature count or infinity.
        Time and space are O(X.size + n_features). Input is unchanged.
        """
        X = self._transform_input(X)
        return np.where(np.isnan(X), self.statistics_, X)


def _category_groups(column):
    """Encode distinct values once, keeping their first row and inverse codes."""
    missing = _missing_mask(column, np.nan)
    rows = np.flatnonzero(~missing)
    present = column[rows]
    if present.dtype.kind == "O":
        # Python integers and floats compare exactly across dtypes.
        present = np.fromiter(
            (value.item() if isinstance(value, np.generic) else value for value in present),
            dtype=object, count=len(present),
        )
    kind = _discrete_kind(present) if len(present) else None
    distinct, first, indices = np.unique(present, return_index=True, return_inverse=True)
    keys = distinct.tolist()
    first_rows = rows[first]
    codes = np.empty(len(column), dtype=np.intp)
    codes[rows] = indices
    if missing.any():
        codes[missing] = len(keys)
        keys.append(None)  # Input validation reserves None for the NaN bucket.
        first_rows = np.append(first_rows, np.argmax(missing))
    return keys, first_rows, codes, kind


class OneHotEncoder(BaseTransformer):
    """Append output columns as new categorical values arrive.

    handle_unknown is "ignore" (default) to emit zeros for an unseen category,
    or "error" to raise ValueError. NaN is a separate learnable category.
    Each input column uses one comparable kind of value: numbers, strings
    or bytes. Use an object matrix for columns with different kinds.

    Output column IDs are assigned in first-seen order across rows and input
    columns. An update never shifts an existing column. categories_ gives the
    (feature_index, value) pairs in output order.
    """

    def __init__(self, handle_unknown="ignore"):
        if handle_unknown not in ("ignore", "error"):
            raise ValueError("handle_unknown must be 'ignore' or 'error'")
        self.handle_unknown = handle_unknown

    @staticmethod
    def _matrix(X):
        X = check_categories(X, ensure_2d=True)
        if not X.shape[1]:
            raise ValueError("X must have at least one feature")
        return X

    @staticmethod
    def _groups(X):
        for feature in range(X.shape[1]):
            try:
                yield _category_groups(X[:, feature])
            except TypeError as exc:
                raise TypeError(f"feature {feature} contains incomparable categories") from exc

    def _learn(self, X, ids, kinds):
        additions = []
        for feature, (keys, first_rows, _, kind) in enumerate(self._groups(X)):
            if kind is not None:
                if kinds[feature] is not None and kind != kinds[feature]:
                    raise TypeError(f"feature {feature} changed category kind")
                kinds[feature] = kind
            for key, row in zip(keys, first_rows):
                pair = (feature, key)
                if pair not in ids:
                    additions.append((row, feature, pair))
        # All columns are validated before the existing vocabulary can change.
        additions.sort(key=lambda entry: entry[:2])
        for _, _, pair in additions:
            ids[pair] = len(ids)
        self._ids = ids
        self._kinds = kinds
        self.n_features_in_ = X.shape[1]
        return self

    def fit(self, X, y=None):
        """Replace the vocabulary from a nonempty (n_samples, n_features) matrix.

        Ignores y and returns self. Invalid shape, infinity or empty training data
        raise ValueError. Unsupported or incomparable categories raise TypeError.
        Time is O(d*n*log(n+1) + u*log(u+1)) for u distinct additions and
        scratch space is O(n + u), apart from optional input conversion.
        """
        X = self._matrix(X)
        if not len(X):
            raise ValueError("fit requires at least one sample")
        return self._learn(X, {}, [None] * X.shape[1])

    def partial_fit(self, X, y=None):
        """Learn from a chunk using the same input, errors and cost as fit.

        Empty valid chunks do nothing. A changed feature width raises ValueError.
        Older IDs keep their positions. Retains O(K + d) vocabulary state for K
        encoded categories across d features. Does not preserve raw input rows.
        """
        X = self._matrix(X)
        fitted = hasattr(self, "_ids")
        if fitted:
            self._check_n_features(X)
        if not len(X):
            return self
        ids = self._ids if fitted else {}
        kinds = self._kinds.copy() if fitted else [None] * X.shape[1]
        return self._learn(X, ids, kinds)

    @property
    def categories_(self):
        """Return (feature index, category) pairs in output column order."""
        self._check_is_fitted("_ids")
        return tuple((feature, np.nan if value is None else value)
                     for feature, value in self._ids)

    def transform(self, X):
        """Return a float64 array of shape (n_samples, n_encoded_categories).

        NaNs use their learned category ID. Unknown categories follow
        handle_unknown. Raises NotFittedError before fitting, TypeError for
        incompatible categories, and ValueError for shape mismatch, infinity,
        or an unknown category when handle_unknown="error". Time and space
        include the dense output, O(n_samples * n_encoded_categories).
        Grouping the input adds O(d*n*log(n+1)) time and O(n) scratch space.
        """
        self._check_is_fitted("_ids")
        X = self._matrix(X)
        self._check_n_features(X)
        result = np.zeros((len(X), len(self._ids)))
        rows = np.arange(len(X))
        for feature, (keys, _, codes, _) in enumerate(self._groups(X)):
            lookup = np.fromiter((self._ids.get((feature, key), -1) for key in keys),
                                 dtype=np.intp, count=len(keys))
            encoded = lookup[codes]
            known = encoded >= 0
            if self.handle_unknown == "error" and not known.all():
                raise ValueError(f"feature {feature} has an unseen category")
            result[rows[known], encoded[known]] = 1
        return result
