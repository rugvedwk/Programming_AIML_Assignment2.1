"""Classification trees with vectorised splits and buffered retraining."""

from dataclasses import dataclass

import numpy as np

from numcompute.pipeline import BaseEstimator, NotFittedError
from numcompute.utils import check_array, check_labels, check_same_length

from ._buffer import _TrainingBuffer, _wide_integer

__all__ = ["DecisionTreeClassifier"]

_GAIN_TOLERANCE = 32 * np.finfo(np.float64).eps


@dataclass(frozen=True, slots=True)
class _Split:
    feature: int
    threshold: np.generic
    missing_left: bool
    gain: float


@dataclass(frozen=True, slots=True)
class _Branch:
    split: _Split
    left: int
    right: int


@dataclass(frozen=True, slots=True)
class _Leaf:
    winner: int
    class_ids: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True, slots=True)
class _Tree:
    classes: np.ndarray
    n_features: int
    nodes: tuple


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")
    return int(value)


def _feature_count(max_features, n_features):
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError("max_features must be 'sqrt', 'log2', an integer, a fraction or None")
    if isinstance(max_features, (float, np.floating)):
        if not 0 < max_features <= 1:
            raise ValueError("a max_features fraction must be in (0, 1]")
        return max(1, int(max_features * n_features))
    count = _integer(max_features, "max_features", 1)
    if count > n_features:
        raise ValueError(f"max_features={count} exceeds X's {n_features} features")
    return count


def _class_occurrences(codes, counts):
    """For each row in feature order, count earlier rows of its class."""
    grouped = np.argsort(codes, kind="stable")
    offsets = np.cumsum(counts) - counts
    earlier = np.empty_like(codes)
    earlier[grouped] = np.arange(len(codes)) - offsets[codes[grouped]]
    return earlier


def _pairwise_prefix(values):
    """Compute prefix sums with a pairwise scan over disjoint strided views."""
    sums = values.copy()
    stride = 1
    # First combine adjacent blocks into power-of-two totals.
    while 2 * stride <= sums.size:
        ends = sums[2 * stride - 1::2 * stride]
        ends += sums[stride - 1::2 * stride][:ends.size]
        stride *= 2
    # Then add preceding block totals to the endpoints missing that prefix.
    while stride > 1:
        stride //= 2
        ends = sums[3 * stride - 1::2 * stride]
        ends += sums[2 * stride - 1::2 * stride][:ends.size]
    return sums


def _boundary_gains(codes, counts, missing_counts, boundaries, earlier, terms, increments,
                    criterion, missing_left):
    """Compute the gain at each boundary from the children's class terms.

    Weighted Gini depends on sum(c²) / child_size; weighted entropy depends
    on child_size * log2(child_size) - sum(c * log2(c)).
    """
    left_before = earlier + (missing_counts[codes] if missing_left else 0)
    right_after = counts[codes] - left_before - 1
    missing_term = terms[missing_counts].sum()
    left_terms = np.empty(len(codes) + 1)
    right_terms = np.empty_like(left_terms)
    left_terms[0] = missing_term if missing_left else 0.0
    right_terms[-1] = 0.0 if missing_left else missing_term
    # Each moved row changes only one class term. Sum from each end to avoid
    # subtracting two large totals for the small child at a lopsided split.
    # Entropy terms are fractional. Pairwise addition limits roundoff that
    # can otherwise make a long sequence of zero-gain boundaries look useful.
    prefix_sum = np.cumsum if criterion == "gini" else _pairwise_prefix
    left_terms[1:] = left_terms[0] + prefix_sum(increments[left_before])
    right_terms[:-1] = right_terms[-1] + prefix_sum(increments[right_after][::-1])[::-1]
    left_size = boundaries + (missing_counts.sum() if missing_left else 0)
    size = int(counts.sum())
    right_size = size - left_size
    parent_terms = terms[counts].sum()
    if criterion == "gini":
        improvement = (
            left_terms[boundaries] / left_size + right_terms[boundaries] / right_size
            - parent_terms / size
        )
    else:
        improvement = (
            terms[size] - terms[left_size] - terms[right_size]
            + left_terms[boundaries] + right_terms[boundaries] - parent_terms
        )
    return improvement / size


def _feature_split(feature, values, codes, counts, terms, increments, criterion):
    finite = ~np.isnan(values)
    observed = values[finite]
    if not observed.size:
        return None
    sorted_order = np.argsort(observed, kind="stable")
    observed = observed[sorted_order]

    boundaries = 1 + np.flatnonzero(observed[:-1] < observed[1:])
    sorted_codes = codes[finite][sorted_order]
    observed_counts = np.bincount(sorted_codes, minlength=len(counts))
    missing_counts = counts - observed_counts
    earlier = _class_occurrences(sorted_codes, observed_counts)
    has_missing = observed.size < values.size
    # With missing values sent right, the final observed value is also a
    # candidate. It permits a split whose signal is entirely missingness.
    right_boundaries = np.append(boundaries, len(observed)) if has_missing else boundaries
    if not right_boundaries.size:
        return None

    right = _boundary_gains(sorted_codes, counts, missing_counts, right_boundaries, earlier,
                            terms, increments, criterion, missing_left=False)
    left = np.full_like(right, -np.inf)
    if has_missing and boundaries.size:
        left[:-1] = _boundary_gains(sorted_codes, counts, missing_counts, boundaries, earlier,
                                    terms, increments, criterion, missing_left=True)
    gains = np.column_stack((right, left))
    best = gains.max()
    if best <= _GAIN_TOLERANCE:
        return None
    tied = (gains >= best - _GAIN_TOLERANCE) & (gains > _GAIN_TOLERANCE)
    boundary, route = divmod(int(np.argmax(tied)), 2)
    cut = right_boundaries[boundary]
    threshold = np.float64(np.inf) if cut == len(observed) else observed[cut - 1]
    return _Split(feature, threshold, bool(route), float(gains[boundary, route]))


def _best_split(X, rows, codes, counts, features, criterion):
    sizes = np.arange(len(codes) + 1, dtype=np.float64)
    if criterion == "gini":
        terms = sizes**2
    else:
        terms = sizes * np.log2(np.maximum(sizes, 1.0))
    increments = np.diff(terms)
    splits = []
    for feature in features:
        split = _feature_split(int(feature), X[rows, feature], codes, counts,
                               terms, increments, criterion)
        if split is not None:
            splits.append(split)
    if not splits:
        return None
    best_gain = max(split.gain for split in splits)
    return next(split for split in splits if split.gain >= best_gain - _GAIN_TOLERANCE)


def _left_mask(values, split):
    threshold = split.threshold
    common = np.result_type(values.dtype, threshold.dtype)
    if any(_wide_integer(dtype, common) for dtype in (values.dtype, threshold.dtype)):
        # Python's exact float/int comparison preserves large integer boundaries.
        left = values.astype(object) <= threshold.item()
    else:
        left = values.astype(common, copy=False) <= np.asarray(threshold, dtype=common)
    return np.where(np.isnan(values), split.missing_left, left)


def _make_leaf(counts):
    class_ids = np.flatnonzero(counts)
    return _Leaf(int(counts.argmax()), class_ids, counts[class_ids] / counts.sum())


def _build_nodes(X, codes, n_classes, feature_count, max_depth, min_samples_split,
                 criterion, rng):
    nodes = [None]
    pending = [(0, 0, np.arange(len(X)))]
    while pending:
        index, depth, rows = pending.pop()
        labels = codes[rows]
        counts = np.bincount(labels, minlength=n_classes)
        split = None
        if depth < max_depth and len(rows) >= min_samples_split and np.count_nonzero(counts) > 1:
            if feature_count == X.shape[1]:
                features = np.arange(X.shape[1])
            else:
                features = np.sort(rng.choice(X.shape[1], feature_count, replace=False))
            split = _best_split(X, rows, labels, counts, features, criterion)
        if split is None:
            nodes[index] = _make_leaf(counts)
            continue
        left = _left_mask(X[rows, split.feature], split)
        first_child = len(nodes)
        nodes[index] = _Branch(split, first_child, first_child + 1)
        nodes.extend((None, None))
        pending.append((first_child + 1, depth + 1, rows[~left]))
        pending.append((first_child, depth + 1, rows[left]))
    return tuple(nodes)


class DecisionTreeClassifier(BaseEstimator):
    """Grow a depth-limited binary tree for multiclass classification.

    Parameters
    ----------
    max_depth : int, default 5
        Maximum number of splits on a path from the root. Zero gives one leaf.
    min_samples_split : int, default 2
        Minimum number of samples a node needs to try a split.
    max_features : int, float, {'sqrt', 'log2'} or None, default None
        Features sampled without replacement at each eligible node. A fraction
        is multiplied by the total feature count and rounded down, with a
        minimum of one. An integer is the exact count. None considers all.
    buffer_size : int, default 500
        Positive number of pending rows that triggers streaming retraining.
        The training batch includes the entire triggering chunk.
    criterion : {'gini', 'entropy'}, default 'gini'
        Impurity used for weighted information gain. Entropy is measured in bits.
    random_state : int or None, default 0
        Nonnegative seed for feature sampling. An integer restarts the local
        generator on each fit. None uses NumPy's default entropy source.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Sorted class labels defining the probability columns and the order for
        tied votes. Available after fitting.
    n_features_in_ : int
        Feature count used at fit. Available after fitting.
    pending_rows_ : int
        Number of rows waiting for the next streaming retrain.
    n_retrains_ : int
        Number of successful threshold-triggered fits since construction or the
        last explicit fit. The first threshold-triggered fit counts.
    last_retrain_size_ : int
        Row count of the last streaming retrain, or zero before the first.

    Notes
    -----
    Thresholds use the lower observed endpoint of a split, preserving large
    integers and adjacent floating-point values. Nonmissing values at or below
    the threshold go left. The split search learns which side receives NaNs.
    A missingness-only split sends all nonmissing values left.
    Gains within 32 float64 epsilons tie. The lowest feature index, then the
    lowest threshold, then sending NaNs right take precedence.
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None, buffer_size=500,
                 *, criterion="gini", random_state=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.criterion = criterion
        self.random_state = random_state
        self._buffer = _TrainingBuffer()
        self.n_retrains_ = 0
        self.last_retrain_size_ = 0

    @property
    def classes_(self):
        """Read-only sorted class labels, with shape (n_classes,)."""
        return self._tree.classes

    @property
    def n_features_in_(self):
        """Number of input features observed at fit."""
        return self._tree.n_features

    @property
    def pending_rows_(self):
        """Number of rows waiting for the next streaming retrain."""
        return self._buffer.rows

    def _training_parameters(self, n_features):
        max_depth = _integer(self.max_depth, "max_depth", 0)
        min_samples_split = _integer(self.min_samples_split, "min_samples_split", 2)
        seed = None if self.random_state is None else _integer(self.random_state, "random_state", 0)
        if not isinstance(self.criterion, str) or self.criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        return (_feature_count(self.max_features, n_features), max_depth,
                min_samples_split, self.criterion, seed)

    @staticmethod
    def _training_arrays(X, y):
        X = check_array(X, ensure_2d=True, allow_inf=False)
        y = check_labels(y)
        check_same_length(X, y)
        if not X.shape[1]:
            raise ValueError("training requires at least one feature")
        return X, y

    def _train(self, X, y, parameters):
        """Build the replacement completely before publishing it."""
        feature_count, max_depth, min_samples_split, criterion, seed = parameters
        try:
            classes, codes = np.unique(y, return_inverse=True)
        except TypeError as exc:
            raise TypeError("y must contain mutually comparable labels") from exc
        nodes = _build_nodes(X, codes, len(classes), feature_count, max_depth,
                             min_samples_split, criterion, np.random.default_rng(seed))
        classes.setflags(write=False)
        self._tree = _Tree(classes, X.shape[1], nodes)

    def fit(self, X, y):
        """Fit a fresh tree to the entire batch and return self.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Real numeric or boolean matrix. NaNs are routed by learned splits.
        y : array_like, shape (n_samples,)
            Comparable finite numeric or string labels.

        Raises
        ------
        TypeError
            For unsupported feature, label or configuration types.
        ValueError
            For invalid shapes, empty axes, infinities or invalid settings.

        Notes
        -----
        Inputs are never modified. A successful fit drops pending rows and
        resets streaming counters. A failed fit preserves the model and buffer.
        At a node with m rows, k candidate features and c known classes, split
        search takes O(k * (m log m + c)) time and O(m + c) scratch per feature.
        Fit repeats the search at eligible nodes and stores splits and sparse
        leaf class frequencies.
        """
        _integer(self.buffer_size, "buffer_size", 1)
        X, y = self._training_arrays(X, y)
        if not len(X):
            raise ValueError("fit requires at least one sample")
        self._train(X, y, self._training_parameters(X.shape[1]))
        self._buffer.clear()
        self.n_retrains_ = 0
        self.last_retrain_size_ = 0
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """Add one chunk and retrain once if enough rows are pending.

        Parameters
        ----------
        X_chunk : array_like, shape (n_samples, n_features)
            Real numeric or boolean features with a consistent column count.
            NaNs are allowed and infinities are rejected.
        y_chunk : array_like, shape (n_samples,)
            Finite labels of one comparable kind within the buffer: numeric,
            text or bytes. A new trained batch may introduce different classes.

        Returns
        -------
        self : DecisionTreeClassifier

        Raises
        ------
        TypeError
            If labels, features or settings use incompatible types.
        ValueError
            If the input shape, values or settings are invalid, or combining
            feature dtypes would round integer values.

        Notes
        -----
        Empty chunks with valid shapes are no-ops. Pending data is copied.
        Reaching or exceeding buffer_size trains on all pending rows and
        all rows of this chunk, clears the buffer and publishes the new tree.
        Until then, predict and predict_proba use the current tree and raise
        NotFittedError if no tree has been trained. A failed update leaves the
        pending rows, counters and fitted tree intact. There is no implicit
        flush of a final short buffer.

        Validation and copying are linear in the chunk size. Changing the
        common feature dtype can require checking pending integer values.
        Buffer storage is linear in the pending data size. A retrain assembles
        at most one combined batch and incurs the batch fit cost.
        """
        threshold = _integer(self.buffer_size, "buffer_size", 1)
        X, y = self._training_arrays(X_chunk, y_chunk)
        if hasattr(self, "_tree"):
            self._check_n_features(X)
        parameters = self._training_parameters(X.shape[1])
        trained = self._buffer.submit(X, y, threshold, lambda X, y: self._train(X, y, parameters))
        if trained:
            self.n_retrains_ += 1
            self.last_retrain_size_ = trained
        return self

    def _prediction_leaves(self, X):
        self._check_is_fitted("_tree")
        X = check_array(X, ensure_2d=True, allow_inf=False)
        self._check_n_features(X)
        return X, self._walk(X)

    def _walk(self, X):
        """Route batches of rows, yielding each occupied leaf and its row indices."""
        if not len(X):
            return
        pending = [(0, np.arange(len(X)))]
        while pending:
            index, rows = pending.pop()
            node = self._tree.nodes[index]
            if isinstance(node, _Leaf):
                yield node, rows
                continue
            left = _left_mask(X[rows, node.split.feature], node.split)
            if left.any():
                pending.append((node.left, rows[left]))
            if not left.all():
                pending.append((node.right, rows[~left]))

    def predict(self, X):
        """Return the winning class for each row in X, with shape (n_samples,).

        X is a real numeric or boolean matrix of shape (n_samples, n_features).
        NaNs follow each split's learned route. Equal leaf votes pick the first
        label in classes_. Empty input gives an empty vector. Inputs are unchanged.
        Raises NotFittedError before fit, TypeError for unsupported dtypes, or
        ValueError for a bad shape, feature count or infinity. After validation
        of all features, routing costs O(n_samples * depth) time and
        O(n_samples) working space.
        """
        X, leaves = self._prediction_leaves(X)
        predictions = np.empty(len(X), dtype=self.classes_.dtype)
        for leaf, rows in leaves:
            predictions[rows] = self.classes_[leaf.winner]
        return predictions

    def predict_proba(self, X):
        """Return leaf class frequencies with shape (n_samples, n_classes).

        Columns match classes_. X uses the same numeric shape, NaN rules and
        errors as predict. Inputs are unchanged. Empty input gives an empty
        matrix with n_classes columns. After feature validation, routing costs
        O(n_samples * depth) time and O(n_samples) working space, plus
        O(n_samples * n_classes) time and space to construct the result.
        """
        X, leaves = self._prediction_leaves(X)
        probabilities = np.zeros((len(X), len(self.classes_)))
        for leaf, rows in leaves:
            probabilities[np.ix_(rows, leaf.class_ids)] = leaf.probabilities
        return probabilities
