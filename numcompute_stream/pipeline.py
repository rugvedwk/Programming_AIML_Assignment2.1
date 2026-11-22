"""Streaming composition with a fitted preprocessing and classifier snapshot."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from numcompute.pipeline import BaseEstimator, NotFittedError, _call_fit, _validate_named_steps
from numcompute.utils import check_categories, check_labels, check_same_length

from ._buffer import _TrainingBuffer
from ._validation import _integer

__all__ = ["Pipeline"]


@dataclass(frozen=True)
class _FittedPipeline:
    transformers: tuple
    classifier: object


def _input_matrix(X):
    X = check_categories(X, ensure_2d=True)
    if not X.shape[1]:
        raise ValueError("X must have at least one feature")
    return X


def _transform(name, step, X):
    result = np.asarray(step.transform(X))
    if result.ndim != 2 or result.shape[0] != len(X) or not result.shape[1]:
        raise ValueError(
            f"step {name!r} must return a matrix with {len(X)} rows and at least "
            f"one feature; got {result.shape}"
        )
    return result


def _transform_all(transformers, X):
    for name, step in transformers:
        X = _transform(name, step, X)
    return X


class Pipeline(BaseEstimator):
    """Train a classifier from raw chunks and deploy it with its transformers.

    Parameters
    ----------
    steps : sequence of (str, component)
        Unique, nonempty names. Supply unfitted, deep-copyable components.
        Transformers implement fit, partial_fit and transform. The final
        classifier implements fit and predict. For probability predictions it
        must also expose predict_proba and classes_. A classifier-only pipeline
        is allowed. Components preserve caller arrays, and transformations
        preserve the row count and order.
    buffer_size : int or None, default None
        Retrain threshold for raw rows. None uses the final classifier's
        buffer_size setting. A call crossing it trains once on the whole buffer.

    Attributes
    ----------
    pending_rows_ : int
        Raw rows waiting for the next retrain.
    n_retrains_ : int
        Number of successful buffered fits since initialization or explicit fit.
    last_retrain_size_ : int
        Row count of the last buffered fit, initially zero.
    n_features_in_ : int
        Raw feature count, available after the first nonempty accepted chunk.

    Notes
    -----
    Each transformer learns from the current chunk after its predecessors
    update and transform it. A downstream accumulator therefore sees its
    predecessors' outputs at arrival time. At a retrain, the latest states
    transform all pending raw rows consistently. The classifier is fitted on
    that batch and deployed with those states. A width-growing stage may
    precede only components that can accept its changing width.
    """

    def __init__(self, steps, *, buffer_size=None):
        steps = _validate_named_steps(steps, "streaming Pipeline")
        for index, (name, component) in enumerate(steps):
            if not name.strip():
                raise ValueError("step names must be nonempty")
            methods = ("fit", "predict") if index == len(steps) - 1 else (
                "fit", "partial_fit", "transform",
            )
            for method in methods:
                if not callable(getattr(component, method, None)):
                    raise TypeError(f"step {name!r} must implement {method}()")
        threshold = getattr(steps[-1][1], "buffer_size", None) if buffer_size is None else buffer_size
        self.buffer_size = _integer(threshold, "buffer_size", 1)
        self._templates = self._copy_steps(steps)
        self._learning = self._templates[:-1]
        self._fitted = None
        self._buffer = _TrainingBuffer()
        self.n_retrains_ = 0
        self.last_retrain_size_ = 0

    @staticmethod
    def _copy_steps(steps):
        return tuple((name, deepcopy(step)) for name, step in steps)

    @property
    def steps(self):
        """Return independent copies of the configured steps in order."""
        return self._copy_steps(self._templates)

    @property
    def named_steps(self):
        """Return inspectable copies of the fitted pair, or templates before fit."""
        if self._fitted is None:
            return dict(self.steps)
        model_name = self._templates[-1][0]
        steps = (*self._fitted.transformers, (model_name, self._fitted.classifier))
        return dict(self._copy_steps(steps))

    @property
    def pending_rows_(self):
        """Number of raw rows waiting for the next buffered fit."""
        return self._buffer.rows

    @property
    def fitted_(self):
        """Whether an explicit fit or buffered retrain has deployed a model."""
        return self._fitted is not None

    @property
    def classes_(self):
        """Return a copy of the fitted classifier's probability column labels."""
        return np.asarray(self._require_fitted().classifier.classes_).copy()

    def _require_fitted(self):
        if self._fitted is None:
            raise NotFittedError("Pipeline needs fit or a complete buffer before inference")
        return self._fitted

    @staticmethod
    def _training_arrays(X, y):
        X, y = _input_matrix(X), check_labels(y)
        check_same_length(X, y)
        return X, y

    def _fit_classifier(self, X, y):
        classifier = deepcopy(self._templates[-1][1])
        _call_fit(classifier.fit, X, y)
        return classifier

    def fit(self, X, y):
        """Fit on a nonempty raw matrix and a length-matched label vector.

        Returns self. Each stage learns from the previous fitted stage's
        output. Success deploys the new pair and resets pending rows and
        counters. Validation or stage failures preserve existing state.
        Inputs may be numeric, boolean or categorical matrices. Use an object
        matrix for mixed kinds. Stage-specific errors propagate. Time and
        scratch space are the sum of the stages' fit and transform costs.
        """
        X, y = self._training_arrays(X, y)
        if not len(X):
            raise ValueError("fit requires at least one sample")
        width = X.shape[1]
        transformers = self._copy_steps(self._templates[:-1])
        for name, step in transformers:
            _call_fit(step.fit, X, y)
            X = _transform(name, step, X)
        model = self._fit_classifier(X, y)
        self._learning = transformers
        self._fitted = _FittedPipeline(transformers, model)
        self.n_features_in_ = width
        self._buffer.clear()
        self.n_retrains_ = 0
        self.last_retrain_size_ = 0
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """Learn from a raw chunk and retrain once when the threshold is met.

        X_chunk is a matrix of shape (n_samples, n_features). y_chunk is a
        label vector of shape (n_samples,). Width stays fixed after the first
        admitted chunk. Compatible empty chunks do nothing. Returns self.

        Invalid inputs raise TypeError or ValueError, and component errors
        propagate without changing the learning state, buffer or fitted
        pair. The pipeline owns pending raw rows. Retraining costs one pass
        through the fitted transforms and one classifier fit on the assembled
        buffer. Each call also copies transformer state and updates it from the
        incoming chunk. The fitted classifier is copied only when training.
        """
        threshold = _integer(self.buffer_size, "buffer_size", 1)
        X, y = self._training_arrays(X_chunk, y_chunk)
        if hasattr(self, "n_features_in_"):
            self._check_n_features(X)
        if not len(X):
            return self
        learning = self._copy_steps(self._learning)
        transformed = X
        for index, (name, step) in enumerate(learning):
            _call_fit(step.partial_fit, transformed, y)
            if index < len(learning) - 1:
                transformed = _transform(name, step, transformed)

        candidate = None

        def train(raw, labels):
            nonlocal candidate
            model = self._fit_classifier(_transform_all(learning, raw), labels)
            candidate = _FittedPipeline(learning, model)

        trained = self._buffer.submit(X, y, threshold, train)
        self._learning = learning
        self.n_features_in_ = X.shape[1]
        if trained:
            self._fitted = candidate
            self.n_retrains_ += 1
            self.last_retrain_size_ = trained
        return self

    def _prepare(self, X):
        fitted = self._require_fitted()
        X = _input_matrix(X)
        self._check_n_features(X)
        return fitted, _transform_all(fitted.transformers, X)

    def transform(self, X):
        """Return the fitted classifier's input matrix for a raw matrix X.

        Output shape is (n_samples, n_transformed_features). A classifier-only
        pipeline returns the validated X, possibly sharing memory. Inference
        requires a fitted pair and consistent feature count. Stage costs and
        errors propagate. Inference does not update any learning state.
        """
        return self._prepare(X)[1]

    def predict(self, X):
        """Predict one label per row of X with the last fitted pair.

        Input rules and errors are those of transform. Time and scratch
        space include the transformer chain and classifier prediction.
        """
        fitted, X = self._prepare(X)
        return fitted.classifier.predict(X)

    def predict_proba(self, X):
        """Predict a (n_samples, n_classes) matrix in classes_ order.

        Input rules, errors and transform costs are those of predict. The
        fitted classifier must implement predict_proba. An absent method
        raises AttributeError.
        """
        fitted, X = self._prepare(X)
        return fitted.classifier.predict_proba(X)
