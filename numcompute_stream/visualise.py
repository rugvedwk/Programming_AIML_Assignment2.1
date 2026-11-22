"""Small Matplotlib plots for chunk metrics and classification predictions."""

import numpy as np

from numcompute.utils import check_array

from .metrics import _make_batch

__all__ = ["plot_metric_over_time", "compare_models", "plot_predictions_vs_ground_truth"]


def _axes(ax):
    if ax is None:
        from matplotlib import pyplot as plt

        _, ax = plt.subplots(layout="constrained")
    return ax


def _finish(ax, save_path, show):
    if save_path is not None:
        ax.figure.savefig(save_path)
    if show:
        from matplotlib import pyplot as plt

        plt.show()
    return ax


def _series(values):
    values = check_array(values, dtype=np.float64, allow_inf=False)
    if values.ndim != 1:
        raise ValueError("metric values must have shape (n_chunks,)")
    return values


def plot_metric_over_time(metric_values, title, ylabel, *, ax=None, save_path=None, show=False):
    """Plot a 1-D metric trace against one-based chunk indices and return Axes.

    NaNs appear as gaps, including warm-up chunks. Invalid shapes or infinity
    raise ValueError. Provide an Axes to reuse it and a path to save the figure.
    Set show=True for immediate display or display the returned figure inline.
    Uses O(n_chunks) plotted values and preserves inputs.
    """
    values = _series(metric_values)
    ax = _axes(ax)
    ax.plot(np.arange(1, len(values) + 1), values, marker="o", markersize=3)
    ax.set(title=title, xlabel="Chunk", ylabel=ylabel)
    ax.set_xlim(0.5, max(len(values), 1) + 0.5)
    return _finish(ax, save_path, show)


def compare_models(metric1, metric2, labels, *, ax=None, save_path=None, show=False):
    """Compare two 1-D chunk traces and return Axes.

    labels supplies two legend strings. Traces may differ in length and both
    start at chunk 1. NaN gaps, display and saving work as in
    plot_metric_over_time. Work and plotted data are linear in trace lengths.
    """
    if not isinstance(labels, (list, tuple, np.ndarray)) or np.shape(labels) != (2,):
        raise ValueError("labels must contain two legend strings")
    if not all(isinstance(label, str) for label in labels):
        raise TypeError("labels must contain two legend strings")
    values = (_series(metric1), _series(metric2))
    ax = _axes(ax)
    for series, label in zip(values, labels):
        ax.plot(np.arange(1, len(series) + 1), series, marker="o", markersize=3, label=label)
    ax.set(title="Model comparison", xlabel="Chunk", ylabel="Metric")
    ax.set_xlim(0.5, max(1, len(values[0]), len(values[1])) + 0.5)
    ax.legend()
    return _finish(ax, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, *, ax=None, save_path=None, show=False):
    """Plot paired 1-D categorical labels by sample and return Axes.

    Labels are displayed as sorted category ticks. Invalid label types or
    mismatched lengths raise TypeError or ValueError. Work is O(n log n) to
    encode categories and O(n) for plotted points. Display/save options are
    the same as plot_metric_over_time. Inputs are unchanged.
    """
    batch = _make_batch(y_true, y_pred, None)
    labels, codes = np.unique(np.concatenate((batch.truth, batch.predicted)), return_inverse=True)
    truth, predicted = np.split(codes, 2)
    samples = np.arange(1, len(truth) + 1)
    ax = _axes(ax)
    ax.scatter(samples - 0.10, truth, marker="o", label="Ground truth")
    ax.scatter(samples + 0.10, predicted, marker="x", label="Prediction")
    ax.set(title="Predictions and ground truth", xlabel="Sample in chunk", ylabel="Class")
    ax.set_yticks(np.arange(len(labels)), labels=[str(label) for label in labels])
    ax.legend()
    return _finish(ax, save_path, show)
