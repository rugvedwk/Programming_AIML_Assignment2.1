"""Check plot data, existing axes and headless file output."""

import matplotlib
import numpy as np
import pytest

from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_metric_plot_retains_warmup_gaps_and_saves_png(tmp_path):
    path = tmp_path / "accuracy.png"
    ax = plot_metric_over_time([np.nan, 0.8, 0.9], "Online evaluation", "Accuracy", save_path=path)
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [np.nan, 0.8, 0.9])
    assert ax.get_xlim() == (0.5, 3.5)
    assert ax.get_title() == "Online evaluation" and ax.get_ylabel() == "Accuracy"
    assert path.read_bytes().startswith(b"\x89PNG")


def test_compare_models_uses_requested_axes_and_legend(monkeypatch):
    _, ax = plt.subplots()
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    assert compare_models([0.2, 0.4], [0.3], ("buffer 30", "buffer 60"), ax=ax, show=True) is ax
    assert shown == [True]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["buffer 30", "buffer 60"]
    assert [len(line.get_xdata()) for line in ax.lines] == [2, 1]


def test_predictions_plot_supports_text_labels_and_keeps_pairs_visible(tmp_path):
    path = tmp_path / "predictions.svg"
    ax = plot_predictions_vs_ground_truth(["sun", "wet"], ["wet", "wet"], save_path=path)
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["sun", "wet"]
    assert len(ax.collections) == 2
    np.testing.assert_array_equal(ax.collections[0].get_offsets()[:, 1], [0, 1])
    np.testing.assert_array_equal(ax.collections[1].get_offsets()[:, 1], [1, 1])
    assert "<svg" in path.read_text()


@pytest.mark.parametrize("plot,args", [
    (plot_metric_over_time, ([[0.5]], "title", "accuracy")),
    (plot_metric_over_time, ([np.inf], "title", "accuracy")),
    (compare_models, ([1], [1], ["one"])),
    (plot_predictions_vs_ground_truth, ([0, 1], [0])),
])
def test_invalid_plot_inputs_are_rejected(plot, args):
    with pytest.raises(ValueError):
        plot(*args)
