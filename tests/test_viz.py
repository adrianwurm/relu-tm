"""Smoke tests for the figures (skipped without matplotlib)."""

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from relu_tm import build_network, run, viz  # noqa: E402
from relu_tm.examples import binary_adder, busy_beaver  # noqa: E402

RESULTS = Path(__file__).parents[1] / "experiments" / "results"


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_head_positions_follow_the_moves():
    tm = busy_beaver(2)
    assert viz.head_positions(tm, tm.trace(steps=6)) == [0, 1, 0, -1, -2, -1, 0]


def test_exactly_one_table_row_fires_per_step():
    tm = busy_beaver(4)
    net = build_network(tm)
    result = run(tm, network=net)
    matrix, labels = viz.transition_matrix(net, result.vectors)
    assert matrix.shape == (108, len(labels)) and "A, 0 (tail)" in labels
    assert np.array_equal(matrix.sum(axis=1), np.ones(108))


def test_figures_render():
    tm = binary_adder()
    net = build_network(tm)
    result = run(tm, "101+11", network=net)
    viz.use_style()
    viz.tape_diagram(tm, result.network, plt.figure().gca())
    _, axes = plt.subplots(3, 1)
    viz.step_inside(net, result.vectors[3], axes)
    viz.transition_heatmap(net, result.vectors, plt.figure().gca())
    viz.core_figure(tm, net, result.network, step=3)
    viz.capacity_plot(RESULTS / "numerics_step1_summary.csv")
    viz.error_plot(RESULTS / "numerics_step1_trace.csv")


def test_tape_diagram_limits_symbols():
    from relu_tm import TuringMachine

    tm = TuringMachine({("a", "_"): ("a", "w", "R")}, start="a", blank="_", symbols=("_", "x", "y", "z", "w"))
    with pytest.raises(ValueError):
        viz.tape_diagram(tm, tm.trace(steps=2), plt.figure().gca())
