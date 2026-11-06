"""Figures for the notebook and the README (needs matplotlib: ``pip install relu-tm[notebook]``).

All plots use one light palette (validated for colour-vision deficiencies):
the first three categorical slots for identity, a single blue ramp for
magnitude, hairline grey chrome.  Identity is never colour alone - every
multi-series plot also has a legend with distinct marker shapes.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle

from .encoding import encode
from .machine import Configuration, TuringMachine
from .network import CompiledNetwork
from .numerics import pass_of

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"  # blank cells, pass shading
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3 (pass all-pairs)
CRITICAL = "#d03b3b"
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
MARKERS = ("o", "s", "^")

PASS_LABELS = {"read": "read", "table": "table", "scale": "scale", "left": "left", "state": "state"}


def use_style() -> None:
    """Apply the palette and hairline chrome to matplotlib."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.labelcolor": INK_2,
            "lines.linewidth": 2,
            "font.size": 9,
            "text.color": INK,
        }
    )


# ---------------------------------------------------------------------------
# The tape over time
# ---------------------------------------------------------------------------


def head_positions(tm: TuringMachine, configs: Sequence[Configuration]) -> list[int]:
    """Absolute head position of each configuration of a trace (start = 0)."""
    pos, out = 0, [0]
    for c in configs[:-1]:
        act = tm.action(c.state, c.head_symbol)
        if act is not None:
            pos += {"L": -1, "R": 1, "S": 0}[act[2]]
        out.append(pos)
    return out


def tape_diagram(
    tm: TuringMachine, configs: Sequence[Configuration], ax=None, title: str = "", every: int = 1
) -> plt.Axes:
    """Space-time diagram: one row per step (time runs down), one column per
    cell; blank cells light grey, other symbols in categorical colours, the
    head cell outlined."""
    ax = ax or plt.gca()
    heads = head_positions(tm, configs)
    rows = []
    for c, h in zip(configs, heads, strict=True):
        left, right = tuple(reversed(c.left)), c.right or (tm.blank,)
        rows.append({h - len(left) + i: s for i, s in enumerate(left + right)})
    lo = min(min(r) for r in rows if r)
    hi = max(max(r) for r in rows if r)
    lo, hi = min(lo, min(heads)), max(hi, max(heads))
    symbols = [s for s in tm.symbols if s != tm.blank]
    if len(symbols) > len(SERIES):
        raise ValueError("tape_diagram shows at most three non-blank symbols")
    index = {tm.blank: 0, **{s: i + 1 for i, s in enumerate(symbols)}}
    grid = np.zeros((len(rows), hi - lo + 1))
    for t, r in enumerate(rows):
        for p, s in r.items():
            grid[t, p - lo] = index[s]
    cmap = ListedColormap([NEUTRAL, *SERIES[: len(symbols)]])
    ax.imshow(grid[::every], cmap=cmap, vmin=-0.5, vmax=len(symbols) + 0.5, aspect="auto", interpolation="nearest")
    for col in range(hi - lo):  # 2px surface gaps between the cells
        ax.axvline(col + 0.5, color=SURFACE, linewidth=1.5)
    ts = list(range(0, len(rows), every))
    ax.plot(
        [heads[t] - lo for t in ts],
        [t // every for t in ts],
        linestyle="none",
        marker="o",
        markersize=2.5,
        color=INK,
        label="head",
    )
    ax.set_xlabel("tape cell")
    ax.set_ylabel("step" if every == 1 else f"step (every {every})")
    ax.set_xticks([0, -lo, hi - lo], [str(lo), "0", str(hi)])
    handles = [Rectangle((0, 0), 1, 1, color=NEUTRAL, label=f"blank {tm.blank!r}")]
    handles += [Rectangle((0, 0), 1, 1, color=SERIES[i], label=repr(s)) for i, s in enumerate(symbols)]
    handles.append(plt.Line2D([], [], linestyle="none", marker="o", markersize=4, color=INK, label="head"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=1.0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Inside one application of the network
# ---------------------------------------------------------------------------


def layer_values(net: CompiledNetwork, vector: Sequence[object]) -> np.ndarray:
    """Rows: input, every hidden layer, read-out; columns l, r, q, w
    (exact values, converted to float for plotting)."""
    vector = [Fraction(v) for v in vector]
    acts = net.activations_exact(vector)
    out = net.step_exact(vector)
    rows = [[*vector, 0], *acts, [*out, 0]]
    return np.array([[float(v) for v in row] for row in rows])


def _shade_passes(ax, net: CompiledNetwork, label: bool) -> None:
    """Shade every other pass; hidden layer k sits at x = k."""
    names = [pass_of(layer.note) for layer in net.program.layers]
    order = list(PASS_LABELS)
    start = 0
    for i in range(1, len(names) + 1):
        if i == len(names) or names[i] != names[start]:
            if order.index(names[start]) % 2 == 0:
                ax.axvspan(start + 0.5, i + 0.5, color=NEUTRAL, zorder=0, linewidth=0)
            if label:
                ax.text(
                    (start + 1 + i) / 2,
                    1.02,
                    PASS_LABELS[names[start]],
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=INK_2,
                )
            start = i


def step_inside(net: CompiledNetwork, vector: Sequence[object], axes=None, title: str = ""):
    """Registers and work neuron through all layers of one step, with the
    five passes shaded.  Three stacked panels on one shared x-axis."""
    values = layer_values(net, vector)
    x = np.arange(len(values))
    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1, 1]})
    ax_tape, ax_q, ax_w = axes
    for ax in axes:
        _shade_passes(ax, net, label=ax is ax_tape)
        ax.grid(axis="y")
    for col, name, color in (
        (0, "l (left tape)", SERIES[0]),
        (1, "r (head + right tape)", SERIES[1]),
    ):
        ax_tape.step(x, values[:, col], where="mid", color=color, label=name)
        ax_tape.text(x[-1] + 0.8, values[-1, col], name.split()[0], color=INK_2, va="center", fontsize=8)
    ax_tape.legend(loc="upper left", ncol=2)
    ax_tape.set_ylabel("tape registers")
    ax_q.step(x, values[:, 2], where="mid", color=SERIES[0])
    ax_q.set_ylabel("q (state / case code)")
    ax_w.step(x, values[:, 3], where="mid", color=SERIES[0])
    ax_w.set_ylabel("w (work neuron)")
    ax_w.set_xlabel("layer (0 = input, last = read-out)")
    if title:
        ax_tape.set_title(title, pad=18)
    return axes


def transition_matrix(net: CompiledNetwork, vectors: Sequence[Sequence[object]]) -> tuple[np.ndarray, list[str]]:
    """For every step, the value of each case's hat function (1 = this row
    of the transition table fires).  Returns (steps x cases, case labels)."""
    hats = [
        (i, layer.note.split(":")[0].replace("case ", ""))
        for i, layer in enumerate(net.program.layers)
        if layer.note.startswith("case ") and "hat ->" in layer.note
    ]
    labels = []
    for i, _ in hats:
        prev = net.program.layers[i - 1].note  # "case c = (state, symbol): hat, part 1"
        labels.append(prev[prev.index("(") + 1 : prev.rindex("):")])
    matrix = np.array([[float(net.activations_exact(v)[i][3]) for i, _ in hats] for v in vectors])
    return matrix, labels


def transition_heatmap(net: CompiledNetwork, vectors: Sequence[Sequence[object]], ax=None, title: str = ""):
    """Steps x table rows: exactly one hat lights up per step."""
    ax = ax or plt.gca()
    matrix, labels = transition_matrix(net, vectors)
    used = matrix.max(axis=0) > 0
    cmap = LinearSegmentedColormap.from_list("hat", [SURFACE, BLUE_RAMP[4]])
    ax.imshow(matrix[:, used], cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(int(used.sum())), [lab for lab, u in zip(labels, used, strict=True) if u], rotation=90)
    ax.set_xlabel("transition δ(state, symbol read)")
    ax.set_ylabel("step")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Numerics (from experiments/results/*.csv)
# ---------------------------------------------------------------------------


def _read_csv(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def capacity_plot(summary_csv: str | Path, axes=None):
    """Longest correct register per base, float32 and float64 side by side."""
    rows = _read_csv(summary_csv)
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    bases = sorted({int(r["base"]) for r in rows})
    series = (
        ("filler", "edge", "filler (never reads back)"),
        ("zigzag", "edge", "zigzag, ramps of the paper"),
        ("zigzag", "centre", "zigzag, centred ramps"),
    )
    for ax, dtype in zip(axes, ("float32", "float64"), strict=True):
        ax.grid(axis="y")
        for k, (machine, ramps, label) in enumerate(series):
            ys = []
            for b in bases:
                (r,) = [
                    r
                    for r in rows
                    if r["machine"] == machine and r["ramps"] == ramps and r["dtype"] == dtype and int(r["base"]) == b
                ]
                ys.append(int(r["max_correct_digits"]))
            offset = (k - 1) * 0.12
            ax.plot(
                np.arange(len(bases)) + offset,
                ys,
                linestyle="none",
                marker=MARKERS[k],
                markersize=7,
                color=SERIES[k],
                markeredgecolor=SURFACE,
                markeredgewidth=1,
                label=label,
            )
        pred = [
            next(int(r["predicted_push"]) for r in rows if r["dtype"] == dtype and int(r["base"]) == b)
            if any(r["predicted_push"] for r in rows if r["dtype"] == dtype and int(r["base"]) == b)
            else np.nan
            for b in bases
        ]
        ax.plot(np.arange(len(bases)), pred, linestyle="none", marker="_", markersize=16, color=INK, label="prediction")
        ax.set_xticks(range(len(bases)), [str(b) for b in bases])
        ax.set_xlabel("base β (powers of two: 4, 8, 16)")
        ax.set_title(dtype)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("longest correct register (cells)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return axes


def error_plot(trace_csv: str | Path, bases=(6, 10), dtype: str = "float64", axes=None):
    """Largest register error per step for the zigzag, ramps of the paper vs
    centred; a cross marks the first failure."""
    rows = [r for r in _read_csv(trace_csv) if r["machine"] == "zigzag" and r["dtype"] == dtype]
    if axes is None:
        _, axes = plt.subplots(1, len(bases), figsize=(9, 3.2), sharey=True)
    for ax, base in zip(np.atleast_1d(axes), bases, strict=True):
        ax.grid(axis="y")
        for ramps, label in (("edge", "ramps of the paper"), ("centre", "centred ramps")):
            rs = [r for r in rows if int(r["base"]) == base and r["ramps"] == ramps]
            steps = np.array([int(r["step"]) for r in rs])
            err = np.array([max(float(r["err_l"]), float(r["err_r"]), float(r["err_q"])) for r in rs])
            err[err == 0] = np.nan
            color = SERIES[1] if ramps == "edge" else SERIES[0]
            ax.plot(steps, err, color=color, label=label)
            fails = [int(r["step"]) for r in rs if r["status"] in ("wrong", "invalid")]
            if fails:
                f = fails[0]
                ax.plot([f], [err[steps == f][0]], marker="X", markersize=9, color=color, markeredgecolor=SURFACE)
                ax.annotate(
                    f"fails at step {f}",
                    (f, err[steps == f][0]),
                    textcoords="offset points",
                    xytext=(6, -12 if ramps == "edge" else 6),
                    fontsize=8,
                    color=INK_2,
                )
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_title(f"zigzag, base {base}, {dtype}")
    np.atleast_1d(axes)[0].set_ylabel("largest register error")
    np.atleast_1d(axes)[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return axes


def core_figure(tm: TuringMachine, net: CompiledNetwork, configs: Sequence[Configuration], step: int):
    """The README figure: the tape decoded from the network, and one step
    inside the network (state register and work neuron)."""
    use_style()
    fig = plt.figure(figsize=(11, 4.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.6], height_ratios=[1, 1], wspace=0.45, hspace=0.25)
    ax_tape = fig.add_subplot(grid[:, 0])
    tape_diagram(tm, configs, ax_tape, title=f"{tm.name}: tape decoded from the network")
    values = layer_values(net, encode(configs[step], net.layout))
    x = np.arange(len(values))
    ax_q, ax_w = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
    for ax in (ax_q, ax_w):
        _shade_passes(ax, net, label=ax is ax_q)
        ax.grid(axis="y")
    ax_q.step(x, values[:, 2], where="mid", color=SERIES[0])
    ax_q.set_ylabel("state register q")
    reads = [i for i, layer in enumerate(net.program.layers, 1) if pass_of(layer.note) == "read"]
    jump = int(np.argmax(np.diff(values[:, 2]))) + 1  # where the hat of the taken row fires
    notes = (
        (reads[-1], values[reads[-1], 2], "state·K + symbol read"),
        (jump, values[jump, 2], "row found: code of (new state, move)"),
        (len(values) - 1, values[-1, 2], "new state"),
    )
    for xi, yi, text in notes:
        ax_q.annotate(
            text,
            (xi, yi),
            textcoords="offset points",
            xytext=(-4, 6) if xi == len(values) - 1 else (6, -14),
            fontsize=8,
            color=INK_2,
            ha="right" if xi == len(values) - 1 else "left",
        )
    ax_q.set_title(f"inside one step (step {step}): {net.depth} layers of width 4", pad=18)
    ax_w.step(x, values[:, 3], where="mid", color=SERIES[0])
    ax_w.set_ylabel("work neuron w")
    ax_w.set_xlabel("layer")
    return fig
