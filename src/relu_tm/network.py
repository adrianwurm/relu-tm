"""Construction of the width-4 ReLU network ``N`` for a Turing machine.

``N`` maps the encoding ``(l, r, q)`` of a configuration to the encoding of
its successor, so ``N`` applied ``t`` times simulates ``t`` steps.  All
weights are computed from the transition table - there is no training.

One application of ``N`` runs five passes (``K = |Γ| + 1`` head codes per
state, ``n = |Q|`` states, ``C = n·K`` *case codes*):

0. **read**   pop the head symbol from ``r`` into ``q``:
              ``q := K·q + σ(head)``, ``r := β·r - lead(r)``.
1. **table**  for every case code ``c`` (one hat function each):
              ``q := D_m + q'`` (a "done" code that also records the move
              ``m``) and add the written digit to ``l`` (move R) or ``r``.
2. **scale**  ``l := l/β`` for moves R, ``r := r/β`` for moves L and S
              (gated by the range of ``q``).
3. **left**   moves L only: move the leading digit of ``l`` over to ``r``
              (a blank tail becomes an explicit blank digit).
4. **state**  ``q := q - D_m``, i.e. back to the plain state index.

Hidden layers: ``5·|Γ| + 2·C + 12``, all of width exactly 4.  The details
and every inequality the gates rely on are written up in
``docs/construction.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING

from .encoding import Layout, Vector
from .machine import TuringMachine
from .program import Aff, Program, ProgramBuilder

if TYPE_CHECKING:  # pragma: no cover
    import torch

WIDTH = 4


RAMPS = ("edge", "centre")


def _gap_test(y: Aff, ramps: str) -> Aff:
    """Pre-activation of a digit test whose ramp lies in the gap ``(0, 1)``
    of ``y``.  ``"edge"`` (as in the papers): the ramp fills the whole gap,
    ``min(1, ReLU(y))``.  ``"centre"``: the ramp covers only ``[1/4, 3/4]``,
    so values within 1/4 of an admissible case still give exactly 0 or 1 -
    this tolerates rounding errors (see ``docs/numerics.md``)."""
    return y if ramps == "edge" else 2 * y - Fraction(1, 2)


def build_program(tm: TuringMachine, layout: Layout, ramps: str = "edge") -> Program:
    """The register program that performs one step of ``tm``."""
    if ramps not in RAMPS:
        raise ValueError(f"ramps must be one of {RAMPS}")
    beta = Fraction(layout.base)
    k = len(layout.symbols)  # tape symbols incl. blank
    K = k + 1  # head codes per state: blank tail + k digits
    n = len(layout.states)
    C = n * K  # number of case codes
    D = {"L": C, "S": C + n, "R": C + 2 * n}  # "done" code ranges
    kappa = 1 - 1 / beta

    l, r, q, w = (Aff.var(v) for v in ("l", "r", "q", "w"))
    b = ProgramBuilder()

    # -- pass 0: read the head symbol --------------------------------------
    # Digits are 0 (tail), 1, 3, 5, ...  Gap g separates digit e_{g-1} from
    # e_g; after subtracting the digits below, beta*r lies in [0,1] or >= 1 + theta.
    b.affine(q=K * q)
    for g in range(1, k + 1):
        theta, delta = (0, 1) if g == 1 else (1, 2)
        b.relu(_gap_test(beta * r - theta, ramps), note=f"read: gap {g}, w = ReLU(beta r - {theta})")
        b.relu(1 - w, note=f"read: gap {g}, w = 1 - [digit >= {2 * g - 1}]")
        b.add(q=1 - w, r=-(delta / beta) * (1 - w))
    b.affine(r=beta * r)

    # -- pass 1: transition table -------------------------------------------
    for c in range(C):
        qi, s = divmod(c, K)
        state = layout.states[qi]
        symbol = layout.blank if s == 0 else layout.symbols[s - 1]
        action = tm.action(state, symbol)
        new_state, written, move = action if action is not None else (state, symbol, "S")
        done = D[move] + layout.state_index(new_state)
        digit = layout.digit(written)
        shown = f"{symbol} (tail)" if s == 0 else symbol
        b.relu(c - q, note=f"case {c} = ({state}, {shown}): hat, part 1")
        b.relu(1 - q + c - 2 * w, note=f"case {c}: hat -> {new_state},{written},{move}")
        increments = {"q": (done - c) * w}
        if move == "R":
            increments["l"] = digit * w
        elif move == "L":
            increments["r"] = (digit + beta) * w  # +beta becomes +1 after pass 2
        else:
            increments["r"] = digit * w
        b.add(**increments)

    # -- pass 2: gated scaling ----------------------------------------------
    b.relu(D["R"] - q, note="scale: dist(q, R-range)")
    b.relu(kappa * l - kappa * beta * w, note="scale: (1-1/beta) l if move R")
    b.add(l=-w)
    b.relu(q - (D["R"] - 1), note="scale: dist(q, L/S-range)")
    b.relu(kappa * r - kappa * 2 * beta * w, note="scale: (1-1/beta) r if move L or S")
    b.add(r=-w)

    # -- pass 3: moves L transfer the leading digit of l to r ------------------
    gate_l = q - (D["S"] - 1)  # = 0 on the L-range, >= 1 elsewhere
    for g in range(1, k + 1):
        theta, delta = (0, 1) if g == 1 else (1, 2)
        rho = 0 if g == 1 else 2  # a blank tail (0) is pushed as the blank digit 1
        b.relu(gate_l, note="left: dist(q, L-range)")
        b.relu(_gap_test(beta * l - theta - beta * w, ramps), note=f"left: gap {g} test (gated)")
        b.relu(1 - w, note=f"left: gap {g}, w = 1 - inc")
        b.add(l=-(delta / beta) * (1 - w), r=rho * (1 - w))
    b.relu(gate_l, note="left: dist(q, L-range)")
    b.relu((beta - 1) * l - (beta - 1) * w, note="left: (beta-1) l if move L")
    b.add(l=w)
    b.relu(gate_l, note="left: dist(q, L-range)")
    b.relu(kappa * r - kappa * beta * w, note="left: (1-1/beta) r if move L")
    b.add(r=-w)

    # -- pass 4: done code -> state index -----------------------------------
    for start in (D["R"], D["S"]):
        b.relu(q - start + 1, note=f"state: q >= {start}?")
        b.relu(1 - w, note=f"state: w = 1 - [q >= {start}]")
        b.add(q=-n * (1 - w))
    b.affine(q=q - D["L"])

    return b.finish()


def _matvec(W: Sequence[Sequence[Fraction]], x: Sequence[Fraction], bias: Sequence[Fraction]) -> list[Fraction]:
    out = []
    for row, b in zip(W, bias, strict=True):
        s = b
        for coef, xi in zip(row, x, strict=True):
            if coef:
                s += coef * xi
        out.append(s)
    return out


@dataclass
class CompiledNetwork:
    """The network ``N`` for one Turing machine, with exact rational weights."""

    machine: TuringMachine
    layout: Layout
    program: Program
    hidden: list[tuple[list[list[Fraction]], list[Fraction]]] = field(repr=False)
    output: tuple[list[list[Fraction]], list[Fraction]] = field(repr=False)

    @property
    def width(self) -> int:
        return max(len(W) for W, _ in self.hidden)

    @property
    def depth(self) -> int:
        """Number of hidden layers."""
        return len(self.hidden)

    @property
    def n_parameters(self) -> int:
        mats = [*self.hidden, self.output]
        return sum(len(W) * len(W[0]) + len(b) for W, b in mats)

    # -- exact evaluation (the paper's model: rational arithmetic) ------------

    def activations_exact(self, vector: Sequence[object]) -> list[list[Fraction]]:
        """Outputs of every hidden layer (after ReLU) for one application."""
        x = [Fraction(v) for v in vector]
        acts = []
        for W, b in self.hidden:
            x = [max(v, Fraction(0)) for v in _matvec(W, x, b)]
            acts.append(x)
        return acts

    def step_exact(self, vector: Sequence[object]) -> Vector:
        """One application of ``N`` in exact rational arithmetic."""
        acts = self.activations_exact(vector)
        W, b = self.output
        out = _matvec(W, acts[-1], b)
        return out[0], out[1], out[2]

    def iterate_exact(self, vector: Sequence[object], steps: int) -> list[Vector]:
        """``[x, N(x), N(N(x)), ...]`` with ``steps + 1`` entries."""
        xs = [tuple(Fraction(v) for v in vector)]
        for _ in range(steps):
            xs.append(self.step_exact(xs[-1]))
        return xs

    def is_dyadic(self) -> bool:
        """All weights have power-of-two denominators (true for bases 2^m)."""
        mats = [*self.hidden, self.output]
        dens = {Fraction(v).denominator for W, b in mats for v in [*(x for row in W for x in row), *b]}
        return all(d & (d - 1) == 0 for d in dens)

    def dyadic(self):
        """The same network with integer weights (:mod:`relu_tm.dyadic`)."""
        from .dyadic import DyadicNetwork

        if getattr(self, "_dyadic", None) is None:
            self._dyadic = DyadicNetwork(self)
        return self._dyadic

    def iterate_dyadic(self, vector: Sequence[object], steps: int) -> list[Vector]:
        """Exact like :meth:`iterate_exact`, but with dyadic integers - faster;
        only for power-of-two bases."""
        return self.dyadic().iterate(vector, steps)

    # -- PyTorch ---------------------------------------------------------------

    def to_torch(self, dtype: torch.dtype | None = None) -> torch.nn.Module:
        """An ``nn.Module`` with the same weights (rounded to ``dtype``)."""
        import torch

        from .torch_module import StepNetwork

        return StepNetwork.from_compiled(self, dtype or torch.float64)

    def iterate_torch(
        self, vector: Sequence[object], steps: int, dtype: torch.dtype | None = None
    ) -> list[tuple[float, float, float]]:
        """Like :meth:`iterate_exact`, but in floating point with PyTorch."""
        import torch

        dtype = dtype or torch.float64
        module = self.to_torch(dtype)
        x = torch.tensor([float(Fraction(v)) for v in vector], dtype=dtype)
        xs = [tuple(x.tolist())]
        with torch.no_grad():
            for _ in range(steps):
                x = module(x)
                xs.append(tuple(x.tolist()))
        return xs


def build_network(tm: TuringMachine, base: int | None = None, ramps: str = "edge") -> CompiledNetwork:
    """Compute the width-4 ReLU network that simulates one step of ``tm``."""
    layout = Layout.for_machine(tm, base)
    program = build_program(tm, layout, ramps)
    hidden, output = program.matrices()
    net = CompiledNetwork(tm, layout, program, hidden, output)
    assert net.width == WIDTH
    return net
