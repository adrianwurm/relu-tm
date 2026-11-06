"""A tiny "register program" language that compiles to a width-4 ReLU network.

The network keeps three **registers** ``l, r, q`` (one neuron each) and one
**work neuron** ``w``.  Every hidden layer computes, from the previous
layer's four outputs,

* new register values - arbitrary affine combinations of ``l, r, q, w``;
* a new work value    - ``ReLU`` of an affine combination.

Register values stay non-negative on valid inputs, so the ReLU of a
register neuron is the identity and the register just carries its value
forward.  The only non-linearity is in ``w``; chaining ``w`` through a few
layers evaluates nested expressions such as ``ReLU(1 - ReLU(4r - 1))``,
and adding a multiple of ``w`` to a register commits the result.  This is
the "keep the inputs alive, compute in one extra neuron" discipline of
Lemma 3.1 in the NeurIPS 2026 paper.

:class:`ProgramBuilder` offers three operations:

``affine(**updates)``  register := affine function of the registers (free:
                       folded into the next layer, costs no layer);
``relu(expr)``         w := ReLU(expr), expr over the *current* registers
                       and the current w (costs exactly one layer);
``add(**exprs)``       register += expr, expr may use w (folded into the
                       next layer).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction

REGISTERS = ("l", "r", "q")
NEURONS = (*REGISTERS, "w")
ONE = "1"

Number = int | Fraction


class Aff:
    """An exact affine expression ``Σ c_v · v + c_1`` over named variables."""

    __slots__ = ("terms",)

    def __init__(self, terms: Mapping[str, Number] | None = None) -> None:
        self.terms: dict[str, Fraction] = {}
        for name, coef in (terms or {}).items():
            coef = Fraction(coef)
            if coef:
                self.terms[name] = self.terms.get(name, Fraction(0)) + coef

    @staticmethod
    def var(name: str) -> Aff:
        return Aff({name: 1})

    @staticmethod
    def const(value: Number) -> Aff:
        return Aff({ONE: value})

    @staticmethod
    def lift(x: Aff | Number) -> Aff:
        return x if isinstance(x, Aff) else Aff.const(x)

    def coef(self, name: str) -> Fraction:
        return self.terms.get(name, Fraction(0))

    @property
    def constant(self) -> Fraction:
        return self.coef(ONE)

    def variables(self) -> set[str]:
        return set(self.terms) - {ONE}

    def __add__(self, other: Aff | Number) -> Aff:
        other = Aff.lift(other)
        out = dict(self.terms)
        for k, v in other.terms.items():
            out[k] = out.get(k, Fraction(0)) + v
        return Aff(out)

    __radd__ = __add__

    def __neg__(self) -> Aff:
        return Aff({k: -v for k, v in self.terms.items()})

    def __sub__(self, other: Aff | Number) -> Aff:
        return self + (-Aff.lift(other))

    def __rsub__(self, other: Aff | Number) -> Aff:
        return Aff.lift(other) - self

    def __mul__(self, scalar: Number) -> Aff:
        if isinstance(scalar, Aff):
            raise TypeError("Aff * Aff is not affine")
        s = Fraction(scalar)
        return Aff({k: v * s for k, v in self.terms.items()})

    __rmul__ = __mul__

    def __truediv__(self, scalar: Number) -> Aff:
        return self * (1 / Fraction(scalar))

    def substitute(self, mapping: Mapping[str, Aff]) -> Aff:
        """Replace variables by expressions (missing variables stay)."""
        out = Aff.const(self.constant)
        for name, coef in self.terms.items():
            if name != ONE:
                out = out + mapping.get(name, Aff.var(name)) * coef
        return out

    def evaluate(self, values: Mapping[str, Fraction]) -> Fraction:
        return self.constant + sum((c * values[v] for v, c in self.terms.items() if v != ONE), Fraction(0))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Aff) and self.terms == other.terms

    def __hash__(self) -> int:  # pragma: no cover
        return hash(frozenset(self.terms.items()))

    def __repr__(self) -> str:
        order = {n: i for i, n in enumerate((*NEURONS, ONE))}
        parts = []
        for name in sorted(self.terms, key=lambda n: order.get(n, 99)):
            c = self.terms[name]
            if name == ONE:
                parts.append(f"{c}")
            elif c == 1:
                parts.append(name)
            elif c == -1:
                parts.append(f"-{name}")
            else:
                parts.append(f"{c}·{name}")
        return " + ".join(parts).replace("+ -", "- ") if parts else "0"


def _identity() -> dict[str, Aff]:
    return {x: Aff.var(x) for x in REGISTERS}


@dataclass
class LayerSpec:
    """One hidden layer: each neuron's pre-activation as an affine expression
    of the previous layer's outputs ``l, r, q, w``."""

    registers: dict[str, Aff]
    work: Aff
    note: str = ""


@dataclass
class Program:
    layers: list[LayerSpec]
    output: dict[str, Aff]  # final affine read-out (A_out)

    def matrices(self) -> tuple[list[tuple[list[list[Fraction]], list[Fraction]]], tuple]:
        """Exact weights: ``[(W_1, b_1), ..., (W_L, b_L)]`` and ``(W_out, b_out)``.

        ``W_1`` is 4x3 (inputs ``l, r, q``), the other ``W_t`` are 4x4 with
        rows/columns ordered ``l, r, q, w``; ``W_out`` is 3x4.
        """
        hidden = []
        for t, layer in enumerate(self.layers):
            inputs = REGISTERS if t == 0 else NEURONS
            exprs = [layer.registers[x] for x in REGISTERS] + [layer.work]
            if t == 0:
                exprs = [e.substitute({"w": Aff()}) for e in exprs]
            W = [[e.coef(v) for v in inputs] for e in exprs]
            b = [e.constant for e in exprs]
            hidden.append((W, b))
        W_out = [[self.output[x].coef(v) for v in NEURONS] for x in REGISTERS]
        b_out = [self.output[x].constant for x in REGISTERS]
        return hidden, (W_out, b_out)

    def listing(self) -> str:
        """A readable table of the program (one line per hidden layer)."""
        lines = [f"{'#':>4}  {'w := ReLU(...)':<34}  register updates / note"]
        for t, layer in enumerate(self.layers, 1):
            changed = {x: e for x, e in layer.registers.items() if e != Aff.var(x)}
            upd = ", ".join(f"{x}:={e!r}" for x, e in changed.items())
            lines.append(f"{t:>4}  {layer.work!r:<34}  {upd}{'  | ' if upd and layer.note else ''}{layer.note}")
        out = {x: e for x, e in self.output.items() if e != Aff.var(x)}
        lines.append(f"{'out':>4}  {'':<34}  " + ", ".join(f"{x}:={e!r}" for x, e in out.items()))
        return "\n".join(lines)


@dataclass
class ProgramBuilder:
    layers: list[LayerSpec] = field(default_factory=list)
    _pending: dict[str, Aff] = field(default_factory=_identity)

    def affine(self, **updates: Aff | Number) -> None:
        """Registers := affine functions of the current registers (no layer)."""
        for e in updates.values():
            if "w" in Aff.lift(e).variables():
                raise ValueError("use add() for updates that involve w")
        new = dict(self._pending)
        for x, e in updates.items():
            new[x] = Aff.lift(e).substitute(self._pending)
        self._pending = new

    def add(self, **increments: Aff | Number) -> None:
        """Registers += expressions in the current registers and w (no layer)."""
        new = dict(self._pending)
        for x, e in increments.items():
            new[x] = new[x] + Aff.lift(e).substitute(self._pending)
        self._pending = new

    def relu(self, expr: Aff | Number, note: str = "") -> None:
        """Emit one layer: commit pending register updates, w := ReLU(expr)."""
        work = Aff.lift(expr).substitute(self._pending)
        self.layers.append(LayerSpec(dict(self._pending), work, note))
        self._pending = _identity()

    def finish(self) -> Program:
        return Program(list(self.layers), dict(self._pending))
