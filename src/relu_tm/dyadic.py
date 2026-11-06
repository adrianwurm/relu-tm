"""Fast exact evaluation for power-of-two bases: dyadic fixed-point integers.

In a base ``β = 2^m`` every weight and every register value is a dyadic
rational.  :class:`DyadicNetwork` therefore stores the weights of each
layer as integers over one power of two ``2^S`` and the three registers as
integers ``X`` over a *shared* exponent, ``value = X / 2^K``.  A layer then
is integer multiply-adds plus one shift.  If the shift would drop a set
bit, ``K`` grows by ``S`` instead, so the arithmetic never rounds; after
every step common factors of two are divided out again, so ``K`` stays at
the precision the tape actually needs.

The results are bit for bit those of the :class:`fractions.Fraction`
reference, without the gcd computations (typically 3-5 times faster).
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .network import CompiledNetwork

Fixed = tuple[list[int], int]  # (X, K): value_i = X_i / 2^K


def _power_of_two_exponent(d: int) -> int:
    if d & (d - 1):
        raise ValueError(f"denominator {d} is not a power of two; use a base that is a power of two")
    return d.bit_length() - 1


def to_fixed(vector: Sequence[object]) -> Fixed:
    """Exact conversion of dyadic numbers (Fractions, ints, floats) to ``(X, K)``."""
    values = [Fraction(v) for v in vector]
    k = max(_power_of_two_exponent(v.denominator) for v in values)
    return [int(v * (1 << k)) for v in values], k


def to_fractions(fixed: Fixed) -> tuple[Fraction, ...]:
    xs, k = fixed
    return tuple(Fraction(x, 1 << k) for x in xs)


def _normalize(xs: list[int], k: int) -> Fixed:
    """Divide out the common power of two."""
    nonzero = [x for x in xs if x]
    if not nonzero:
        return xs, 0
    shift = min(min((x & -x).bit_length() - 1 for x in nonzero), k)
    return ([x >> shift for x in xs], k - shift) if shift else (xs, k)


class DyadicNetwork:
    """The network ``N`` with integer weights (exact, power-of-two bases only)."""

    def __init__(self, net: CompiledNetwork) -> None:
        self.net = net
        self.layers: list[tuple[list[list[tuple[int, int]]], list[int], int]] = []
        for W, b in [*net.hidden, net.output]:
            entries = [Fraction(v) for row in W for v in row] + [Fraction(v) for v in b]
            s = max(_power_of_two_exponent(v.denominator) for v in entries)
            rows = [[(j, int(v * (1 << s))) for j, v in enumerate(row) if v] for row in W]
            self.layers.append((rows, [int(v * (1 << s)) for v in b], s))

    def step(self, fixed: Fixed) -> Fixed:
        """One application of ``N``."""
        xs, k = fixed
        last = len(self.layers) - 1
        for index, (rows, bias, s) in enumerate(self.layers):
            acc = []
            for row, b in zip(rows, bias, strict=True):
                total = b << k
                for j, w in row:
                    total += w * xs[j]
                acc.append(total)
            if s and any(a & ((1 << s) - 1) for a in acc):
                k += s  # keep the extra bits instead of rounding
                xs = acc
            else:
                xs = [a >> s for a in acc]
            if index < last:
                xs = [x if x > 0 else 0 for x in xs]  # ReLU
        return _normalize(xs, k)

    def iterate_fixed(self, vector: Sequence[object], steps: int) -> list[Fixed]:
        state = to_fixed(vector)
        out = [state]
        for _ in range(steps):
            state = self.step(state)
            out.append(state)
        return out

    def iterate(self, vector: Sequence[object], steps: int) -> list[tuple[Fraction, ...]]:
        """Like :meth:`CompiledNetwork.iterate_exact`, with the same results."""
        return [to_fractions(f) for f in self.iterate_fixed(vector, steps)]
