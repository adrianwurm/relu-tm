"""Encoding Turing-machine configurations as three numbers ``(l, r, q)``.

This follows the {1,3}-quadradic encoding of Bournez, Cohen & Wurm
(MFCS 2025, Def. of DD_{1,3} and Lemma 19):

* ``r`` encodes the head cell and everything to its right,
* ``l`` encodes the cells to the left of the head, nearest cell first,
* ``q`` is the index of the current state,

where a word ``c_1 c_2 ... c_n`` becomes the base-``β`` fraction
``Σ digit(c_i) · β^(-i)``.  Tape symbol number ``i`` (the blank is number 0)
gets the **odd** digit ``2i + 1``; the infinite all-blank tail is the digit
``0``.  Every encoded value therefore lies in a set with gaps: the leading
digit can be read off by a *continuous* piecewise-linear function, which is
what a ReLU network can compute.

For a machine with one non-blank symbol (e.g. busy beavers: blank -> 1,
symbol -> 3) the registers are {1,3}-quadradics in base 4.  Larger
alphabets use the smallest power of two ``β ≥ 2·|Γ|`` (so that floats stay
exact as long as the tape is short); any even base ``β ≥ 2·|Γ|`` is
allowed.  The paper's machines over {1, 3, B} therefore use base 8 here.

Deviation from the paper (documented in ``docs/construction.md``): the
blank gets its own non-zero digit.  The digit ``0`` only ever appears in
the infinite tail, so the head can step onto the blank region and the
machine can write blanks anywhere without breaking the gap structure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real

from .machine import Configuration, State, Symbol, TuringMachine

Vector = tuple[Fraction, Fraction, Fraction]


class DecodeError(ValueError):
    """Raised when register values are not a valid encoding."""


def default_base(n_symbols: int) -> int:
    """Smallest power of two that is at least ``2 · n_symbols`` (and ≥ 4)."""
    base = 4
    while base < 2 * n_symbols:
        base *= 2
    return base


@dataclass(frozen=True)
class Layout:
    """Everything needed to translate between configurations and numbers."""

    states: tuple[State, ...]
    symbols: tuple[Symbol, ...]  # symbols[0] is the blank
    base: int

    def __post_init__(self) -> None:
        if self.base % 2 or self.base < 2 * len(self.symbols):
            raise ValueError(f"base must be even and at least 2·|Γ| = {2 * len(self.symbols)}")
        if len(set(self.states)) != len(self.states) or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("states and symbols must be unique")

    @classmethod
    def for_machine(cls, tm: TuringMachine, base: int | None = None) -> Layout:
        return cls(tm.states, tm.symbols, base or default_base(len(tm.symbols)))

    @property
    def blank(self) -> Symbol:
        return self.symbols[0]

    def digit(self, symbol: Symbol) -> int:
        return 2 * self.symbols.index(symbol) + 1

    def symbol(self, digit: int) -> Symbol:
        if digit % 2 == 0 or not 0 < digit < 2 * len(self.symbols):
            raise DecodeError(f"{digit} is not a valid digit")
        return self.symbols[(digit - 1) // 2]

    def state_index(self, state: State) -> int:
        return self.states.index(state)


def encode_word(word: Sequence[Symbol], layout: Layout) -> Fraction:
    """``c_1 ... c_n  ->  Σ digit(c_i) β^(-i)`` (exact)."""
    value = Fraction(0)
    for symbol in reversed(word):  # Horner scheme
        value = (value + layout.digit(symbol)) / layout.base
    return value


def encode(config: Configuration, layout: Layout) -> Vector:
    """The register vector ``(l, r, q)`` of a configuration."""
    if config.blank != layout.blank:
        raise ValueError("configuration and layout use different blanks")
    return (
        encode_word(config.left, layout),
        encode_word(config.right, layout),
        Fraction(layout.state_index(config.state)),
    )


def _to_fraction(x: object) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if hasattr(x, "item"):  # numpy / torch scalars
        x = x.item()
    if isinstance(x, (int, float, Real)):
        return Fraction(x)  # exact binary value of a float
    raise TypeError(f"cannot interpret {x!r} as a number")


def decode_word(value: object, layout: Layout, max_digits: int = 100_000) -> tuple[Symbol, ...]:
    """Inverse of :func:`encode_word`; raises :class:`DecodeError` if the value
    is not in the encoding set (e.g. after a floating-point error)."""
    x = _to_fraction(value)
    if not 0 <= x < 1:
        raise DecodeError(f"register value {float(x)} outside [0, 1)")
    word: list[Symbol] = []
    while x:
        if len(word) >= max_digits:
            raise DecodeError("expansion does not terminate")
        x *= layout.base
        digit = int(x)  # floor, x >= 0
        x -= digit
        if digit == 0:
            raise DecodeError("digit 0 (blank tail) followed by non-zero digits")
        word.append(layout.symbol(digit))
    return tuple(word)


def decode(vector: Sequence[object], layout: Layout) -> Configuration:
    """Turn register values back into a configuration (exactly)."""
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    l_val, r_val, q_val = (_to_fraction(v) for v in vector)
    if q_val.denominator != 1 or not 0 <= q_val < len(layout.states):
        raise DecodeError(f"state register {float(q_val)} is not a state index")
    return Configuration(
        layout.states[int(q_val)],
        decode_word(l_val, layout),
        decode_word(r_val, layout),
        layout.blank,
    )


def is_valid(vector: Sequence[object], layout: Layout) -> bool:
    try:
        decode(vector, layout)
    except DecodeError:
        return False
    return True
