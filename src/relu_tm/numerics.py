"""Exact versus floating-point simulation: where and how does it break?

Two tools:

* :func:`read` - a *tolerant* decoder.  It reads a register digit by digit
  the way the network does (``y = β·x``, leading digit, remainder), but
  assigns each value to the nearest admissible case, splitting every gap in
  the middle (tail if ``y < 1/2``, digit ``d`` if ``y ∈ [d-1/2, d+3/2)``).
  Exact values decode exactly as with :func:`relu_tm.encoding.decode`.  A
  perturbed value still decodes correctly as long as the error, amplified
  by ``β`` per digit, stays below 1/2 - which is also what the network sees
  when it later reads those digits back.

* :func:`precision_profile` - runs the exact and the floating-point network
  side by side and records for every step the error of each register, the
  register lengths in digits and a status:

  ``exact``    float registers equal the exact ones;
  ``inexact``  they differ, but the configuration is still read correctly;
  ``wrong``    they read as a valid configuration that differs from the
               emulator's (a *silent* error);
  ``invalid``  they cannot be read at all (a *detectable* error).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction

from .encoding import Layout, decode_word, encode
from .machine import Configuration, Symbol, TuringMachine
from .network import CompiledNetwork, build_network

HALF = Fraction(1, 2)
STATUSES = ("exact", "inexact", "wrong", "invalid")


# ---------------------------------------------------------------------------
# Tolerant decoding
# ---------------------------------------------------------------------------


def read_word(value: object, layout: Layout, max_digits: int = 400) -> tuple[Symbol, ...] | None:
    """Tolerant inverse of :func:`relu_tm.encoding.encode_word`; ``None`` if
    the value is not within 1/2 (after scaling) of an admissible case."""
    x = Fraction(value)
    beta = layout.base
    top = 2 * len(layout.symbols) - 1  # largest digit
    word: list[Symbol] = []
    while True:
        y = x * beta
        if y < HALF:  # blank tail (also absorbs small negative errors)
            return tuple(word)
        if len(word) >= max_digits:
            return None
        d = 2 * math.floor((y - HALF) / 2) + 1
        if d > top:
            if y >= top + 1 + HALF:
                return None
            d = top
        word.append(layout.symbol(d))
        x = y - d


def read(vector: Sequence[object], layout: Layout) -> Configuration | None:
    """Tolerant version of :func:`relu_tm.encoding.decode`."""
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    l_val, r_val, q_val = (Fraction(v) for v in vector)
    q = math.floor(q_val + HALF)
    if not 0 <= q < len(layout.states):
        return None
    left, right = read_word(l_val, layout), read_word(r_val, layout)
    if left is None or right is None:
        return None
    return Configuration(layout.states[q], left, right, layout.blank)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepRecord:
    step: int
    digits_l: int  # length of the left tape word (exact, without trailing blanks)
    digits_r: int
    err_l: float  # |float - exact| per register
    err_r: float
    err_q: float
    status: str

    @property
    def digits(self) -> int:
        return max(self.digits_l, self.digits_r)

    @property
    def error(self) -> float:
        return max(self.err_l, self.err_r, self.err_q)


@dataclass
class Profile:
    machine: str
    base: int
    dtype: str
    records: list[StepRecord]

    def first(self, *statuses: str) -> StepRecord | None:
        """First step whose status is one of ``statuses``."""
        for rec in self.records:
            if rec.status in statuses:
                return rec
        return None

    @property
    def first_inexact(self) -> StepRecord | None:
        return self.first("inexact", "wrong", "invalid")

    @property
    def first_failure(self) -> StepRecord | None:
        return self.first("wrong", "invalid")

    @property
    def max_correct_digits(self) -> int:
        """Longest register (in digits) reached before the first failure."""
        stop = self.first_failure
        good = [r.digits for r in self.records if stop is None or r.step < stop.step]
        return max(good, default=0)

    def rows(self) -> list[dict]:
        return [{"machine": self.machine, "base": self.base, "dtype": self.dtype, **asdict(r)} for r in self.records]


def precision_profile(
    tm: TuringMachine,
    word: Sequence[Symbol] | str = (),
    steps: int = 50,
    dtype: str = "float64",
    base: int | None = None,
    network: CompiledNetwork | None = None,
    ramps: str = "edge",
) -> Profile:
    """Run the exact and the floating-point network (each fed with its own
    output) and compare them after every step."""
    import torch

    net = network or build_network(tm, base, ramps)
    torch_dtype = {"float64": torch.float64, "float32": torch.float32}[dtype]
    emulator = tm.trace(word, steps)
    x0 = encode(emulator[0], net.layout)
    exact = net.iterate_dyadic(x0, steps) if net.is_dyadic() else net.iterate_exact(x0, steps)
    floats = net.iterate_torch(x0, steps, torch_dtype)

    records = []
    for t, (e, f, config) in enumerate(zip(exact, floats, emulator, strict=True)):
        errs = [abs(Fraction(fv) - ev) for fv, ev in zip(f, e, strict=True)]
        if not any(errs):
            status = "exact"
        else:
            seen = read(f, net.layout)
            status = "invalid" if seen is None else ("inexact" if seen == config else "wrong")
        # Trailing blank digits do not count: rounding them away is harmless,
        # because a blank tail means the same to the machine.
        lengths = Configuration(config.state, decode_word(e[0], net.layout), decode_word(e[1], net.layout), tm.blank)
        records.append(
            StepRecord(
                step=t,
                digits_l=len(lengths.left),
                digits_r=len(lengths.right),
                err_l=float(errs[0]),
                err_r=float(errs[1]),
                err_q=float(errs[2]),
                status=status,
            )
        )
    return Profile(tm.name, net.layout.base, dtype, records)


# ---------------------------------------------------------------------------
# Prediction for power-of-two bases
# ---------------------------------------------------------------------------

MANTISSA_BITS = {"float32": 24, "float64": 53}


def predicted_capacity(base: int, dtype: str, digit: int | None = None) -> int | None:
    """Longest register (in digits) that floats keep exact, for ``β = 2^m``.

    The binding operation is the gated scaling of a push: it forms
    ``κ·(x + d)`` with ``κ = 1 - 1/β``.  If ``x`` has ``n`` digits, that
    product has its lowest bit at ``2^-m(n+1)`` and its highest at ``2^h``,
    ``h = ⌊log2 κ(d+1)⌋``, so it fits into ``p`` significand bits iff
    ``h + 1 + m(n+1) ≤ p``; the register then has ``n + 1`` digits:

    * ``digit=d`` (the digit being pushed, e.g. 3 for the filler):
      ``⌊(p - 1 - h)/m⌋`` - matches the measurement exactly;
    * ``digit=None``: worst case ``d = β - 1``, ``h = m - 1``, giving
      ``⌊p/m⌋ - 1`` - a lower bound that held in every measured run.

    ``None`` for other bases: there the weights themselves are rounded.
    """
    m = base.bit_length() - 1
    if base != 1 << m:
        return None
    p = MANTISSA_BITS[dtype]
    if digit is None:
        return p // m - 1
    top = Fraction(base - 1, base) * (digit + 1)  # sup of κ(x + d), not attained
    h = math.ceil(math.log2(top)) - 1  # highest bit of values in [κd, κ(d+1))
    return (p - 1 - h) // m


# ---------------------------------------------------------------------------
# Inside one step: which pass rounds first, which pass amplifies?
# ---------------------------------------------------------------------------

PASSES = ("read", "table", "scale", "left", "state", "output")
_PREFIX = {"read": "read", "case": "table", "scale": "scale", "left": "left", "state": "state"}


def pass_of(note: str) -> str:
    """Name of the pass (see :mod:`relu_tm.network`) a layer belongs to."""
    return _PREFIX.get(note.split(maxsplit=1)[0].rstrip(":"), "?") if note else "?"


@dataclass(frozen=True)
class LayerError:
    layer: int  # 1-based hidden layer; 0 = input, depth + 1 = read-out
    pass_name: str
    errors: tuple[float, ...]  # |float - exact| per neuron (l, r, q, w) or (l, r, q)

    @property
    def max(self) -> float:
        return max(self.errors)

    @property
    def neuron(self) -> str | None:
        """First neuron with a non-zero error."""
        for name, err in zip("lrqw", self.errors, strict=False):
            if err:
                return name
        return None


def layer_errors(
    net: CompiledNetwork, exact_vector: Sequence[object], float_vector: Sequence[float], dtype: str = "float64"
) -> list[LayerError]:
    """Run one application of ``N`` exactly (from ``exact_vector``) and in
    floating point (from ``float_vector``) and compare every layer."""
    import torch

    torch_dtype = {"float64": torch.float64, "float32": torch.float32}[dtype]
    module = net.to_torch(torch_dtype)
    x = torch.tensor([float(v) for v in float_vector], dtype=torch_dtype)
    float_acts = [*module.activations(x).tolist(), module(x).tolist()]
    exact_acts = net.activations_exact(exact_vector)
    exact_acts.append(list(net.step_exact(exact_vector)))
    notes = [layer.note for layer in net.program.layers]
    out = []
    for i, (fa, ea) in enumerate(zip(float_acts, exact_acts, strict=True)):
        name = pass_of(notes[i]) if i < len(notes) else "output"
        errs = tuple(float(abs(Fraction(f) - e)) for f, e in zip(fa, ea, strict=True))
        out.append(LayerError(i + 1, name, errs))
    return out


def first_rounding(net: CompiledNetwork, vector: Sequence[object], dtype: str = "float64") -> LayerError | None:
    """First layer in which the float network leaves the exact values when
    both start from ``vector``.  Layer 0 (``"input"``) means that ``vector``
    itself is not representable in ``dtype`` - the tape is already too long."""
    import torch

    torch_dtype = {"float64": torch.float64, "float32": torch.float32}[dtype]
    rounded = torch.tensor([float(Fraction(v)) for v in vector], dtype=torch_dtype).tolist()
    in_err = tuple(float(abs(Fraction(f) - Fraction(v))) for f, v in zip(rounded, vector, strict=True))
    if any(in_err):
        return LayerError(0, "input", in_err)
    for err in layer_errors(net, vector, rounded, dtype):
        if err.max:
            return err
    return None


def pass_errors(
    net: CompiledNetwork, exact_vector: Sequence[object], float_vector: Sequence[float], dtype: str = "float64"
) -> dict[str, float]:
    """Largest error at the end of each pass (plus ``"input"``)."""
    errs = {
        "input": max(float(abs(Fraction(f) - Fraction(e))) for f, e in zip(float_vector, exact_vector, strict=True))
    }
    for err in layer_errors(net, exact_vector, float_vector, dtype):
        errs[err.pass_name] = err.max  # the last layer of a pass wins
    return errs
