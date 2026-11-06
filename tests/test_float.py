"""Floating point: exact while the tape fits into the mantissa, then it breaks.

The precise limits are the subject of the numerics study (step 5); here we
only pin down the qualitative behaviour.
"""

from hypothesis import given

from relu_tm import run
from relu_tm.examples import filler
from strategies import machine_and_word


def test_filler_float64_breaks_between_20_and_40_digits():
    # each step adds one base-4 digit (2 bits) to the left register
    result = run(filler(), steps=40, backend="float64")
    assert result.first_mismatch is not None
    assert 20 < result.first_mismatch <= 40
    assert all(e == n for e, n in zip(result.emulator[:20], result.network[:20], strict=True))


def test_filler_float32_breaks_earlier():
    f32 = run(filler(), steps=40, backend="float32").first_mismatch
    f64 = run(filler(), steps=40, backend="float64").first_mismatch
    assert 6 < f32 < f64


@given(machine_and_word(max_states=3, max_symbols=3, max_size=3))
def test_float64_is_exact_on_short_tapes(case):
    """At most 3 + 8 = 11 cells, i.e. <= 33 bits per register in base 8."""
    tm, word = case
    result = run(tm, word, steps=8, backend="float64")
    assert result.agree, "\n" + result.table()
