"""Tolerant decoding, precision profiles and the capacity prediction."""

from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from relu_tm import Layout, build_network, run
from relu_tm.cli import main
from relu_tm.encoding import decode_word, encode_word
from relu_tm.examples import filler, zigzag
from relu_tm.numerics import precision_profile, predicted_capacity, read, read_word
from strategies import machine_and_word

SYMBOLS = ("_", "a", "b")
words = st.lists(st.sampled_from(SYMBOLS), max_size=12).map(tuple)


@pytest.mark.parametrize("base", [6, 8, 10])
@given(word=words)
def test_read_equals_exact_decoding_on_exact_values(base, word):
    layout = Layout(("p",), SYMBOLS, base)
    x = encode_word(word, layout)
    assert read_word(x, layout) == decode_word(x, layout)


@given(word=words, sign=st.sampled_from([-1, 1]))
def test_read_tolerates_small_errors(word, sign):
    layout = Layout(("p",), SYMBOLS, 8)
    x = encode_word(word, layout)
    eps = Fraction(1, 8 ** (len(word) + 2))  # amplified to 1/64 at the tail
    assert read_word(x + sign * eps, layout) == decode_word(x, layout)


def test_read_rejects_values_far_from_the_encoding_set():
    layout = Layout(("p",), SYMBOLS, 8)
    assert read_word(Fraction(7, 8), layout) is None  # digit 7 > largest digit 5 (+1/2)
    assert read((0, 0, Fraction(5, 2)), Layout(("p", "q"), SYMBOLS, 8)) is None  # state 2.5 -> 3: no such state
    assert read((0, 0, Fraction(9, 10)), Layout(("p", "q"), SYMBOLS, 8)).state == "q"


@pytest.mark.parametrize(
    ("base", "dtype", "push", "bound"),
    [
        (4, "float32", 11, 11),
        (4, "float64", 25, 25),
        (8, "float32", 7, 7),
        (8, "float64", 17, 16),
        (16, "float64", 12, 12),
    ],
)
def test_predicted_capacity(base, dtype, push, bound):
    assert predicted_capacity(base, dtype, digit=3) == push
    assert predicted_capacity(base, dtype) == bound


def test_no_prediction_for_other_bases():
    assert predicted_capacity(6, "float64") is None


@pytest.mark.parametrize("base", [4, 8, 16])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_filler_breaks_exactly_where_predicted(base, dtype):
    """The filler pushes digit 3 every step, so step t has t digits."""
    prof = precision_profile(filler(), steps=30, dtype=dtype, base=base)
    capacity = predicted_capacity(base, dtype, digit=3)
    assert prof.max_correct_digits == capacity
    assert prof.first_inexact.step == prof.first_failure.step == capacity + 1  # abrupt


def test_zigzag_respects_the_worst_case_bound():
    prof = precision_profile(zigzag(), steps=80, dtype="float32", base=4)
    assert prof.max_correct_digits >= predicted_capacity(4, "float32")
    assert prof.first_failure is not None


@given(machine_and_word())
def test_centred_ramps_are_still_exact(case):
    tm, word = case
    assert run(tm, word, steps=30, network=build_network(tm, ramps="centre")).agree


def test_centred_ramps_tolerate_rounding():
    """Base 6 rounds from the first step; centred ramps keep the error from
    being amplified (docs/numerics.md)."""
    edge = precision_profile(zigzag(), steps=60, dtype="float32", base=6, ramps="edge")
    centre = precision_profile(zigzag(), steps=60, dtype="float32", base=6, ramps="centre")
    assert edge.max_correct_digits <= 5 < 7 <= centre.max_correct_digits


def test_invalid_ramps_rejected():
    with pytest.raises(ValueError):
        build_network(filler(), ramps="smooth")


def test_cli_precision(capsys):
    assert main(["precision", "filler", "-n", "30", "--every", "10"]) == 0
    out = capsys.readouterr().out
    assert "longest correct register: 11 digits (guaranteed exact up to 11)" in out
    assert "longest correct register: 25 digits" in out


# -- step 2: inside one step ------------------------------------------------------


def test_pass_names():
    from relu_tm.numerics import pass_of

    net = build_network(filler())
    names = {pass_of(layer.note) for layer in net.program.layers}
    assert names == {"read", "table", "scale", "left", "state"}


def test_first_rounding_is_the_scaling_product():
    """At the filler's breaking point the first rounded value is κ·l in pass 2."""
    from relu_tm import encode
    from relu_tm.numerics import first_rounding

    net = build_network(filler())
    start = net.iterate_exact(encode(filler().initial_configuration(), net.layout), 25)[-1]  # 25 digits
    err = first_rounding(net, start, "float64")
    assert err.pass_name == "scale" and err.neuron == "w"
    assert "(1-1/beta) l" in net.program.layers[err.layer - 1].note


def test_first_rounding_reports_unrepresentable_input():
    from relu_tm import encode
    from relu_tm.numerics import first_rounding

    tm = filler()
    net = build_network(tm)
    long = encode(tm.trace((), 30)[-1], net.layout)  # 30 digits do not fit into float64
    err = first_rounding(net, long, "float64")
    assert err.layer == 0 and err.pass_name == "input"


def test_pass_errors_are_zero_for_exact_runs():
    from relu_tm import encode
    from relu_tm.numerics import pass_errors

    net = build_network(zigzag())
    x = net.iterate_exact(encode(zigzag().initial_configuration(), net.layout), 10)[-1]
    errs = pass_errors(net, x, [float(v) for v in x])
    assert set(errs) == {"input", "read", "table", "scale", "left", "state", "output"}
    assert not any(errs.values())
