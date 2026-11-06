from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from relu_tm import DecodeError, Layout, decode, encode
from relu_tm.encoding import decode_word, default_base, encode_word
from relu_tm.examples import busy_beaver
from strategies import configurations

LAYOUT = Layout(states=("p", "q", "r"), symbols=("_", "a", "b"), base=8)


@pytest.mark.parametrize("base", [6, 8, 10, 16])
@given(config=configurations())
def test_roundtrip(base, config):
    layout = Layout(LAYOUT.states, LAYOUT.symbols, base)
    assert decode(encode(config, layout), layout) == config


def test_busy_beavers_use_the_papers_encoding():
    layout = Layout.for_machine(busy_beaver(4))
    assert layout.base == 4
    assert [layout.digit(s) for s in layout.symbols] == [1, 3]  # {1,3}-quadradics


def test_encode_word_example():
    # "ab" -> digits 3, 5 in base 8 -> 3/8 + 5/64
    assert encode_word(("a", "b"), LAYOUT) == Fraction(3, 8) + Fraction(5, 64)


@given(st.lists(st.sampled_from(LAYOUT.symbols), min_size=1, max_size=15))
def test_gap_property(word):
    """The value lies in [d/β, (d+1)/β] for its leading digit d, i.e. the
    leading digit can be read with gaps of width 1/β between the cases."""
    x = encode_word(word, LAYOUT)
    d = LAYOUT.digit(word[0])
    assert Fraction(d, 8) <= x <= Fraction(d + 1, 8)


@pytest.mark.parametrize(
    "value",
    [
        Fraction(1, 64),  # digit 0 followed by a non-zero digit
        Fraction(2, 8),  # even digit
        Fraction(7, 8),  # digit 7 > 2|Γ|-1
        Fraction(1),  # outside [0, 1)
        Fraction(1, 3),  # infinite expansion
    ],
)
def test_decode_rejects_invalid_values(value):
    with pytest.raises(DecodeError):
        decode_word(value, LAYOUT, max_digits=200)


def test_decode_rejects_bad_state_register():
    with pytest.raises(DecodeError):
        decode((0, 0, Fraction(1, 2)), LAYOUT)
    with pytest.raises(DecodeError):
        decode((0, 0, 3), LAYOUT)


def test_decode_accepts_floats_exactly():
    config = decode((0.375, 0.0, 1.0), LAYOUT)
    assert config.left == ("a",) and config.state == "q"


@pytest.mark.parametrize(("k", "base"), [(1, 4), (2, 4), (3, 8), (4, 8), (5, 16), (9, 32)])
def test_default_base(k, base):
    assert default_base(k) == base


def test_layout_validation():
    with pytest.raises(ValueError):
        Layout(("p",), ("_", "a", "b"), base=4)  # base < 2|Γ|
    with pytest.raises(ValueError):
        Layout(("p",), ("_", "a"), base=5)  # odd base
