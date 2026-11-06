"""The dyadic integer backend computes exactly what the Fraction reference computes."""

from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from relu_tm import build_network, encode, run
from relu_tm.cli import main
from relu_tm.dyadic import DyadicNetwork, to_fixed, to_fractions
from relu_tm.examples import busy_beaver, filler, zigzag
from strategies import machine_and_word


@given(machine_and_word(max_states=4, max_symbols=4))
def test_identical_to_fractions_on_random_machines(case):
    tm, word = case
    net = build_network(tm)
    x0 = encode(tm.initial_configuration(word), net.layout)
    assert net.iterate_dyadic(x0, 30) == net.iterate_exact(x0, 30)


@pytest.mark.parametrize("ramps", ["edge", "centre"])
def test_identical_on_long_runs(ramps):
    for tm, steps in ((filler(), 300), (zigzag(), 500), (busy_beaver(4), 107)):
        net = build_network(tm, ramps=ramps)
        x0 = encode(tm.initial_configuration(), net.layout)
        assert net.iterate_dyadic(x0, steps) == net.iterate_exact(x0, steps)


def test_precision_grows_only_as_needed():
    net = build_network(zigzag())
    states = net.dyadic().iterate_fixed(encode(zigzag().initial_configuration(), net.layout), 1000)
    xs, k = states[-1]
    assert k <= 2 * 45 + 2  # 1000 steps: tape of ~44 cells, 2 bits per base-4 digit
    assert any(x & 1 for x in xs if x)  # normalised: no common factor of two left


@given(
    st.lists(
        st.fractions(min_value=0, max_value=1).map(lambda f: Fraction(round(f * 1024), 1024)), min_size=3, max_size=3
    )
)
def test_fixed_roundtrip(values):
    assert to_fractions(to_fixed(values)) == tuple(values)


def test_other_bases_are_rejected():
    net = build_network(filler(), base=6)
    assert not net.is_dyadic()
    with pytest.raises(ValueError, match="power of two"):
        DyadicNetwork(net)


def test_run_and_cli_backend(capsys):
    assert run(busy_beaver(4), backend="dyadic").agree
    assert main(["run", "bb3", "-b", "dyadic"]) == 0
    assert "dyadic" in capsys.readouterr().out
