from itertools import product

import pytest
from hypothesis import given
from hypothesis import strategies as st

from relu_tm import DPDA, EPS, accepts
from relu_tm.examples import (
    anbn_dpda,
    balanced_parentheses_dpda,
    divisible_by_3_dfa,
    even_ones_dfa,
    mirror_dpda,
)
from strategies import dfas


def all_words(alphabet, max_len):
    for n in range(max_len + 1):
        yield from ("".join(w) for w in product(alphabet, repeat=n))


def balanced(w):
    depth = 0
    for ch in w:
        depth += 1 if ch == "(" else -1
        if depth < 0:
            return False
    return depth == 0


def is_anbn(w):
    n = len(w) // 2
    return w == "a" * n + "b" * n


def is_mirror(w):
    if w.count("c") != 1:
        return False
    left, right = w.split("c")
    return right == left[::-1]


DFA_CASES = [
    (even_ones_dfa(), lambda w: w.count("1") % 2 == 0),
    (divisible_by_3_dfa(), lambda w: int(w or "0", 2) % 3 == 0),
]
PDA_CASES = [
    (balanced_parentheses_dpda(), "()", balanced),
    (anbn_dpda(), "ab", is_anbn),
    (mirror_dpda(), "abc", is_mirror),
]


@pytest.mark.parametrize(("dfa", "spec"), DFA_CASES)
def test_dfa_examples(dfa, spec):
    tm = dfa.to_turing_machine()
    for w in all_words("01", 8):
        assert accepts(dfa, w) == spec(w)
        assert accepts(tm, w) == spec(w)


@pytest.mark.parametrize(("pda", "alphabet", "spec"), PDA_CASES)
def test_dpda_examples(pda, alphabet, spec):
    tm = pda.to_turing_machine()
    for w in all_words(alphabet, 6 if len(alphabet) == 2 else 5):
        assert accepts(pda, w) == spec(w), w
        assert accepts(tm, w, max_steps=100_000) == spec(w), w


@given(dfas(), st.text(alphabet="01", max_size=8))
def test_random_dfa_vs_its_turing_machine(dfa, word):
    tm = dfa.to_turing_machine()
    assert accepts(dfa, word) == accepts(tm, word)
    # one TM step per DFA step: the states coincide step by step
    d_conf, t_conf = dfa.initial_configuration(word), tm.initial_configuration(word)
    while not dfa.is_halted(d_conf):
        d_conf, t_conf = dfa.step(d_conf), tm.step(t_conf)
        assert d_conf.state == t_conf.state


def test_dpda_must_be_deterministic():
    with pytest.raises(ValueError):
        DPDA({("q", "a", "Z"): ("q", ()), ("q", EPS, "Z"): ("q", ())}, start="q")


def test_dpda_symbol_clash_detected():
    pda = DPDA({("q", "#", "Z"): ("q", ())}, start="q")
    with pytest.raises(ValueError):
        pda.to_turing_machine()
