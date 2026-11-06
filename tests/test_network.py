"""The central property: network and emulator agree after every step."""

import pytest
import torch
from hypothesis import given

from relu_tm import build_network, decode, encode, run
from relu_tm.examples import (
    TURING_MACHINES,
    balanced_parentheses_dpda,
    binary_adder,
    binary_counter,
    busy_beaver,
    even_ones_dfa,
    filler,
    mirror_dpda,
    palindrome,
)
from strategies import machine_and_word


def assert_agree(result):
    assert result.agree, "\n" + result.table()


@pytest.mark.parametrize("name", sorted(TURING_MACHINES))
def test_width_and_depth(name):
    tm = TURING_MACHINES[name]()
    net = build_network(tm)
    assert all(len(W) == 4 for W, _ in net.hidden), "every hidden layer has width 4"
    k, n = len(tm.symbols), len(tm.states)
    assert net.depth == 5 * k + 2 * n * (k + 1) + 12


@pytest.mark.parametrize("name", sorted(TURING_MACHINES))
def test_weights_are_dyadic_rationals(name):
    net = build_network(TURING_MACHINES[name]())
    for W, b in [*net.hidden, net.output]:
        for v in [x for row in W for x in row] + list(b):
            d = v.denominator
            assert d & (d - 1) == 0, f"denominator {d} is not a power of two"


@pytest.mark.parametrize("n", [2, 3, 4])
def test_busy_beavers_exact(n):
    assert_agree(run(busy_beaver(n)))


def test_binary_counter_exact():
    assert_agree(run(binary_counter(), "0", steps=300))


@pytest.mark.parametrize("word", ["", "a", "ab", "aba", "abba", "abab", "baab"])
def test_palindrome_exact(word):
    assert_agree(run(palindrome(), word))


@pytest.mark.parametrize(("x", "y"), [(0, 0), (5, 3), (6, 7), (1, 12)])
def test_adder_exact(x, y):
    result = run(binary_adder(), f"{x:b}+{y:b}")
    assert_agree(result)
    _, cells = result.network[-1].cells()
    assert int("".join(cells).strip("_"), 2) == x + y


def test_filler_exact_long_tape():
    assert_agree(run(filler(), steps=200))


@given(machine_and_word())
def test_random_machines_exact(case):
    """Random small TMs on random inputs, 30 steps, exact arithmetic."""
    tm, word = case
    assert_agree(run(tm, word, steps=30))


@given(machine_and_word(max_states=2, max_symbols=5))
def test_random_machines_other_bases(case):
    tm, word = case
    assert_agree(run(tm, word, steps=15, base=2 * len(tm.symbols) + 2))


def test_halted_configurations_are_fixed_points():
    tm = busy_beaver(3)
    net = build_network(tm)
    final = tm.run()[-1]
    x = encode(final, net.layout)
    for _ in range(3):
        x = net.step_exact(x)
        assert decode(x, net.layout) == final


def test_automata_through_their_turing_machines():
    for automaton, word in [(even_ones_dfa(), "1101"), (balanced_parentheses_dpda(), "(()"), (mirror_dpda(), "abcba")]:
        assert_agree(run(automaton.to_turing_machine(), word))


def test_torch_module_structure():
    net = build_network(busy_beaver(2))
    module = net.to_torch(torch.float64)
    linears = [m for m in module.modules() if isinstance(m, torch.nn.Linear)]
    assert linears[0].in_features == 3 and linears[-1].out_features == 3
    assert all(lin.out_features == 4 for lin in linears[:-1])
    assert len(linears) == net.depth + 1
    assert not any(p.requires_grad for p in module.parameters())
    acts = module.activations(torch.zeros(3, dtype=torch.float64))
    assert acts.shape == (net.depth, 4)


def test_torch_float64_matches_exact_on_bb4():
    assert_agree(run(busy_beaver(4), backend="float64"))
