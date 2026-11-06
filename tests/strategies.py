"""Hypothesis strategies: random small machines, words and configurations."""

from __future__ import annotations

from hypothesis import strategies as st

from relu_tm import DFA, Configuration, TuringMachine

HALT = "H"


@st.composite
def turing_machines(draw, max_states: int = 4, max_symbols: int = 3) -> TuringMachine:
    """Random deterministic TMs; about 10% of the transitions are left
    undefined (= halt) and some lead to the halting state ``H``."""
    n = draw(st.integers(1, max_states))
    k = draw(st.integers(2, max_symbols))
    states = [f"q{i}" for i in range(n)]
    symbols = ["_"] + [chr(ord("a") + i) for i in range(k - 1)]
    targets = [*states, HALT]
    transitions = {}
    for s in states:
        for a in symbols:
            if draw(st.integers(0, 9)) == 0:
                continue
            transitions[(s, a)] = (
                draw(st.sampled_from(targets)),
                draw(st.sampled_from(symbols)),
                draw(st.sampled_from("LRS")),
            )
    return TuringMachine(
        transitions,
        start="q0",
        blank="_",
        halt_states=frozenset({HALT}),
        states=(*states, HALT),
        symbols=tuple(symbols),
    )


def words(tm: TuringMachine, max_size: int = 6):
    """Input words; may contain blanks in the middle."""
    return st.lists(st.sampled_from(tm.symbols), max_size=max_size).map(tuple)


@st.composite
def machine_and_word(draw, max_states: int = 4, max_symbols: int = 3, max_size: int = 6):
    tm = draw(turing_machines(max_states, max_symbols))
    return tm, draw(words(tm, max_size))


@st.composite
def configurations(draw, symbols=("_", "a", "b"), states=("p", "q", "r"), max_size: int = 12) -> Configuration:
    cells = st.lists(st.sampled_from(symbols), max_size=max_size).map(tuple)
    return Configuration(draw(st.sampled_from(states)), draw(cells), draw(cells), blank=symbols[0])


@st.composite
def dfas(draw, max_states: int = 4) -> DFA:
    n = draw(st.integers(1, max_states))
    states = [f"s{i}" for i in range(n)]
    transitions = {}
    for s in states:
        for a in "01":
            if draw(st.integers(0, 7)):
                transitions[(s, a)] = draw(st.sampled_from(states))
    accept = draw(st.sets(st.sampled_from(states)))
    return DFA(transitions, start="s0", accept_states=accept, input_symbols=("0", "1"))
