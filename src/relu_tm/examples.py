"""Example machines used in the tests, the CLI and the notebook."""

from __future__ import annotations

from collections.abc import Callable

from .automata import DFA, DPDA, EPS
from .machine import TuringMachine

# Busy beaver champions in the bbchallenge.org standard format.
BB2 = "1RB1LB_1LA1RZ"  # halts after 6 steps with 4 ones
BB3 = "1RB1RZ_1LB0RC_1LC1LA"  # halts after 21 steps with 5 ones (max. steps)
BB4 = "1RB1LB_1LA0LC_1RZ1LD_1RD0RA"  # halts after 107 steps with 13 ones


def busy_beaver(n: int) -> TuringMachine:
    """The step-champion busy beaver with ``n`` in {2, 3, 4} states."""
    table = {2: BB2, 3: BB3, 4: BB4}
    return TuringMachine.from_standard_format(table[n], name=f"BB({n})")


def filler() -> TuringMachine:
    """Writes 1 and moves right forever: the left register gains one digit
    per step - the simplest stress test for floating-point precision."""
    return TuringMachine.from_standard_format("1RA1RA", name="filler")


def zigzag() -> TuringMachine:
    """Sweeps back and forth and adds a 1 at each end: the tape grows by one
    cell per sweep and every cell is read again - a stress test for
    rounding errors that are read back."""
    return TuringMachine.from_standard_format("1LB1RA_1RA1LB", name="zigzag")


def pendulum() -> TuringMachine:
    """Moves between two cells forever: many steps on a short tape."""
    return TuringMachine.from_standard_format("1RB1RB_1LA1LA", name="pendulum")


def binary_counter() -> TuringMachine:
    """Counts upward in binary forever (head starts on the most significant
    bit; try the input ``"0"``)."""
    t = {
        ("right", "0"): ("right", "0", "R"),
        ("right", "1"): ("right", "1", "R"),
        ("right", "_"): ("carry", "_", "L"),
        ("carry", "1"): ("carry", "0", "L"),
        ("carry", "0"): ("right", "1", "R"),
        ("carry", "_"): ("right", "1", "R"),
    }
    return TuringMachine(t, start="right", blank="_", symbols=("_", "0", "1"), name="binary counter")


def palindrome() -> TuringMachine:
    """Decides whether a word over {a, b} is a palindrome."""
    t = {
        ("start", "a"): ("have_a", "_", "R"),
        ("start", "b"): ("have_b", "_", "R"),
        ("start", "_"): ("accept", "_", "S"),
        ("have_a", "a"): ("have_a", "a", "R"),
        ("have_a", "b"): ("have_a", "b", "R"),
        ("have_a", "_"): ("check_a", "_", "L"),
        ("have_b", "a"): ("have_b", "a", "R"),
        ("have_b", "b"): ("have_b", "b", "R"),
        ("have_b", "_"): ("check_b", "_", "L"),
        ("check_a", "a"): ("back", "_", "L"),
        ("check_a", "b"): ("reject", "b", "S"),
        ("check_a", "_"): ("accept", "_", "S"),
        ("check_b", "b"): ("back", "_", "L"),
        ("check_b", "a"): ("reject", "a", "S"),
        ("check_b", "_"): ("accept", "_", "S"),
        ("back", "a"): ("back", "a", "L"),
        ("back", "b"): ("back", "b", "L"),
        ("back", "_"): ("start", "_", "R"),
    }
    return TuringMachine(
        t,
        start="start",
        blank="_",
        halt_states=frozenset({"accept", "reject"}),
        accept_states=frozenset({"accept"}),
        symbols=("_", "a", "b"),
        name="palindrome",
    )


def binary_adder() -> TuringMachine:
    """Adds two binary numbers: ``"101+11"`` becomes ``"1000"``.

    Repeats "decrement the right number, increment the left number" until
    the right number is zero, then erases ``+`` and the right number.
    """
    t = {
        # go to the right end
        ("seek", "0"): ("seek", "0", "R"),
        ("seek", "1"): ("seek", "1", "R"),
        ("seek", "+"): ("seek", "+", "R"),
        ("seek", "_"): ("dec", "_", "L"),
        # decrement the right number (borrow moves left)
        ("dec", "0"): ("dec", "1", "L"),
        ("dec", "1"): ("back", "0", "L"),
        ("dec", "+"): ("clean", "_", "R"),  # it was zero: done
        # go back to the left number
        ("back", "0"): ("back", "0", "L"),
        ("back", "1"): ("back", "1", "L"),
        ("back", "+"): ("inc", "+", "L"),
        # increment the left number (carry moves left)
        ("inc", "1"): ("inc", "0", "L"),
        ("inc", "0"): ("seek", "1", "R"),
        ("inc", "_"): ("seek", "1", "R"),
        # erase the (now all-ones) right number
        ("clean", "1"): ("clean", "_", "R"),
        ("clean", "_"): ("done", "_", "S"),
    }
    return TuringMachine(
        t,
        start="seek",
        blank="_",
        halt_states=frozenset({"done"}),
        accept_states=frozenset({"done"}),
        symbols=("_", "0", "1", "+"),
        name="binary adder",
    )


# -- automata -------------------------------------------------------------------


def even_ones_dfa() -> DFA:
    """Binary words with an even number of 1s."""
    t = {("even", "0"): "even", ("even", "1"): "odd", ("odd", "0"): "odd", ("odd", "1"): "even"}
    return DFA(t, start="even", accept_states={"even"}, name="even number of 1s")


def divisible_by_3_dfa() -> DFA:
    """Binary numbers (most significant bit first) divisible by 3."""
    t = {(f"r{m}", b): f"r{(2 * m + int(b)) % 3}" for m in range(3) for b in "01"}
    return DFA(t, start="r0", accept_states={"r0"}, name="divisible by 3")


def balanced_parentheses_dpda() -> DPDA:
    t = {
        ("q0", "(", "Z"): ("q1", ("(", "Z")),
        ("q1", "(", "("): ("q1", ("(", "(")),
        ("q1", ")", "("): ("q1", ()),
        ("q1", EPS, "Z"): ("q0", ("Z",)),
    }
    return DPDA(t, start="q0", start_stack="Z", accept_states={"q0"}, name="balanced parentheses")


def anbn_dpda() -> DPDA:
    """The language { a^n b^n : n >= 0 }."""
    t = {
        ("q0", "a", "Z"): ("q1", ("A", "Z")),
        ("q1", "a", "A"): ("q1", ("A", "A")),
        ("q1", "b", "A"): ("q2", ()),
        ("q2", "b", "A"): ("q2", ()),
        ("q2", EPS, "Z"): ("acc", ("Z",)),
    }
    return DPDA(t, start="q0", start_stack="Z", accept_states={"q0", "acc"}, name="a^n b^n")


def mirror_dpda() -> DPDA:
    """The language { w c reverse(w) : w in {a, b}* }."""
    t: dict = {}
    for x in ("Z", "A", "B"):
        t[("push", "a", x)] = ("push", ("A", x))
        t[("push", "b", x)] = ("push", ("B", x))
        t[("push", "c", x)] = ("pop", (x,))
    t[("pop", "a", "A")] = ("pop", ())
    t[("pop", "b", "B")] = ("pop", ())
    t[("pop", EPS, "Z")] = ("acc", ("Z",))
    return DPDA(t, start="push", start_stack="Z", accept_states={"acc"}, name="w c w^R")


TURING_MACHINES: dict[str, Callable[[], TuringMachine]] = {
    "bb2": lambda: busy_beaver(2),
    "bb3": lambda: busy_beaver(3),
    "bb4": lambda: busy_beaver(4),
    "counter": binary_counter,
    "palindrome": palindrome,
    "adder": binary_adder,
    "filler": filler,
    "zigzag": zigzag,
    "pendulum": pendulum,
}

DEFAULT_INPUTS: dict[str, str] = {
    "counter": "0",
    "palindrome": "abba",
    "adder": "101+11",
}
