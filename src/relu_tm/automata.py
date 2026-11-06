"""Finite automata and deterministic pushdown automata.

Both share the small :class:`relu_tm.machine.Machine` interface with Turing
machines, and both can be compiled into an equivalent
:class:`~relu_tm.machine.TuringMachine` (``to_turing_machine``), which is how
the ReLU network simulates them.  The automaton itself then serves as an
independent oracle for the compiled machine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .machine import Action, State, Symbol, TuringMachine, as_symbols

ACCEPT = "ACCEPT"
REJECT = "REJECT"
EPS = ""  # the empty input word in DPDA transitions


# ---------------------------------------------------------------------------
# Deterministic finite automata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DFAConfig:
    state: State
    remaining: tuple[Symbol, ...]


@dataclass(frozen=True)
class DFA:
    """A (possibly partial) deterministic finite automaton.

    A missing transition rejects.  The word is accepted if it is read
    completely and the automaton ends in an accepting state.
    """

    transitions: Mapping[tuple[State, Symbol], State] = field(hash=False)
    start: State
    accept_states: frozenset[State] = frozenset()
    name: str = ""
    input_symbols: tuple[Symbol, ...] = ()  # optional; inferred from the table

    def __post_init__(self) -> None:
        object.__setattr__(self, "transitions", dict(self.transitions))
        object.__setattr__(self, "accept_states", frozenset(self.accept_states))

    @property
    def alphabet(self) -> tuple[Symbol, ...]:
        return tuple(dict.fromkeys([*self.input_symbols, *(a for (_q, a) in self.transitions)]))

    @property
    def states(self) -> tuple[State, ...]:
        qs = [self.start, *(q for (q, _a) in self.transitions), *self.transitions.values()]
        return tuple(dict.fromkeys([*qs, *sorted(self.accept_states)]))

    def initial_configuration(self, word: Sequence[Symbol] | str) -> DFAConfig:
        return DFAConfig(self.start, as_symbols(word))

    def is_halted(self, c: DFAConfig) -> bool:
        return not c.remaining or (c.state, c.remaining[0]) not in self.transitions

    def is_accepting(self, c: DFAConfig) -> bool:
        return not c.remaining and c.state in self.accept_states

    def step(self, c: DFAConfig) -> DFAConfig:
        if self.is_halted(c):
            return c
        return DFAConfig(self.transitions[(c.state, c.remaining[0])], c.remaining[1:])

    def to_turing_machine(self, blank: Symbol = "_") -> TuringMachine:
        """One TM step per DFA step: read, keep the symbol, move right.
        On the blank after the input the TM moves to ACCEPT or REJECT."""
        if blank in self.alphabet:
            raise ValueError("the blank must not be an input symbol")
        states = self.states
        delta: dict[tuple[State, Symbol], Action] = {}
        for (q, a), p in self.transitions.items():
            delta[(q, a)] = (p, a, "R")
        for q in states:
            delta[(q, blank)] = (ACCEPT if q in self.accept_states else REJECT, blank, "S")
        return TuringMachine(
            delta,
            start=self.start,
            blank=blank,
            halt_states=frozenset({ACCEPT, REJECT}),
            accept_states=frozenset({ACCEPT}),
            states=(*states, ACCEPT, REJECT),
            symbols=(blank, *self.alphabet),
            name=f"TM({self.name or 'DFA'})",
        )


# ---------------------------------------------------------------------------
# Deterministic pushdown automata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDAConfig:
    state: State
    remaining: tuple[Symbol, ...]
    stack: tuple[Symbol, ...]  # top first


@dataclass(frozen=True)
class DPDA:
    """A deterministic pushdown automaton.

    ``transitions`` maps ``(state, input symbol or EPS, stack top)`` to
    ``(next state, pushed string)``; the pushed string replaces the top and
    is given **top first**.  Determinism: if an ε-move exists for
    ``(state, top)`` there is no input move for that pair.

    Acceptance (kept simple on purpose): the automaton halts - no move
    applies - after reading the whole input, in an accepting state.
    """

    transitions: Mapping[tuple[State, Symbol, Symbol], tuple[State, tuple[Symbol, ...]]] = field(hash=False)
    start: State
    start_stack: Symbol = "Z"
    accept_states: frozenset[State] = frozenset()
    name: str = ""

    def __post_init__(self) -> None:
        delta = {k: (p, tuple(push)) for k, (p, push) in self.transitions.items()}
        object.__setattr__(self, "transitions", delta)
        object.__setattr__(self, "accept_states", frozenset(self.accept_states))
        for q, a, x in delta:
            if a != EPS and (q, EPS, x) in delta:
                raise ValueError(f"not deterministic: ε-move and input move for ({q}, {x})")

    @property
    def input_alphabet(self) -> tuple[Symbol, ...]:
        return tuple(dict.fromkeys(a for (_q, a, _x) in self.transitions if a != EPS))

    @property
    def stack_alphabet(self) -> tuple[Symbol, ...]:
        syms = [self.start_stack]
        for (_q, _a, x), (_p, push) in self.transitions.items():
            syms += [x, *push]
        return tuple(dict.fromkeys(syms))

    @property
    def states(self) -> tuple[State, ...]:
        qs = [self.start]
        for (q, _a, _x), (p, _push) in self.transitions.items():
            qs += [q, p]
        return tuple(dict.fromkeys([*qs, *sorted(self.accept_states)]))

    def initial_configuration(self, word: Sequence[Symbol] | str) -> PDAConfig:
        return PDAConfig(self.start, as_symbols(word), (self.start_stack,))

    def _move(self, c: PDAConfig) -> tuple[bool, tuple[State, tuple[Symbol, ...]]] | None:
        if not c.stack:
            return None
        top = c.stack[0]
        if (c.state, EPS, top) in self.transitions:
            return False, self.transitions[(c.state, EPS, top)]
        if c.remaining and (c.state, c.remaining[0], top) in self.transitions:
            return True, self.transitions[(c.state, c.remaining[0], top)]
        return None

    def is_halted(self, c: PDAConfig) -> bool:
        return self._move(c) is None

    def is_accepting(self, c: PDAConfig) -> bool:
        return self.is_halted(c) and not c.remaining and c.state in self.accept_states

    def step(self, c: PDAConfig) -> PDAConfig:
        move = self._move(c)
        if move is None:
            return c
        consumes, (p, push) = move
        remaining = c.remaining[1:] if consumes else c.remaining
        return PDAConfig(p, remaining, push + c.stack[1:])

    # -- compilation to a single-tape Turing machine --------------------------

    def to_turing_machine(self, blank: Symbol = "_") -> TuringMachine:
        """Compile into a single-tape TM with the usual input convention.

        Tape layout: ``[Y1][Y2]...[Z] # · · a_i ... a_n`` - the stack grows to
        the *left* (top = leftmost stack cell, so pushing never needs to shift
        anything), ``#`` separates it from the input, consumed input cells are
        overwritten with ``·``.  For each PDA move the TM walks from the stack
        top to the first unconsumed input symbol and back, so one PDA step
        costs O(stack height + input length) TM steps.
        """
        sep, used = "#", "·"
        sigma = self.input_alphabet
        gamma = self.stack_alphabet
        stack_sym = {x: f"[{x}]" for x in gamma}
        reserved = {blank, sep, used}
        if reserved & set(sigma) or reserved & set(stack_sym.values()) or set(sigma) & set(stack_sym.values()):
            raise ValueError("symbol clash between input/stack symbols and the TM markers")

        delta: dict[tuple[State, Symbol], Action] = {}
        f = self.accept_states

        # Write "[Z] #" in front of the input and park the head on [Z].
        for s in (*sigma, blank):
            delta[("init0", s)] = ("init1", s, "L")
        delta[("init1", blank)] = ("init2", sep, "L")
        delta[("init2", blank)] = (f"top|{self.start}", stack_sym[self.start_stack], "S")

        trans_ids = {key: i for i, key in enumerate(self.transitions)}

        def apply_moves(src_state: State, src_symbol: Symbol, tid: int, p: State, push: tuple[Symbol, ...]) -> None:
            """Replace the top cell (under the head) by ``push`` (top first)."""
            if not push:
                delta[(src_state, src_symbol)] = (f"top|{p}", blank, "R")
                return
            k = len(push)
            if k == 1:
                delta[(src_state, src_symbol)] = (f"top|{p}", stack_sym[push[0]], "S")
                return
            delta[(src_state, src_symbol)] = (f"push|{tid}|{k - 1}", stack_sym[push[k - 1]], "L")
            for i in range(k - 1, 0, -1):
                nxt = (f"top|{p}", "S") if i == 1 else (f"push|{tid}|{i - 1}", "L")
                delta[(f"push|{tid}|{i}", blank)] = (nxt[0], stack_sym[push[i - 1]], nxt[1])

        for q in self.states:
            top = f"top|{q}"
            # Empty stack: the PDA is halted.
            delta[(top, sep)] = ("check", sep, "R") if q in f else (REJECT, sep, "S")
            for x in gamma:
                key_eps = (q, EPS, x)
                if key_eps in self.transitions:
                    p, push = self.transitions[key_eps]
                    apply_moves(top, stack_sym[x], trans_ids[key_eps], p, push)
                    continue
                seek1, seek2 = f"seek1|{q}|{x}", f"seek2|{q}|{x}"
                delta[(top, stack_sym[x])] = (seek1, stack_sym[x], "R")
                for y in gamma:
                    delta[(seek1, stack_sym[y])] = (seek1, stack_sym[y], "R")
                delta[(seek1, sep)] = (seek2, sep, "R")
                delta[(seek2, used)] = (seek2, used, "R")
                delta[(seek2, blank)] = (ACCEPT if q in f else REJECT, blank, "S")
                for a in sigma:
                    key = (q, a, x)
                    if key in self.transitions:
                        delta[(seek2, a)] = (f"ret2|{trans_ids[key]}", used, "L")
                    else:
                        delta[(seek2, a)] = (REJECT, a, "S")

        for key, (p, push) in self.transitions.items():
            q, a, x = key
            if a == EPS:
                continue
            tid = trans_ids[key]
            ret2, ret1, apply = f"ret2|{tid}", f"ret1|{tid}", f"apply|{tid}"
            delta[(ret2, used)] = (ret2, used, "L")
            delta[(ret2, sep)] = (ret1, sep, "L")
            for y in gamma:
                delta[(ret1, stack_sym[y])] = (ret1, stack_sym[y], "L")
            delta[(ret1, blank)] = (apply, blank, "R")
            apply_moves(apply, stack_sym[x], tid, p, push)

        delta[("check", used)] = ("check", used, "R")
        delta[("check", blank)] = (ACCEPT, blank, "S")
        for a in sigma:
            delta[("check", a)] = (REJECT, a, "S")

        return TuringMachine(
            delta,
            start="init0",
            blank=blank,
            halt_states=frozenset({ACCEPT, REJECT}),
            accept_states=frozenset({ACCEPT}),
            symbols=(blank, *sigma, sep, used, *stack_sym.values()),
            name=f"TM({self.name or 'DPDA'})",
        )
