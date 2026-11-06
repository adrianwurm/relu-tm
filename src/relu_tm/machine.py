"""Deterministic Turing machines - the classical reference ("oracle").

A configuration is stored in *two-stack form*, which is also the form used by
the network encoding (see :mod:`relu_tm.encoding`):

* ``left``  - the cells to the left of the head, **nearest cell first**;
* ``right`` - the head cell followed by the cells to its right.

Both tuples are kept in a normal form without trailing blanks, so two
configurations that describe the same tape compare equal.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

Symbol = str
State = str
Action = tuple[State, Symbol, str]  # (next state, written symbol, move)

MOVES = ("L", "R", "S")


# ---------------------------------------------------------------------------
# A tiny common interface for all machines in this package
# ---------------------------------------------------------------------------


class Machine(Protocol):
    """The minimal interface shared by Turing machines, DFAs and DPDAs."""

    def initial_configuration(self, word: Sequence[Symbol] | str) -> Any: ...

    def step(self, config: Any) -> Any: ...

    def is_halted(self, config: Any) -> bool: ...

    def is_accepting(self, config: Any) -> bool: ...


def iterate(machine: Machine, word: Sequence[Symbol] | str, max_steps: int) -> Iterator[Any]:
    """Yield the initial configuration and every successor until the machine
    halts or ``max_steps`` steps have been taken."""
    config = machine.initial_configuration(word)
    yield config
    for _ in range(max_steps):
        if machine.is_halted(config):
            return
        config = machine.step(config)
        yield config


def accepts(machine: Machine, word: Sequence[Symbol] | str, max_steps: int = 10_000) -> bool | None:
    """``True``/``False`` if the machine halts within ``max_steps`` steps and
    accepts/rejects, ``None`` if it is still running."""
    last = list(iterate(machine, word, max_steps))[-1]
    if not machine.is_halted(last):
        return None
    return machine.is_accepting(last)


def as_symbols(word: Sequence[Symbol] | str) -> tuple[Symbol, ...]:
    """Turn ``"0110"`` into ``("0", "1", "1", "0")``; sequences pass through."""
    return tuple(word)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


def _strip(cells: tuple[Symbol, ...], blank: Symbol) -> tuple[Symbol, ...]:
    end = len(cells)
    while end and cells[end - 1] == blank:
        end -= 1
    return cells[:end]


@dataclass(frozen=True)
class Configuration:
    """State, tape and head position of a Turing machine (two-stack form)."""

    state: State
    left: tuple[Symbol, ...] = ()
    right: tuple[Symbol, ...] = ()
    blank: Symbol = "0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", _strip(tuple(self.left), self.blank))
        object.__setattr__(self, "right", _strip(tuple(self.right), self.blank))

    @property
    def head_symbol(self) -> Symbol:
        return self.right[0] if self.right else self.blank

    def cells(self) -> tuple[int, tuple[Symbol, ...]]:
        """Return ``(position of the head, tape cells)`` with the head cell
        included even if it is blank."""
        left = tuple(reversed(self.left))
        right = self.right or (self.blank,)
        return len(left), left + right

    def render(self) -> str:
        """Human readable form, e.g. ``"B: 1 1 [0] 1"``."""
        head, cells = self.cells()
        parts = [f"[{c}]" if i == head else c for i, c in enumerate(cells)]
        return f"{self.state}: " + " ".join(parts)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.render()


# ---------------------------------------------------------------------------
# Turing machines
# ---------------------------------------------------------------------------


def _unique(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


@dataclass(frozen=True)
class TuringMachine:
    """A deterministic single-tape Turing machine.

    ``transitions`` maps ``(state, read symbol)`` to
    ``(next state, written symbol, move)`` with ``move`` in ``"L"``, ``"R"``,
    ``"S"`` (stay).  A machine halts when it enters a state of
    ``halt_states`` or when no transition is defined for the current
    ``(state, symbol)`` pair.  ``accept_states`` (a subset of the states)
    decides acceptance of a halted configuration.

    ``states`` and ``symbols`` fix an order (used by the encoding).  They are
    inferred from the transition table if omitted; the blank is always the
    first symbol and the start state the first state.
    """

    transitions: Mapping[tuple[State, Symbol], Action] = field(hash=False)
    start: State
    blank: Symbol = "0"
    halt_states: frozenset[State] = frozenset()
    accept_states: frozenset[State] = frozenset()
    states: tuple[State, ...] = ()
    symbols: tuple[Symbol, ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "transitions", dict(self.transitions))
        set_(self, "halt_states", frozenset(self.halt_states))
        set_(self, "accept_states", frozenset(self.accept_states))

        inferred_states = _unique(
            [self.start]
            + [s for (s, _a) in self.transitions]
            + [t[0] for t in self.transitions.values()]
            + sorted(self.halt_states)
            + sorted(self.accept_states)
        )
        inferred_symbols = _unique(
            [self.blank] + [a for (_s, a) in self.transitions] + [t[1] for t in self.transitions.values()]
        )
        states = _unique([*self.states, *inferred_states]) if self.states else inferred_states
        symbols = _unique([self.blank, *self.symbols, *inferred_symbols]) if self.symbols else inferred_symbols
        if states[0] != self.start:
            states = _unique([self.start, *states])
        set_(self, "states", states)
        set_(self, "symbols", symbols)

        for (s, a), (_t, b, m) in self.transitions.items():
            if m not in MOVES:
                raise ValueError(f"invalid move {m!r} in transition ({s}, {a}); use one of {MOVES}")
            for sym in (a, b):
                if sym not in symbols:
                    raise ValueError(f"unknown symbol {sym!r}")
        if not self.accept_states <= set(states):
            raise ValueError("accept_states must be states of the machine")

    # -- semantics ---------------------------------------------------------

    def action(self, state: State, symbol: Symbol) -> Action | None:
        """The transition taken in ``state`` reading ``symbol`` (``None`` = halt)."""
        if state in self.halt_states:
            return None
        return self.transitions.get((state, symbol))

    def initial_configuration(self, word: Sequence[Symbol] | str = ()) -> Configuration:
        word = as_symbols(word)
        for sym in word:
            if sym not in self.symbols:
                raise ValueError(f"symbol {sym!r} is not in the tape alphabet {self.symbols}")
        return Configuration(self.start, (), word, self.blank)

    def is_halted(self, config: Configuration) -> bool:
        return self.action(config.state, config.head_symbol) is None

    def is_accepting(self, config: Configuration) -> bool:
        return self.is_halted(config) and config.state in self.accept_states

    def step(self, config: Configuration) -> Configuration:
        """One step; a halted configuration is returned unchanged (fixed point)."""
        act = self.action(config.state, config.head_symbol)
        if act is None:
            return config
        new_state, written, move = act
        rest = config.right[1:]
        if move == "R":
            left, right = (written, *config.left), rest
        elif move == "L":
            new_head = config.left[0] if config.left else self.blank
            left, right = config.left[1:], (new_head, written, *rest)
        else:  # "S"
            left, right = config.left, (written, *rest)
        return Configuration(new_state, left, right, self.blank)

    def trace(self, word: Sequence[Symbol] | str = (), steps: int = 10) -> list[Configuration]:
        """Exactly ``steps + 1`` configurations (halted ones repeat)."""
        config = self.initial_configuration(word)
        out = [config]
        for _ in range(steps):
            config = self.step(config)
            out.append(config)
        return out

    def run(self, word: Sequence[Symbol] | str = (), max_steps: int = 10_000) -> list[Configuration]:
        """Configurations until halting (or ``max_steps``)."""
        return list(iterate(self, word, max_steps))

    def halting_time(self, word: Sequence[Symbol] | str = (), max_steps: int = 10_000) -> int | None:
        configs = self.run(word, max_steps)
        return len(configs) - 1 if self.is_halted(configs[-1]) else None

    # -- the bbchallenge "standard text format" ------------------------------

    @classmethod
    def from_standard_format(cls, text: str, name: str = "") -> TuringMachine:
        """Parse the compact format used by bbchallenge.org.

        Example: BB(2) is ``"1RB1LB_1LA1RZ"``.  States are ``A, B, C, ...`` in
        order, symbols are ``0, 1, ...`` with ``0`` as blank, ``---`` marks an
        undefined (halting) transition and a target state that is not defined
        (conventionally ``Z`` or ``H``) is a halting state.
        """
        groups = text.strip().split("_")
        n_symbols = len(groups[0]) // 3
        if not groups[0] or any(len(g) != 3 * n_symbols for g in groups):
            raise ValueError(f"malformed machine {text!r}")
        states = [chr(ord("A") + i) for i in range(len(groups))]
        symbols = [str(i) for i in range(n_symbols)]
        transitions: dict[tuple[State, Symbol], Action] = {}
        halts: set[str] = set()
        for state, group in zip(states, groups, strict=True):
            for i, sym in enumerate(symbols):
                chunk = group[3 * i : 3 * i + 3]
                if chunk == "---":
                    continue
                if not re.fullmatch(r"[0-9][LR][A-Z]", chunk):
                    raise ValueError(f"malformed transition {chunk!r} in {text!r}")
                write, move, target = chunk
                if write not in symbols:
                    raise ValueError(f"symbol {write!r} out of range in {text!r}")
                transitions[(state, sym)] = (target, write, move)
                if target not in states:
                    halts.add(target)
        return cls(
            transitions,
            start="A",
            blank="0",
            halt_states=frozenset(halts),
            states=tuple(states + sorted(halts)),
            symbols=tuple(symbols),
            name=name or text,
        )

    def to_standard_format(self) -> str:
        """Write the machine in the bbchallenge format (inverse of
        :meth:`from_standard_format` up to renaming).

        Symbols become ``0, 1, ...`` in the order of :attr:`symbols` (blank =
        ``0``), the non-halting states become ``A, B, ...`` (start first) and
        every halting state becomes ``Z``.  The format has no stay moves and no
        accepting states, so machines using ``S`` are rejected and the
        accept/reject distinction is lost.
        """
        if self.start in self.halt_states:
            raise ValueError("the start state must not be a halting state")
        working = [s for s in self.states if s not in self.halt_states]
        if len(working) > 25:
            raise ValueError("the standard format supports at most 25 non-halting states")
        if len(self.symbols) > 10:
            raise ValueError("the standard format supports at most 10 symbols")
        letter = {s: chr(ord("A") + i) for i, s in enumerate(working)}
        digit = {a: str(i) for i, a in enumerate(self.symbols)}
        groups = []
        for s in working:
            chunks = []
            for a in self.symbols:
                act = self.action(s, a)
                if act is None:
                    chunks.append("---")
                    continue
                target, write, move = act
                if move == "S":
                    raise ValueError(f"stay move in ({s}, {a}) cannot be written in the standard format")
                chunks.append(f"{digit[write]}{move}{letter.get(target, 'Z')}")
            groups.append("".join(chunks))
        return "_".join(groups)

    def ones(self, config: Configuration) -> int:
        """Number of non-blank cells (the "score" of a busy beaver)."""
        return sum(1 for c in config.left + config.right if c != self.blank)
