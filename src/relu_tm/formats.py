"""Reading and writing machines in the formats of other tools.

Importing translates a file into the package's own machine types
(:class:`~relu_tm.machine.TuringMachine`, :class:`~relu_tm.automata.DFA`,
:class:`~relu_tm.automata.DPDA`); :func:`relu_tm.build_network` then works
as usual.  Exporting is the reverse direction.

=====================  ===========  ======================================  =========================
format                 suffix       machines                                extra software
=====================  ===========  ======================================  =========================
bbchallenge text       ``.txt``     Turing machines (symbols 0-9, L/R)      none
JFLAP                  ``.jff``     Turing machines, finite automata, PDAs  none (stdlib XML)
turingmachine.io       ``.yaml``    Turing machines (L/R)                   PyYAML (``relu-tm[formats]``)
=====================  ===========  ======================================  =========================

Semantics that the formats do not share with this package are documented
at each function (e.g. JFLAP accepts as soon as a final state is reached,
so JFLAP final states become halting accept states).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from .automata import DFA, DPDA, EPS
from .machine import Action, State, Symbol, TuringMachine

try:  # protects against entity-expansion attacks if available
    from defusedxml import ElementTree as ET
except ImportError:  # pragma: no cover - the stdlib parser is fine for trusted files
    from xml.etree import ElementTree as ET

AnyMachine = TuringMachine | DFA | DPDA


class FormatError(ValueError):
    """The file is well-formed but cannot be translated faithfully."""


@dataclass(frozen=True)
class MachineFile:
    """A machine read from a file, plus an input word stored in the file."""

    machine: AnyMachine
    input: tuple[Symbol, ...] = ()
    format: str = ""

    def turing_machine(self) -> TuringMachine:
        """The machine itself, or the TM that an automaton compiles to."""
        if isinstance(self.machine, TuringMachine):
            return self.machine
        return self.machine.to_turing_machine()


FORMATS = ("bbchallenge", "jflap", "turingmachine.io")
_SUFFIXES = {".jff": "jflap", ".yaml": "turingmachine.io", ".yml": "turingmachine.io", ".txt": "bbchallenge"}


def load(path: str | Path, format: str | None = None) -> MachineFile:
    """Read a machine file; the format is taken from the suffix
    (``.jff``, ``.yaml``/``.yml``, ``.txt``) unless given explicitly."""
    path = Path(path)
    fmt = format or _SUFFIXES.get(path.suffix.lower())
    if fmt is None:
        raise FormatError(f"cannot tell the format of {path.name}; use one of {FORMATS}")
    result = loads(path.read_text(encoding="utf-8"), fmt)
    if result.machine.name:
        return result
    return MachineFile(replace(result.machine, name=path.stem), result.input, result.format)


def loads(text: str, format: str) -> MachineFile:
    """Like :func:`load`, for a string."""
    if format == "jflap":
        return MachineFile(from_jflap(text), (), format)
    if format == "turingmachine.io":
        return from_turingmachine_io(text)
    if format == "bbchallenge":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if len(lines) != 1:
            raise FormatError("a bbchallenge file must contain exactly one machine")
        return MachineFile(TuringMachine.from_standard_format(lines[0]), (), format)
    raise FormatError(f"unknown format {format!r}; use one of {FORMATS}")


def dumps(machine: AnyMachine, format: str, input: Iterable[Symbol] = ()) -> str:
    """Write a machine in one of :data:`FORMATS`."""
    if format == "jflap":
        return to_jflap(machine)
    if not isinstance(machine, TuringMachine):
        raise FormatError(f"{format} only describes Turing machines; compile the automaton first")
    if format == "turingmachine.io":
        return to_turingmachine_io(machine, input)
    if format == "bbchallenge":
        return machine.to_standard_format() + "\n"
    raise FormatError(f"unknown format {format!r}; use one of {FORMATS}")


# ---------------------------------------------------------------------------
# JFLAP (.jff)
# ---------------------------------------------------------------------------

JFLAP_BLANK = "□"  # how JFLAP displays the blank; in files it is an empty element


def _text(elem: Any, tag: str) -> str:
    """Text of the child ``tag`` (only tape 1 for multi-tape attributes)."""
    for child in elem.findall(tag):
        if child.get("tape") in (None, "1"):
            return child.text or ""
    return ""


@dataclass
class _JflapAutomaton:
    kind: str
    names: dict[str, State]  # id -> name
    order: list[State]
    initial: State
    finals: set[State]
    transitions: list[Any]  # <transition> elements


def _parse_jflap(text: str) -> _JflapAutomaton:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise FormatError(f"not a valid JFLAP/XML file: {exc}") from exc
    if root.tag != "structure":
        raise FormatError("a JFLAP file has a <structure> root element")
    kind = (root.findtext("type") or "").strip()
    tapes = (root.findtext("tapes") or "1").strip()
    if tapes != "1":
        raise FormatError(f"multi-tape machines ({tapes} tapes) are not supported")
    auto = root.find("automaton")
    if auto is None:  # very old files have no <automaton> wrapper
        auto = root
    elements = auto.findall("state") + auto.findall("block")
    names: dict[str, State] = {}
    order: list[State] = []
    initial = None
    finals: set[State] = set()
    taken: set[str] = set()
    for elem in sorted(elements, key=lambda e: int(e.get("id", "0"))):
        sid = elem.get("id")
        if sid is None:
            raise FormatError("state without id")
        name = elem.get("name") or f"q{sid}"
        if name in taken:
            name = f"{name}#{sid}"
        taken.add(name)
        tag = (elem.findtext("tag") or "").strip()
        if elem.tag == "block" and tag:
            sub = auto.find(tag) if re.fullmatch(r"[A-Za-z_][\w.-]*", tag) else None
            if sub is not None and len(sub):
                raise FormatError(f"state {name!r} is a building block (sub-machine); not supported")
        names[sid] = name
        order.append(name)
        if elem.find("initial") is not None:
            if initial is not None:
                raise FormatError("more than one initial state")
            initial = name
        if elem.find("final") is not None:
            finals.add(name)
    if initial is None:
        raise FormatError("no initial state")
    transitions = auto.findall("transition")
    for t in transitions:
        if t.get("block") == "true":
            raise FormatError("building-block transitions are not supported")
        for end in ("from", "to"):
            if (t.findtext(end) or "").strip() not in names:
                raise FormatError(f"transition refers to an unknown state id {t.findtext(end)!r}")
    return _JflapAutomaton(kind, names, order, initial, finals, transitions)


def from_jflap(text: str) -> AnyMachine:
    """Translate a JFLAP file (``<type>`` ``turing``, ``fa`` or ``pda``).

    * **Turing machines** (single tape): an empty ``<read/>``/``<write/>`` is
      the blank; the wildcards of JFLAP's simulator are supported - read
      ``~`` (any symbol), read ``!a`` (any symbol except ``a``), write ``~``
      (keep the symbol).  Wildcards range over the symbols that occur in the
      file.  JFLAP accepts as soon as a final state is reached, so final
      states become halting accept states.
    * **Finite automata** must be deterministic (no λ, one symbol per edge).
    * **Pushdown automata** must be deterministic; ``pop`` may be λ (then
      the move is repeated for every stack top) or one symbol, ``push`` is
      written top first, the start stack symbol is ``Z``.  Acceptance here is
      "halted after the whole input in a final state" (JFLAP lets you choose
      final state or empty stack at run time).

    Nondeterminism, multi-tape machines, building blocks and JFLAP's
    variable binding (``}``) raise :class:`FormatError`.
    """
    a = _parse_jflap(text)
    if a.kind == "turing":
        return _jflap_turing(a)
    if a.kind == "fa":
        return _jflap_fa(a)
    if a.kind == "pda":
        return _jflap_pda(a)
    raise FormatError(f"JFLAP type {a.kind!r} is not supported (use turing, fa or pda)")


def _jflap_turing(a: _JflapAutomaton) -> TuringMachine:
    raw = []
    literal: set[str] = set()
    for t in a.transitions:
        read, write, move = _text(t, "read"), _text(t, "write"), _text(t, "move").strip().upper()
        if move not in ("L", "R", "S"):
            raise FormatError(f"invalid move {move!r}")
        if "}" in read:
            raise FormatError("variable binding ('}') in transitions is not supported")
        if read.startswith("!"):  # '!' alone means 'not blank'
            pattern = ("not", read[1:] or "")
        elif read == "~":
            pattern = ("any", "")
        else:
            pattern = ("sym", read)
        for sym in (pattern[1], "" if write == "~" else write):
            if sym == JFLAP_BLANK:
                sym = ""
            if len(sym) > 1:
                raise FormatError(f"symbol {sym!r}: JFLAP tape symbols are single characters")
            if sym:
                literal.add(sym)
        raw.append((a.names[t.findtext("from").strip()], pattern, write, move, a.names[t.findtext("to").strip()]))

    blank = "_" if "_" not in literal else JFLAP_BLANK
    symbols = [blank, *sorted(literal)]

    def tm_symbol(s: str) -> Symbol:
        return blank if s in ("", JFLAP_BLANK) else s

    delta: dict[tuple[State, Symbol], Action] = {}
    for src, (kind, arg), write, move, dst in raw:
        if kind == "sym":
            matched = [tm_symbol(arg)]
        elif kind == "any":
            matched = list(symbols)
        else:
            matched = [s for s in symbols if s != tm_symbol(arg)]
        for sym in matched:
            out = (dst, sym if write == "~" else tm_symbol(write), move)
            key = (src, sym)
            if key in delta and delta[key] != out:
                raise FormatError(
                    f"nondeterministic: two transitions from {src!r} on {sym!r} "
                    "(JFLAP would explore both; replace the wildcard by explicit symbols)"
                )
            delta[key] = out
    return TuringMachine(
        delta,
        start=a.initial,
        blank=blank,
        halt_states=frozenset(a.finals),
        accept_states=frozenset(a.finals),
        states=tuple(a.order),
        symbols=tuple(symbols),
    )


def _jflap_fa(a: _JflapAutomaton) -> DFA:
    delta: dict[tuple[State, Symbol], State] = {}
    for t in a.transitions:
        read = t.findtext("read") or ""
        if read == "":
            raise FormatError("λ-transition: this is an NFA, not a DFA")
        if len(read) != 1:
            raise FormatError(f"edge label {read!r}: only single symbols are supported")
        src, dst = a.names[t.findtext("from").strip()], a.names[t.findtext("to").strip()]
        if delta.get((src, read), dst) != dst:
            raise FormatError(f"nondeterministic: two edges from {src!r} on {read!r}")
        delta[(src, read)] = dst
    alphabet = tuple(sorted({sym for (_q, sym) in delta}))
    return DFA(delta, start=a.initial, accept_states=frozenset(a.finals), input_symbols=alphabet)


def _jflap_pda(a: _JflapAutomaton, start_stack: Symbol = "Z") -> DPDA:
    raw = []
    stack_symbols = {start_stack}
    for t in a.transitions:
        read, pop, push = (t.findtext(x) or "" for x in ("read", "pop", "push"))
        if len(read) > 1:
            raise FormatError(f"read {read!r}: only single symbols or λ are supported")
        if len(pop) > 1:
            raise FormatError(f"pop {pop!r}: popping more than one symbol is not supported")
        stack_symbols.update(pop, push)
        raw.append((a.names[t.findtext("from").strip()], read, pop, tuple(push), a.names[t.findtext("to").strip()]))
    delta: dict[tuple[State, Symbol, Symbol], tuple[State, tuple[Symbol, ...]]] = {}
    for src, read, pop, push, dst in raw:
        tops = [pop] if pop else sorted(stack_symbols)
        for top in tops:
            key = (src, read or EPS, top)
            out = (dst, push + ((top,) if not pop else ()))
            if delta.get(key, out) != out:
                raise FormatError(f"nondeterministic: two moves for {key}")
            delta[key] = out
    return DPDA(delta, start=a.initial, start_stack=start_stack, accept_states=frozenset(a.finals))


def _jflap_document(kind: str, states: list[State], initial: State, finals: set[State], transitions: list[str]) -> str:
    tag = "block" if kind == "turing" else "state"
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?><!--Created with relu-tm.--><structure>',
        f"\t<type>{kind}</type>",
        "\t<automaton>",
        "\t\t<!--The list of states.-->",
    ]
    radius = 60 + 25 * len(states)
    for i, s in enumerate(states):
        angle = 2 * math.pi * i / max(len(states), 1)
        x, y = 100 + radius * (1 + math.cos(angle)), 100 + radius * (1 + math.sin(angle))
        lines.append(f'\t\t<{tag} id="{i}" name="{escape(s, {chr(34): "&quot;"})}">')
        if kind == "turing":
            lines.append("\t\t\t<tag>Machine0</tag>")
        lines.append(f"\t\t\t<x>{x:.1f}</x>")
        lines.append(f"\t\t\t<y>{y:.1f}</y>")
        if s == initial:
            lines.append("\t\t\t<initial/>")
        if s in finals:
            lines.append("\t\t\t<final/>")
        lines.append(f"\t\t</{tag}>")
    lines.append("\t\t<!--The list of transitions.-->")
    lines += transitions
    if kind == "turing":
        lines += ["\t\t<!--The list of automata-->", "\t\t<Machine0/>"]
    lines += ["\t</automaton>", "</structure>", ""]
    return "\n".join(lines)


def _elem(tag: str, value: str) -> str:
    return f"<{tag}>{escape(value)}</{tag}>" if value else f"<{tag}/>"


def _single_char(sym: Symbol, what: str) -> None:
    if len(sym) != 1 or sym in ("~", "!", "}", JFLAP_BLANK):
        raise FormatError(f"{what} {sym!r}: JFLAP needs single characters other than ~ ! }} {JFLAP_BLANK}")


def to_jflap(machine: AnyMachine) -> str:
    """Write a JFLAP 7 file (``turing``, ``fa`` or ``pda``)."""
    if isinstance(machine, TuringMachine):
        return _tm_to_jflap(machine)
    if isinstance(machine, DFA):
        states = list(machine.states)
        idx = {s: i for i, s in enumerate(states)}
        trans = []
        for (q, sym), p in machine.transitions.items():
            _single_char(sym, "input symbol")
            trans.append(f"\t\t<transition><from>{idx[q]}</from><to>{idx[p]}</to>{_elem('read', sym)}</transition>")
        return _jflap_document("fa", states, machine.start, set(machine.accept_states), trans)
    if isinstance(machine, DPDA):
        if machine.start_stack != "Z":
            raise FormatError("JFLAP's start stack symbol is Z")
        states = list(machine.states)
        idx = {s: i for i, s in enumerate(states)}
        trans = []
        for (q, sym, top), (p, push) in machine.transitions.items():
            for x in (sym, top, *push):
                if x != EPS:
                    _single_char(x, "symbol")
            trans.append(
                f"\t\t<transition><from>{idx[q]}</from><to>{idx[p]}</to>"
                f"{_elem('read', sym)}{_elem('pop', top)}{_elem('push', ''.join(push))}</transition>"
            )
        return _jflap_document("pda", states, machine.start, set(machine.accept_states), trans)
    raise TypeError(f"cannot export {type(machine).__name__}")


def _tm_to_jflap(tm: TuringMachine) -> str:
    for q in tm.accept_states:
        if q not in tm.halt_states and any(s == q for (s, _a) in tm.transitions):
            raise FormatError(f"accept state {q!r} has outgoing transitions; JFLAP stops in final states")
    for sym in tm.symbols:
        if sym != tm.blank:
            _single_char(sym, "tape symbol")
    states = list(tm.states)
    idx = {s: i for i, s in enumerate(states)}

    def jsym(s: Symbol) -> str:
        return "" if s == tm.blank else s

    trans = []
    for q in states:
        for a in tm.symbols:
            act = tm.action(q, a)
            if act is None:
                continue
            p, b, m = act
            trans.append(
                f"\t\t<transition><from>{idx[q]}</from><to>{idx[p]}</to>"
                f"{_elem('read', jsym(a))}{_elem('write', jsym(b))}<move>{m}</move></transition>"
            )
    return _jflap_document("turing", states, tm.start, set(tm.accept_states), trans)


# ---------------------------------------------------------------------------
# turingmachine.io (YAML)
# ---------------------------------------------------------------------------

ACCEPT_NAMES = ("accept", "accepted", "yes")


def _yaml_loader():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only without PyYAML
        raise ImportError(
            "reading turingmachine.io files needs PyYAML: pip install pyyaml (or relu-tm[formats])"
        ) from exc

    class Loader(yaml.SafeLoader):
        """SafeLoader with js-yaml behaviour: list keys allowed, only
        true/false are booleans, duplicate keys are errors."""

    Loader.yaml_implicit_resolvers = {
        first: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:bool"]
        for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    Loader.add_implicit_resolver(
        "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
    )

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        out = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            if isinstance(key, list):
                key = tuple(key)
            if key in out:
                raise FormatError(f"duplicate key {key!r}")
            out[key] = loader.construct_object(value_node, deep=True)
        return out

    Loader.add_constructor("tag:yaml.org,2002:map", construct_mapping)
    return yaml, Loader


def _js_str(value: Any) -> str:
    """``String(value)`` as in JavaScript (what turingmachine.io does)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (list, tuple)):
        return ",".join("" if v is None else _js_str(v) for v in value)
    return str(value)


def _split_symbols(key: str) -> list[Symbol]:
    """turingmachine.io's comma rule: ``'0,1'`` -> 0, 1 and ``',,'`` -> ','."""
    acc: list[str] = []
    for part in key.split(","):
        if part == "" and acc and acc[-1] == "":
            acc[-1] = ","
        else:
            acc.append(part)
    return acc


def from_turingmachine_io(text: str, accept_names: Iterable[str] = ACCEPT_NAMES) -> MachineFile:
    """Translate a turingmachine.io machine (the bare YAML or a saved
    document with ``source code:``), including ``synonyms``, list keys
    (``[0,1]: R``) and comma keys.

    Halting states (empty entries in the table) whose name is in
    ``accept_names`` become accept states; turingmachine.io itself has no
    notion of acceptance.  The ``input`` stored in the file is returned in
    :attr:`MachineFile.input`.
    """
    yaml, loader = _yaml_loader()
    try:
        doc = yaml.load(text, Loader=loader)
        name = ""
        if isinstance(doc, dict) and "source code" in doc:
            name = _js_str(doc.get("name", "")) if doc.get("name") is not None else ""
            doc = yaml.load(doc["source code"], Loader=loader)
    except yaml.YAMLError as exc:
        raise FormatError(f"YAML error: {exc}") from exc
    if not isinstance(doc, dict):
        raise FormatError("expected a mapping with blank, start state and table")
    if doc.get("blank") is None:
        raise FormatError("no blank symbol was specified")
    blank = _js_str(doc["blank"])
    if len(blank) != 1:
        raise FormatError("the blank symbol must be a string of length 1")
    if doc.get("start state") is None:
        raise FormatError("no start state was specified")
    start = _js_str(doc["start state"])
    table = doc.get("table")
    if not isinstance(table, dict):
        raise FormatError("missing or invalid transition table")
    table = {_js_str(k): v for k, v in table.items()}
    if start not in table:
        raise FormatError("the start state has to be declared in the transition table")

    def instruction(value: Any, synonyms: dict | None) -> tuple[Symbol | None, str, State | None]:
        if isinstance(value, str):
            if value in ("L", "R"):
                return None, value, None
            if synonyms and value in synonyms:
                return synonyms[value]
            raise FormatError(f"unrecognized instruction {value!r}")
        if not isinstance(value, dict) or not value:
            raise FormatError(f"invalid instruction {value!r}")
        bad = set(value) - {"L", "R", "write"}
        if bad:
            raise FormatError(f"unrecognized key {sorted(map(str, bad))[0]!r} in instruction")
        if "L" in value and "R" in value:
            raise FormatError("conflicting tape movements")
        if "L" not in value and "R" not in value:
            raise FormatError("missing movement direction")
        move = "L" if "L" in value else "R"
        target = None if value[move] is None else _js_str(value[move])
        if target is not None and target not in table:
            raise FormatError(f"undeclared state {target!r}")
        write = None
        if "write" in value:
            write = _js_str(value["write"])
            if len(write) != 1:
                raise FormatError("write requires a string of length 1")
        return write, move, target

    synonyms = None
    if doc.get("synonyms") is not None:
        if not isinstance(doc["synonyms"], dict):
            raise FormatError("synonyms must be a mapping")
        synonyms = {_js_str(k): instruction(v, None) for k, v in doc["synonyms"].items()}

    delta: dict[tuple[State, Symbol], Action] = {}
    halting: list[State] = []
    symbols = [blank]
    for state, entries in table.items():
        if not entries:
            halting.append(state)
            continue
        if not isinstance(entries, dict):
            raise FormatError(f"state {state!r}: entries must map symbols to instructions")
        for key, value in entries.items():
            write, move, target = instruction(value, synonyms)
            for sym in _split_symbols(_js_str(key)):  # a later entry wins, as in turingmachine.io
                delta[(state, sym)] = (target if target is not None else state, sym if write is None else write, move)
                symbols += [sym, *([write] if write is not None else [])]
    word = tuple(_js_str(doc["input"])) if doc.get("input") else ()  # JS: falsy input = empty
    symbols += list(word)
    accept = {s for s in halting if s.lower() in {n.lower() for n in accept_names}}
    tm = TuringMachine(
        delta,
        start=start,
        blank=blank,
        halt_states=frozenset(halting),
        accept_states=frozenset(accept),
        states=tuple(table),
        symbols=tuple(dict.fromkeys(symbols)),
        name=name,
    )
    return MachineFile(tm, word, "turingmachine.io")


def _yaml_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def to_turingmachine_io(tm: TuringMachine, input: Iterable[Symbol] = ()) -> str:
    """Write a machine for turingmachine.io (single-character symbols, L/R)."""
    for sym in tm.symbols:
        if len(sym) != 1:
            raise FormatError(f"symbol {sym!r}: turingmachine.io needs single characters")
    lines = [f"# {tm.name}" if tm.name else "# exported by relu-tm"]
    word = "".join(input)
    if word:
        lines.append(f"input: {_yaml_quote(word)}")
    lines += [f"blank: {_yaml_quote(tm.blank)}", f"start state: {_yaml_quote(tm.start)}", "table:"]
    for q in tm.states:
        entries = []
        for a in tm.symbols:
            act = tm.action(q, a)
            if act is None:
                continue
            p, b, m = act
            if m == "S":
                raise FormatError(f"stay move in ({q}, {a}): turingmachine.io only has L and R")
            entries.append(f"    {_yaml_quote(a)}: {{write: {_yaml_quote(b)}, {m}: {_yaml_quote(p)}}}")
        lines.append(f"  {_yaml_quote(q)}:")
        lines += entries
    return "\n".join(lines) + "\n"
