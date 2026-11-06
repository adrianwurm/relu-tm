"""Import/export: JFLAP, turingmachine.io (YAML) and the bbchallenge format."""

from itertools import product
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from relu_tm import DFA, DPDA, Configuration, TuringMachine, accepts, run
from relu_tm.cli import main
from relu_tm.examples import (
    BB2,
    BB3,
    BB4,
    anbn_dpda,
    binary_counter,
    busy_beaver,
    even_ones_dfa,
    mirror_dpda,
    palindrome,
)
from relu_tm.formats import (
    FormatError,
    dumps,
    from_jflap,
    from_turingmachine_io,
    load,
    loads,
    to_jflap,
    to_turingmachine_io,
)
from strategies import turing_machines, words

FIXTURES = Path(__file__).parent / "fixtures"


def all_words(alphabet, max_len):
    for n in range(max_len + 1):
        yield from ("".join(w) for w in product(alphabet, repeat=n))


def same_behaviour(tm1, tm2, word, steps=40):
    return tm1.trace(word, steps) == tm2.trace(word, steps)


def lr_only(tm):
    """Restrict a random machine to L/R moves (needed by two of the formats)."""
    t = {k: (p, b, "R" if m == "S" else m) for k, (p, b, m) in tm.transitions.items()}
    return TuringMachine(t, tm.start, tm.blank, tm.halt_states, tm.accept_states, tm.states, tm.symbols)


# -- turingmachine.io ------------------------------------------------------------


def test_yaml_document_with_list_and_comma_keys():
    mf = load(FIXTURES / "increment_document.yaml")
    tm = mf.machine
    assert mf.format == "turingmachine.io" and tm.name == "increment"
    assert mf.input == tuple("0111")
    assert tm.action("scan", "0") == ("scan", "0", "R")  # [0, 1]: R
    assert tm.action("add", " ") == ("stop", "1", "R")  # '0, ' = '0' and ' '
    assert tm.action("add", "1") == ("add", "0", "L")  # {write: 0, L}
    assert tm.halt_states == {"stop"}
    final = tm.run(mf.input)[-1]
    assert "".join(final.cells()[1]).strip() == "1000"


def test_yaml_synonyms_and_yaml12_booleans():
    mf = load(FIXTURES / "palindrome.yaml")
    tm = mf.machine
    assert tm.action("first", " ") == ("accept", " ", "R")  # synonym 'yes'
    assert tm.accept_states == {"accept"} and tm.halt_states == {"accept", "reject"}
    for w in all_words("ab", 6):
        assert accepts(tm, w) == (w == w[::-1]), w
    assert accepts(tm, mf.input)


def test_yaml_machine_runs_on_the_network():
    mf = load(FIXTURES / "palindrome.yaml")
    assert run(mf.machine, mf.input).agree


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("start state: a\ntable: {a: }", "blank"),
        ("blank: '  '\nstart state: a\ntable: {a: }", "length 1"),
        ("blank: 0\ntable: {a: }", "start state"),
        ("blank: 0\nstart state: b\ntable: {a: }", "declared"),
        ("blank: 0\nstart state: a\ntable: {a: {0: {R: b}}}", "undeclared state"),
        ("blank: 0\nstart state: a\ntable: {a: {0: {L: a, R: a}}}", "conflicting"),
        ("blank: 0\nstart state: a\ntable: {a: {0: {write: 1}}}", "direction"),
        ("blank: 0\nstart state: a\ntable: {a: {0: {move: L}}}", "unrecognized key"),
        ("blank: 0\nstart state: a\ntable: {a: {0: {write: 12, L}}}", "length 1"),
        ("blank: 0\nstart state: a\ntable: {a: {0: jump}}", "unrecognized instruction"),
        ("blank: 0\nstart state: a\ntable:\n  a: {0: L}\n  a: {1: R}", "duplicate"),
        ("blank: [0\n", "YAML error"),
    ],
)
def test_yaml_errors(source, message):
    with pytest.raises(FormatError, match=message):
        from_turingmachine_io(source)


def test_yaml_falsy_input_is_empty_as_in_javascript():
    assert from_turingmachine_io("input: 0\nblank: 0\nstart state: a\ntable: {a: }").input == ()


@given(turing_machines(max_states=4, max_symbols=4).map(lr_only), st.data())
def test_yaml_round_trip(tm, data):
    word = data.draw(words(tm))
    mf = from_turingmachine_io(to_turingmachine_io(tm, word))  # the format has no alphabet, only an input
    assert "".join(mf.input) == "".join(word)
    assert same_behaviour(tm, mf.machine, word)


def test_yaml_export_rejects_stay_moves():
    tm = TuringMachine({("a", "0"): ("a", "1", "S")}, start="a")
    with pytest.raises(FormatError, match="stay"):
        to_turingmachine_io(tm)


# -- JFLAP ---------------------------------------------------------------------


def _tm(transitions):
    states = "<block id='0' name='a'><initial/></block><block id='1' name='b'/>"
    body = "".join(
        f"<transition><from>{f}</from><to>{t}</to><read>{r}</read><write>{w}</write><move>R</move></transition>"
        for f, t, r, w in transitions
    )
    return f"<structure><type>turing</type><automaton>{states}{body}</automaton></structure>"


def test_jflap_turing_machine_with_wildcards():
    tm = load(FIXTURES / "unary_append.jff").machine
    assert tm.name == "unary_append" and tm.blank == "_"
    assert tm.action("q0", "x") == ("q0", "1", "S")  # stay move
    assert tm.action("q1", "1") == ("q1", "1", "L")  # read '!' (not blank), write '~' (keep)
    assert tm.action("q1", "x") == ("q1", "x", "L")
    assert tm.action("q1", "_") == ("done", "_", "R")  # empty <read/> = blank
    assert tm.accept_states == tm.halt_states == {"done"}  # JFLAP stops in final states
    final = tm.run("11x1")[-1]
    assert accepts(tm, "11x1") and "".join(final.cells()[1]).strip("_") == "11111"
    assert run(tm, "11x1").agree


def test_jflap_negation_and_keep_wildcards():
    tm = from_jflap(_tm([(0, 1, "x", "~"), (0, 0, "!x", "y")]))
    assert tm.action("a", "x") == ("b", "x", "R")  # write ~ keeps the symbol
    assert tm.action("a", "y") == ("a", "y", "R")  # !x matches y ...
    assert tm.action("a", "_") == ("a", "y", "R")  # ... and the blank


def test_jflap_tilde_read_matches_every_symbol():
    tm = from_jflap(_tm([(0, 1, "~", "z"), (1, 1, "q", "q")]))
    assert {a for (s, a) in tm.transitions if s == "a"} == set(tm.symbols) == {"_", "q", "z"}


def test_jflap_dfa():
    dfa = load(FIXTURES / "even_ones.jff").machine
    assert isinstance(dfa, DFA)
    for w in all_words("01", 7):
        assert accepts(dfa, w) == (w.count("1") % 2 == 0)


def test_jflap_dpda_with_lambda_pop():
    pda = load(FIXTURES / "anbn.jff").machine
    assert isinstance(pda, DPDA)
    assert pda.transitions[("push", "a", "Z")] == ("push", ("a", "Z"))  # λ-pop expanded per top
    for w in all_words("ab", 6):
        n = len(w) // 2
        assert accepts(pda, w) == (n > 0 and w == "a" * n + "b" * n), w
    assert run(load(FIXTURES / "anbn.jff").turing_machine(), "aabb").agree


BAD_JFLAP = {
    "<structure><type>turing</type><tapes>2</tapes><automaton/></structure>": "multi-tape",
    "<structure><type>turing</type><automaton><block id='0' name='a'/></automaton></structure>": "initial",
    "<structure><type>mealy</type><automaton><state id='0'><initial/></state></automaton></structure>": "mealy",
    "<notjflap/>": "structure",
    "<structure><type>fa": "XML",
}


@pytest.mark.parametrize(("xml", "message"), BAD_JFLAP.items())
def test_jflap_errors(xml, message):
    with pytest.raises(FormatError, match=message):
        from_jflap(xml)


def test_jflap_rejects_nondeterminism_and_unsupported_features():
    with pytest.raises(FormatError, match="nondeterministic"):
        from_jflap(_tm([(0, 1, "x", "x"), (0, 0, "~", "x")]))
    with pytest.raises(FormatError, match="binding"):
        from_jflap(_tm([(0, 1, "a,b}w", "x")]))
    with pytest.raises(FormatError, match="unknown state"):
        from_jflap(_tm([(0, 7, "x", "x")]))
    block = (
        "<structure><type>turing</type><automaton><block id='0' name='a'><tag>Machine1</tag><initial/></block>"
        "<Machine1><structure><type>turing</type></structure></Machine1></automaton></structure>"
    )
    with pytest.raises(FormatError, match="building block"):
        from_jflap(block)
    nfa = (
        "<structure><type>fa</type><automaton><state id='0' name='a'><initial/></state>"
        "<transition><from>0</from><to>0</to><read/></transition></automaton></structure>"
    )
    with pytest.raises(FormatError, match="NFA"):
        from_jflap(nfa)


@given(turing_machines(max_states=4, max_symbols=4), st.data())
def test_jflap_round_trip_turing_machines(tm, data):
    # JFLAP stops in final states, so only halting states may accept
    tm = TuringMachine(tm.transitions, tm.start, tm.blank, tm.halt_states, tm.halt_states, tm.states, tm.symbols)
    back = from_jflap(to_jflap(tm))
    word = data.draw(words(back))  # JFLAP files have no alphabet: only symbols used in transitions survive
    assert same_behaviour(tm, back, word)
    assert back.accept_states == tm.accept_states


@pytest.mark.parametrize(
    ("machine", "alphabet"), [(even_ones_dfa(), "01"), (anbn_dpda(), "ab"), (mirror_dpda(), "abc")]
)
def test_jflap_round_trip_automata(machine, alphabet):
    back = from_jflap(to_jflap(machine))
    assert type(back) is type(machine)
    for w in all_words(alphabet, 5):
        assert accepts(back, w) == accepts(machine, w), w


def test_jflap_export_checks():
    with pytest.raises(FormatError, match="single characters"):
        to_jflap(mirror_dpda().to_turing_machine())  # tape symbols like "[Z]"
    not_halting = TuringMachine({("a", "0"): ("a", "1", "R")}, start="a", accept_states={"a"})
    with pytest.raises(FormatError, match="final"):
        to_jflap(not_halting)


# -- bbchallenge ---------------------------------------------------------------


@pytest.mark.parametrize(("n", "text"), [(2, BB2), (3, BB3), (4, BB4)])
def test_standard_format_round_trip_busy_beavers(n, text):
    assert busy_beaver(n).to_standard_format() == text


@given(turing_machines(max_states=4, max_symbols=4).map(lr_only), st.data())
def test_standard_format_round_trip_random(tm, data):
    """Same behaviour up to renaming (symbols -> 0,1,..., states -> A,B,..., halting -> Z)."""
    back = TuringMachine.from_standard_format(tm.to_standard_format())
    sym = {a: str(i) for i, a in enumerate(tm.symbols)}
    working = [q for q in tm.states if q not in tm.halt_states]
    state = {q: chr(ord("A") + i) for i, q in enumerate(working)} | {q: "Z" for q in tm.halt_states}

    def renamed(c):
        return Configuration(state[c.state], tuple(sym[x] for x in c.left), tuple(sym[x] for x in c.right), "0")

    word = data.draw(words(tm))
    assert [renamed(c) for c in tm.trace(word, 30)] == back.trace(tuple(sym[x] for x in word), 30)


def test_standard_format_export_errors():
    with pytest.raises(ValueError, match="stay"):
        TuringMachine({("a", "0"): ("a", "1", "S")}, start="a").to_standard_format()
    with pytest.raises(ValueError, match="10 symbols"):
        TuringMachine({("a", "0"): ("a", "x", "R")}, start="a", symbols=tuple("0123456789x")).to_standard_format()


def test_loads_bbchallenge_text_with_comments():
    mf = loads("# BB(2)\n1RB1LB_1LA1RZ\n", "bbchallenge")
    assert mf.machine.halting_time() == 6


def test_load_needs_a_known_suffix(tmp_path):
    path = tmp_path / "machine.xyz"
    path.write_text("1RB1LB_1LA1RZ")
    with pytest.raises(FormatError, match="format"):
        load(path)
    assert load(path, format="bbchallenge").machine.halting_time() == 6


def test_dumps_dispatch():
    assert dumps(busy_beaver(2), "bbchallenge") == "1RB1LB_1LA1RZ\n"
    assert "<type>turing</type>" in dumps(binary_counter(), "jflap")
    assert "input: '0'" in dumps(binary_counter(), "turingmachine.io", "0")
    with pytest.raises(FormatError, match="stay"):  # palindrome halts with S moves
        dumps(palindrome(), "turingmachine.io")
    with pytest.raises(FormatError):
        dumps(even_ones_dfa(), "bbchallenge")


# -- command line ----------------------------------------------------------------


def test_cli_runs_files(capsys):
    assert main(["run", str(FIXTURES / "palindrome.yaml"), "--every", "10"]) == 0  # input from the file
    assert "agree" in capsys.readouterr().out
    assert main(["run", str(FIXTURES / "even_ones.jff"), "-i", "1001"]) == 0  # DFA -> TM -> network
    assert main(["info", str(FIXTURES / "unary_append.jff")]) == 0


def test_cli_convert(tmp_path, capsys):
    out = tmp_path / "bb3.yaml"
    assert main(["convert", "bb3", "--to", "turingmachine.io", "-o", str(out)]) == 0
    assert load(out).machine.halting_time() == 21
    assert main(["convert", str(FIXTURES / "anbn.jff"), "--to", "jflap"]) == 0
    assert "<type>pda</type>" in capsys.readouterr().out
    assert main(["convert", str(FIXTURES / "anbn.jff"), "--to", "turingmachine.io"]) == 2  # "[Z]" is not 1 char
