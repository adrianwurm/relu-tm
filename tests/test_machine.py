import pytest
from hypothesis import given
from hypothesis import strategies as st

from relu_tm import Configuration, TuringMachine, accepts
from relu_tm.examples import binary_adder, binary_counter, busy_beaver, palindrome


@pytest.mark.parametrize(("n", "steps", "ones"), [(2, 6, 4), (3, 21, 5), (4, 107, 13)])
def test_busy_beavers(n, steps, ones):
    """Known-answer test for the emulator: the step and ones counts of the
    busy beaver champions are proven values from the literature, so they
    check the oracle itself against an independent source."""
    tm = busy_beaver(n)
    configs = tm.run()
    assert tm.is_halted(configs[-1])
    assert len(configs) - 1 == steps
    assert tm.ones(configs[-1]) == ones


def test_standard_format_parsing():
    tm = TuringMachine.from_standard_format("1RB---_1LA1RZ")
    assert tm.states == ("A", "B", "Z")
    assert tm.symbols == ("0", "1")
    assert tm.halt_states == {"Z"}
    assert tm.action("A", "1") is None  # "---" is undefined = halt
    assert tm.action("B", "1") == ("Z", "1", "R")


@pytest.mark.parametrize("bad", ["", "1RB1L", "1RB1XB_1LA1RZ", "2RB1LB_1LA1RZ"])
def test_standard_format_rejects_garbage(bad):
    with pytest.raises(ValueError):
        TuringMachine.from_standard_format(bad)


def test_invalid_move_rejected():
    with pytest.raises(ValueError):
        TuringMachine({("a", "0"): ("a", "1", "X")}, start="a")


def test_configuration_normal_form():
    a = Configuration("q", ("1", "0", "0"), ("0",), blank="0")
    b = Configuration("q", ("1",), (), blank="0")
    assert a == b
    assert b.head_symbol == "0"
    assert b.render() == "q: 1 [0]"


def test_moving_left_off_the_written_tape():
    tm = TuringMachine({("s", "1"): ("t", "1", "L")}, start="s", symbols=("0", "1"))
    c = tm.step(tm.initial_configuration("1"))
    assert c == Configuration("t", (), ("0", "1"), blank="0")
    assert c.head_symbol == "0"


def test_halted_configuration_is_a_fixed_point():
    tm = busy_beaver(2)
    final = tm.run()[-1]
    assert tm.step(final) == final


def test_binary_counter_counts():
    tm = binary_counter()
    config = tm.initial_configuration("0")
    count = 0
    for _ in range(600):
        nxt = tm.step(config)
        if config.state == "carry" and nxt.state == "right":
            count += 1
            # the tape now holds the binary representation of ``count``
            _, cells = nxt.cells()
            assert "".join(c for c in cells if c != "_").lstrip("0") == format(count, "b")
        config = nxt
    assert count > 20


@given(st.integers(0, 40), st.integers(0, 40))
def test_binary_adder(x, y):
    tm = binary_adder()
    final = tm.run(f"{x:b}+{y:b}")[-1]
    assert final.state == "done"
    _, cells = final.cells()
    assert int("".join(cells).strip("_"), 2) == x + y


@given(st.text(alphabet="ab", max_size=8))
def test_palindrome(word):
    assert accepts(palindrome(), word) == (word == word[::-1])


def test_unknown_input_symbol():
    with pytest.raises(ValueError):
        busy_beaver(2).initial_configuration("2")
