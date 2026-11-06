from fractions import Fraction

import pytest

from relu_tm.network import _matvec
from relu_tm.program import Aff, ProgramBuilder


def evaluate(program, x):
    hidden, (W_out, b_out) = program.matrices()
    x = [Fraction(v) for v in x]
    for W, b in hidden:
        x = [max(v, Fraction(0)) for v in _matvec(W, x, b)]
    return _matvec(W_out, x, b_out)


def test_aff_arithmetic():
    l, r = Aff.var("l"), Aff.var("r")
    e = 2 * l - r / 4 + 1
    assert e.coef("l") == 2 and e.coef("r") == Fraction(-1, 4) and e.constant == 1
    assert (e - e) == Aff()
    assert e.substitute({"l": r + 1}) == Fraction(7, 4) * r + 3
    assert e.evaluate({"l": Fraction(1), "r": Fraction(4)}) == 2
    assert repr(Aff()) == "0"
    with pytest.raises(TypeError):
        _ = l * r


def test_builder_semantics():
    """q += ReLU(l - 1/2); then r := 2r + ReLU(1 - ReLU(4q))."""
    l, r, q, w = (Aff.var(v) for v in "lrqw")
    b = ProgramBuilder()
    b.relu(l - Fraction(1, 2))
    b.add(q=w)
    b.affine(r=2 * r)
    b.relu(4 * q)
    b.relu(1 - w)
    b.add(r=w)
    prog = b.finish()
    assert len(prog.layers) == 3

    def reference(lv, rv, qv):
        qv = qv + max(lv - Fraction(1, 2), 0)
        return [lv, 2 * rv + max(1 - max(4 * qv, 0), 0), qv]

    for x in [(0, 0, 0), (Fraction(3, 4), Fraction(1, 8), 0), (1, 1, Fraction(1, 16))]:
        assert evaluate(prog, x) == reference(*map(Fraction, x))


def test_matrix_shapes():
    b = ProgramBuilder()
    b.relu(Aff.var("l"))
    b.relu(Aff.var("w"))
    hidden, (W_out, _) = b.finish().matrices()
    assert [(len(W), len(W[0])) for W, _ in hidden] == [(4, 3), (4, 4)]
    assert (len(W_out), len(W_out[0])) == (3, 4)


def test_affine_rejects_w():
    with pytest.raises(ValueError):
        ProgramBuilder().affine(l=Aff.var("w"))
