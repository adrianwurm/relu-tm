"""Command line interface.

``relu-tm run bb4``, ``relu-tm run 1RB1LB_1LA1RZ``, ``relu-tm run machine.jff``,
``relu-tm convert palindrome.yaml --to jflap``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .examples import DEFAULT_INPUTS, TURING_MACHINES
from .formats import FORMATS, FormatError, MachineFile, dumps, load
from .machine import TuringMachine
from .network import RAMPS, WIDTH, build_network
from .run import BACKENDS, run

MACHINE_HELP = "example name (see 'relu-tm list'), file (.jff, .yaml, .txt) or bbchallenge string, e.g. 1RB1LB_1LA1RZ"


def load_source(spec: str) -> MachineFile:
    """Resolve an example name, a machine file or a bbchallenge string."""
    if spec in TURING_MACHINES:
        return MachineFile(TURING_MACHINES[spec](), tuple(DEFAULT_INPUTS.get(spec, "")), "example")
    if Path(spec).is_file():
        return load(spec)
    return MachineFile(TuringMachine.from_standard_format(spec), (), "bbchallenge")


def load_machine(spec: str) -> TuringMachine:
    """Like :func:`load_source`, compiled to a Turing machine."""
    return load_source(spec).turing_machine()


def _parse_word(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(text.split()) if " " in text else tuple(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="relu-tm",
        description="Simulate a Turing machine with a fixed width-4 ReLU network and compare "
        "it step by step with a classical emulator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a machine and its network side by side")
    p_run.add_argument("machine", help=MACHINE_HELP)
    p_run.add_argument("-i", "--input", help="input word (characters, or space separated symbols)")
    p_run.add_argument("-n", "--steps", type=int, help="number of steps (default: until halt)")
    p_run.add_argument("-b", "--backend", choices=BACKENDS, default="exact")
    p_run.add_argument("--base", type=int, help="encoding base (default: smallest power of 2 >= 2|Γ|)")
    p_run.add_argument("--every", type=int, default=1, help="print every k-th step")

    p_info = sub.add_parser("info", help="show the constructed network")
    p_info.add_argument("machine", help=MACHINE_HELP)
    p_info.add_argument("--base", type=int)
    p_info.add_argument("--program", action="store_true", help="print the layer-by-layer program")

    p_conv = sub.add_parser("convert", help="translate a machine into another file format")
    p_conv.add_argument("machine", help=MACHINE_HELP)
    p_conv.add_argument("--to", required=True, choices=FORMATS, help="target format")
    p_conv.add_argument("-o", "--output", help="output file (default: print)")

    p_prec = sub.add_parser("precision", help="compare exact and floating-point simulation step by step")
    p_prec.add_argument("machine", help=MACHINE_HELP)
    p_prec.add_argument("-i", "--input", help="input word")
    p_prec.add_argument("-n", "--steps", type=int, default=200, help="number of steps (default 200)")
    p_prec.add_argument("--base", type=int, help="encoding base")
    p_prec.add_argument("--ramps", choices=RAMPS, default="edge", help="digit tests as in the paper or centred")
    p_prec.add_argument("--every", type=int, default=0, help="also print every k-th step")

    sub.add_parser("list", help="list the example machines")

    args = parser.parse_args(argv)

    if args.command == "list":
        for name, factory in TURING_MACHINES.items():
            tm = factory()
            print(f"{name:<11} {tm.name:<16} states={len(tm.states):<3} symbols={len(tm.symbols)}")
        return 0

    try:
        source = load_source(args.machine)
        if args.command == "convert":
            machine = source.machine if args.to == "jflap" else source.turing_machine()
            text = dumps(machine, args.to, source.input)
        else:
            tm = source.turing_machine()
    except (FormatError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "convert":
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"wrote {args.output}")
        else:
            print(text, end="")
        return 0

    if args.command == "precision":
        from .numerics import precision_profile, predicted_capacity

        word = _parse_word(args.input) if args.input is not None else source.input
        for dtype in ("float32", "float64"):
            prof = precision_profile(tm, word, args.steps, dtype, args.base, ramps=args.ramps)
            inexact, fail = prof.first_inexact, prof.first_failure
            bound = predicted_capacity(prof.base, dtype)
            print(f"{dtype} (base {prof.base}, ramps {args.ramps}):")
            print(f"  first inexact step     : {inexact.step if inexact else '-'}")
            if fail:
                print(f"  first failure          : step {fail.step} ({fail.status}, register length {fail.digits})")
            else:
                print(f"  first failure          : none within {args.steps} steps")
            print(f"  longest correct register: {prof.max_correct_digits} digits", end="")
            print(f" (guaranteed exact up to {bound})" if bound is not None else " (base is no power of two)")
            if args.every:
                for rec in prof.records[:: args.every]:
                    print(f"    {rec.step:>6} {rec.status:<8} digits {rec.digits:>3}  error {rec.error:.1e}")
        return 0

    if args.command == "info":
        net = build_network(tm, args.base)
        lay = net.layout
        print(f"machine : {tm.name}")
        print(f"states  : {', '.join(lay.states)}")
        print("symbols : " + ", ".join(f"{s!r}->{lay.digit(s)}" for s in lay.symbols) + "  (tail = 0)")
        print(f"base    : {lay.base}")
        print(f"network : width {net.width}, {net.depth} hidden layers, {net.n_parameters} parameters")
        if args.program:
            print()
            print(net.program.listing())
        return 0

    word = _parse_word(args.input) if args.input is not None else source.input
    if args.steps is None and tm.halting_time(word) is None:
        print("machine does not halt within 10000 steps; pass --steps", file=sys.stderr)
        return 2
    result = run(tm, word, steps=args.steps, backend=args.backend, base=args.base)
    print(result.table(every=args.every))
    net_desc = f"width-{WIDTH} network, {args.backend}"
    if result.agree:
        print(f"\nAll {result.steps} steps agree ({net_desc}).")
        return 0
    print(f"\nFirst disagreement at step {result.first_mismatch} ({net_desc}).")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
