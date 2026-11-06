"""Numerics study, step 2: realistic machines.

A. Does the failure depend only on the register length?  BB(4), the binary
   counter, the adder and the palindrome decider with growing inputs, plus
   300 random machines, in float32 and float64 (power-of-two bases, the
   paper's ramps).  For every run: the longest register before the first
   failure, the length at the failure, and the layer/pass that rounds first.
B. Which pass amplifies errors in other bases?  The zigzag in base 6 and 10,
   ramps "edge" versus "centre": error growth factor of each pass.

Run from the repository root:  python experiments/numerics_step2.py
Writes experiments/results/numerics_step2_runs.csv and ..._amplification.csv.
"""

from __future__ import annotations

import csv
import itertools
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from relu_tm import TuringMachine, build_network, encode
from relu_tm.examples import binary_adder, binary_counter, busy_beaver, palindrome, zigzag
from relu_tm.numerics import first_rounding, pass_errors, precision_profile, predicted_capacity, read

OUT = Path(__file__).parent / "results"
DTYPES = ("float32", "float64")
SEED = 2026


def random_machine(rng: random.Random) -> TuringMachine:
    n, k = rng.randint(2, 4), rng.randint(2, 3)
    states, symbols = [f"q{i}" for i in range(n)], ["_", *"ab"[: k - 1]]
    t = {}
    for s in states:
        for a in symbols:
            if rng.random() < 0.08:
                continue  # undefined -> halt
            t[(s, a)] = (rng.choice([*states, "H"]), rng.choice(symbols), rng.choice("LRS"))
    return TuringMachine(t, "q0", "_", frozenset({"H"}), states=(*states, "H"), symbols=tuple(symbols), name="random")


def workloads(rng: random.Random):
    yield "BB(4)", 0, busy_beaver(4), (), 107
    for n in range(2, 31, 2):
        yield "counter", n, binary_counter(), "1" * n, 4 * n + 10
    for n in range(2, 31, 2):
        x = "1" + "".join(rng.choice("01") for _ in range(n - 1))
        tm = binary_adder()
        yield "adder", n, tm, f"{x}+11", tm.halting_time(f"{x}+11")
    for n in range(2, 31, 2):
        half = "".join(rng.choice("ab") for _ in range(n // 2))
        tm = palindrome()
        yield "palindrome", n, tm, half + half[::-1], tm.halting_time(half + half[::-1])
    for i in range(300):
        tm = random_machine(rng)
        word = tuple(rng.choice(tm.symbols) for _ in range(rng.randint(0, 24)))
        yield "random", i, tm, word, 150


def note_of(net, fr) -> str:
    if fr is None:
        return ""
    if fr.layer == 0:
        return "input: the start vector is already rounded (tape too long)"
    where = net.program.layers[fr.layer - 1].note if fr.layer <= net.depth else "read-out"
    kind = "work neuron" if fr.neuron == "w" else f"register {fr.neuron}"
    return f"{where}  [{kind}]"


def part_a(writer_rows: list[dict]) -> None:
    rng = random.Random(SEED)
    for name, param, tm, word, steps in workloads(rng):
        net = build_network(tm)
        for dtype in DTYPES:
            prof = precision_profile(tm, word, steps, dtype, network=net)
            inexact, fail = prof.first_inexact, prof.first_failure
            fr = None
            if inexact is not None:
                start = net.iterate_exact(encode(tm.initial_configuration(word), net.layout), inexact.step - 1)[-1]
                fr = first_rounding(net, start, dtype)
            writer_rows.append(
                {
                    "workload": name,
                    "param": param,
                    "base": net.layout.base,
                    "dtype": dtype,
                    "steps": steps,
                    "bound": predicted_capacity(net.layout.base, dtype),
                    "max_len": max(r.digits for r in prof.records),
                    "max_correct_len": prof.max_correct_digits,
                    "first_inexact": inexact.step if inexact else "",
                    "first_failure": fail.step if fail else "",
                    "failure_status": fail.status if fail else "",
                    "len_at_failure": fail.digits if fail else "",
                    "first_rounding_pass": fr.pass_name if fr else "",
                    "first_rounding_note": note_of(net, fr),
                }
            )


def part_b(rows: list[dict]) -> None:
    for base in (6, 10):
        for ramps in ("edge", "centre"):
            net = build_network(zigzag(), base, ramps)
            x0 = encode(zigzag().initial_configuration(), net.layout)
            steps = 300
            exact = net.iterate_exact(x0, steps)
            floats = net.iterate_torch(x0, steps, torch.float64)
            logs: dict[str, list[float]] = defaultdict(list)
            for t in range(steps):
                if read(floats[t], net.layout) != read(exact[t], net.layout):
                    break  # stop at the first failure
                errs = pass_errors(net, exact[t], floats[t], "float64")
                order = ["input", *[p for p in ("read", "table", "scale", "left", "state", "output") if p in errs]]
                for prev, cur in itertools.pairwise(order):
                    if errs[prev] > 0 and errs[cur] > 0:
                        logs[cur].append(math.log2(errs[cur] / errs[prev]))
                if errs["input"] > 0 and errs["output"] > 0:
                    logs["whole step"].append(math.log2(errs["output"] / errs["input"]))
            for p, vals in logs.items():
                rows.append(
                    {
                        "base": base,
                        "ramps": ramps,
                        "pass": p,
                        "steps_used": len(vals),
                        "mean_factor": 2 ** (sum(vals) / len(vals)),
                        "max_factor": 2 ** max(vals),
                    }
                )


def write(name: str, rows: list[dict]) -> None:
    with open(OUT / f"numerics_step2_{name}.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report(runs: list[dict], amp: list[dict]) -> None:
    print("## A. Failure versus register length\n")
    print("| base | dtype | runs | failed | failed although max length <= bound | length at failure - bound | abrupt |")
    print("|---|---|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in runs:
        groups[(r["base"], r["dtype"])].append(r)
    for (base, dtype), rs in sorted(groups.items()):
        failed = [r for r in rs if r["first_failure"] != ""]
        early = [r for r in failed if max(r["max_correct_len"], r["len_at_failure"]) <= r["bound"]]
        over = Counter(r["len_at_failure"] - r["bound"] for r in failed)
        abrupt = sum(1 for r in failed if r["first_inexact"] == r["first_failure"])
        over_s = ", ".join(f"+{k}: {v}" for k, v in sorted(over.items())) or "-"
        print(f"| {base} | {dtype} | {len(rs)} | {len(failed)} | {len(early)} | {over_s} | {abrupt}/{len(failed)} |")
    print("\nLongest correct register in runs that went beyond the bound:")
    for (base, dtype), rs in sorted(groups.items()):
        beyond = [r["max_correct_len"] for r in rs if r["max_correct_len"] > r["bound"]]
        top = max(beyond, default="-")
        print(f"  base {base} {dtype}: bound {rs[0]['bound']}, {len(beyond)} runs beyond it, up to {top}")
    print("\nFirst rounding (layer note):")
    for note, c in Counter(r["first_rounding_note"] for r in runs if r["first_rounding_note"]).most_common():
        print(f"  {c:4d}  {note}")
    print("\n## B. Error growth per pass (zigzag, float64)\n")
    print("| base | ramps | pass | mean factor | max factor |")
    print("|---|---|---|---|---|")
    for r in amp:
        print(f"| {r['base']} | {r['ramps']} | {r['pass']} | {r['mean_factor']:.3g} | {r['max_factor']:.3g} |")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    runs: list[dict] = []
    part_a(runs)
    amp: list[dict] = []
    part_b(amp)
    write("runs", runs)
    write("amplification", amp)
    report(runs, amp)
    print(f"\n{len(runs)} runs in {time.time() - t0:.0f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
