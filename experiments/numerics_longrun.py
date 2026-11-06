"""Numerics: do rounding errors accumulate over long runs?

The binary counter makes many steps in very little space without repeating
itself (20 000 steps on at most 13 cells).  Bases 6 (rounded weights) and 8
(dyadic), float32/float64, ramps of the paper and centred.

Run from the repository root:  python experiments/numerics_longrun.py   (~3 min)
Writes experiments/results/numerics_longrun.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

from relu_tm.examples import binary_counter
from relu_tm.numerics import precision_profile

OUT = Path(__file__).parent / "results"
STEPS = 20_000
CASES = ((6, "edge"), (6, "centre"), (8, "edge"))


def main() -> int:
    rows = []
    for base, ramps in CASES:
        for dtype in ("float32", "float64"):
            prof = precision_profile(binary_counter(), "0", STEPS, dtype, base=base, ramps=ramps)
            fail = prof.first_failure
            before = [r for r in prof.records if fail is None or r.step < fail.step]
            checkpoints = {s: max(r.error for r in before if r.step <= s) for s in (100, 1000, 10_000, STEPS)}
            rows.append(
                {
                    "base": base,
                    "ramps": ramps,
                    "dtype": dtype,
                    "steps": STEPS,
                    "first_failure": fail.step if fail else "",
                    "tape_at_failure": fail.digits if fail else "",
                    "max_tape": max(r.digits for r in prof.records),
                    **{f"max_error_until_{s}": f"{e:.2e}" for s, e in checkpoints.items()},
                }
            )
            print(rows[-1])
    OUT.mkdir(exist_ok=True)
    with open(OUT / "numerics_longrun.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
