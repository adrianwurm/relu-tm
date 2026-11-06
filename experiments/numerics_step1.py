"""Numerics study, step 1: at which register length does floating point break?

Workloads (all write the digit 3 = symbol "1" in base 4):
  filler    writes one cell per step and never reads back,
  zigzag    tape grows by one cell per sweep, every cell is read back,
  pendulum  two cells, many steps.
Grid: bases 4, 6, 8, 10, 12, 16 x float32/float64 x ramps "edge" (paper) / "centre".

Run from the repository root:  python experiments/numerics_step1.py
Writes experiments/results/numerics_step1_summary.csv and ..._trace.csv.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from relu_tm.examples import filler, pendulum, zigzag
from relu_tm.numerics import precision_profile, predicted_capacity

OUT = Path(__file__).parent / "results"
BASES = (4, 6, 8, 10, 12, 16)
DTYPES = ("float32", "float64")
RAMPS = ("edge", "centre")
WORKLOADS = ((filler, 80), (zigzag, 900), (pendulum, 2000))
TRACE_MARGIN = 10  # steps kept in the trace after the first failure


def main() -> int:
    OUT.mkdir(exist_ok=True)
    summary, trace = [], []
    t0 = time.time()
    for make, steps in WORKLOADS:
        tm = make()
        for ramps in RAMPS:
            for base in BASES:
                for dtype in DTYPES:
                    prof = precision_profile(tm, steps=steps, dtype=dtype, base=base, ramps=ramps)
                    inexact, fail = prof.first_inexact, prof.first_failure
                    stop = fail.step if fail else steps + 1
                    summary.append(
                        {
                            "machine": tm.name,
                            "ramps": ramps,
                            "base": base,
                            "dtype": dtype,
                            "steps": steps,
                            "predicted_push": predicted_capacity(base, dtype, digit=3),
                            "predicted_bound": predicted_capacity(base, dtype),
                            "first_inexact": inexact.step if inexact else "",
                            "first_failure": fail.step if fail else "",
                            "failure_status": fail.status if fail else "",
                            "max_correct_digits": prof.max_correct_digits,
                            "max_error_before_failure": max(
                                (r.error for r in prof.records if r.step < stop), default=0
                            ),
                        }
                    )
                    if tm.name != "pendulum":
                        trace += [{"ramps": ramps, **row} for row in prof.rows() if row["step"] <= stop + TRACE_MARGIN]
    for name, rows in (("summary", summary), ("trace", trace)):
        with open(OUT / f"numerics_step1_{name}.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    print("| machine | ramps | base | dtype | predicted | longest correct register | first inexact | failure |")
    print("|---|---|---|---|---|---|---|---|")
    for r in summary:
        pred = r["predicted_push"] if r["predicted_push"] is not None else "-"
        fail = f"step {r['first_failure']} ({r['failure_status']})" if r["first_failure"] != "" else "none"
        first = r["first_inexact"] if r["first_inexact"] != "" else "never"
        row = (r["machine"], r["ramps"], r["base"], r["dtype"], pred, r["max_correct_digits"], first, fail)
        print("| " + " | ".join(str(x) for x in row) + " |")
    print(f"\n{len(summary)} runs in {time.time() - t0:.0f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
