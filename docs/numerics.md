# Numerics: exact rationals versus floating point

In exact rational arithmetic the network simulates every step correctly,
for any tape length. This note measures what happens with the *same*
weights in float32 and float64. It is step 1 of the numerics study and
covers register capacity, the role of the base and the role of the
digit tests.

Reproduce:
```bash
python experiments/numerics_step1.py          # all 72 runs, ~1.5 min; writes experiments/results/*.csv
relu-tm precision zigzag --base 6 -n 300      # a single run
relu-tm precision zigzag --base 6 -n 300 --ramps centre
```

## Method

* The exact network (Python `fractions`) and the float network (PyTorch)
  each run on their own output for `t` steps, starting from the same
  encoding.
* `relu_tm.numerics.read` is a **tolerant decoder**. It reads a register
  digit by digit the way the network does, but assigns each value to the
  nearest admissible case and splits every gap in the middle. A
  perturbation is tolerated as long as, multiplied by β for each digit
  read, it stays below 1/2. That is also what the network would see if
  it read those digits back.
* Every step gets one of four statuses:
  * `exact`: the float registers equal the exact ones;
  * `inexact`: they differ, but the configuration still reads correctly;
  * `wrong`: they read as a *valid but different* configuration, a silent error;
  * `invalid`: they cannot be read at all, a detectable error.
* The workloads all write the symbol `1`. Two of them stress opposite
  extremes, the third is a control:
  * `filler`: **long registers in few steps**. It writes one cell per step
    and never reads back.
  * `zigzag`: **many steps on a short tape, without repeating itself**. The
    tape grows by one cell per sweep, so n cells take about n²/2 steps, and
    every cell is read again.
  * `pendulum`: two cells forever, a periodic control. It shows whether
    the number of steps alone matters.

  The binary counter is even more extreme in the zigzag direction: 2ⁿ
  increments on n cells. It is used for the long runs below.

  Each runs with bases 4, 6, 8, 10, 12 and 16, in float32 and float64, with
  the digit tests of the papers (`ramps="edge"`, the default) and with
  centred tests (`ramps="centre"`, see result 2).

"Longest correct register" below is the largest number of digits that a
register reached before the first failure.

## Result 1: powers of two are bit-exact until the mantissa is full

For β = 2^m all weights are dyadic, so the float network computes
*exactly* the same numbers as the exact one, until a register needs more
significand bits than the format has. The binding operation is the gated
scaling of a push. It forms κ·(x + d) with κ = 1 − 1/β, which carries m
more bits than x. For a register of n digits and highest bit 2^h of
κ(x + d), the product fits iff h + 1 + m(n + 1) ≤ p (p = 24 or 53). This
gives the capacity ⌊(p − 1 − h)/m⌋. The worst case over all digits is
h = m − 1, i.e. ⌊p/m⌋ − 1 (`relu_tm.numerics.predicted_capacity`).

| base | dtype | predicted (digit 3) | filler | zigzag | worst-case bound |
|---|---|---|---|---|---|
| 4 | float32 | 11 | 11 | 11 | 11 |
| 4 | float64 | 25 | 25 | 26 | 25 |
| 8 | float32 | 7 | 7 | 7 | 7 |
| 8 | float64 | 17 | 17 | 17 | 16 |
| 16 | float32 | 5 | 5 | 5 | 5 |
| 16 | float64 | 12 | 12 | 12 | 12 |

* For the filler, the prediction matches the measurement exactly in all
  six cases.
* The zigzag also pushes from the other side, where the magnitudes
  differ. It reaches the same length or one digit more, and never falls
  below the worst-case bound.
* The failure is **abrupt**: the first inexact step is also the first
  failing step.
* Only the **register length** matters, not the number of steps. The
  pendulum (two cells) runs 2000 steps bit-exactly in every power-of-two
  base.

In practice this means: in float64 with base 4, every register may hold
about 25 cells, i.e. about 50 bits of tape on each side of the head.
float32 allows 11 cells.

## Result 2: other bases round from the start, and the paper's digit tests amplify the error

With β = 6, 10 or 12, weights such as 1/6 are rounded already, so the
registers are inexact after the first step (error ≈ 1e-16 in float64).
How far the simulation gets then depends on whether the machine reads
its tape back:

| base | dtype | filler | pendulum | zigzag, edge ramps | zigzag, centred ramps |
|---|---|---|---|---|---|
| 6 | float32 | 9 | 2000 steps ok | 4 | 8 |
| 6 | float64 | 19 | 2000 steps ok | 6 | 20 |
| 10 | float32 | 6 | 2000 steps ok | 2 | 6 |
| 10 | float64 | 15 | 2000 steps ok | 4 | 15 |
| 12 | float32 | 5 | 2000 steps ok | 2 | 6 |
| 12 | float64 | 14 | 2000 steps ok | 4 | 13 |

The values are the longest correct register in digits.

* **Without reading back** (filler, pendulum) the error stays at the
  rounding level. The pendulum's error is at most 5e-15 after 2000 steps.
  The filler's limit, about (p − 1)/log₂β digits, is where the deepest
  digit can no longer be read back.
* **With reading back** (zigzag) the error grows by a factor of 3 to 90
  *per step*. The simulation breaks after 3 to 17 steps, with only 2 to 6
  digits on the tape.

**Cause.** A digit test in the papers is a ramp over the whole gap,
min(1, ReLU(y)). For exact values y ≤ 0 or y ≥ 1, so the test returns
exactly 0 or 1. Some exact values sit right at the edge of a gap,
though. One example is the blank tail (y = 0); another is a digit
followed by a long run of top digits, which is only β^−n from the edge.
A rounding error of 1e-16 then gives an increment of 1e-16 instead of
0. After that:
- the state register is no longer an integer;
- the hat functions of pass 1 mix two rows of the transition table with
  weights 1 − δ and δ;
- the large weights there (jumps of the case code by up to |Q|·(|Γ|+4))
  amplify δ.

**Centred ramps** (`build_network(tm, ramps="centre")`) place each ramp
on the middle half of its gap, min(1, ReLU(2y − 1/2)). Values within 1/4
of an admissible case then still give exactly 0 or 1, and the state
register stays an integer. Depth, width and dyadic weights are
unchanged. With this change the zigzag reaches the filler's limit (base
6, float64: 20 digits instead of 6). In power-of-two bases nothing
changes, because those compute exactly anyway. This is a deviation from
the construction in the papers, so it is an option and not the default.

## Result 3: failures are usually detectable, but not always

Most failures are `invalid`: the registers no longer decode, and a check
after every step would notice. Silent `wrong` configurations do occur,
for example filler in base 8 and base 10 (float64), and zigzag with
centred ramps in base 10. There a rounded deep digit turns into another
admissible digit. A validity check therefore does not catch every error.

## Which base should one choose?

With centred ramps, every base holds about the same number of **bits** per
register: about 50 in float64 and about 21 in float32 (zigzag:
46.6–52 bits and 19.9–22 bits). A base 2|Γ| that is not a power of two
therefore stores slightly more **cells**. For three symbols, base 6 holds
20 cells in float64 against 17 for base 8, because a cell costs
log₂6 ≈ 2.6 bits instead of 3. The price:
- the arithmetic is no longer exact;
- the centred ramps are needed, a deviation from the papers;
- silent errors become possible;
- long runs accumulate rounding errors, even with centred ramps (see
  "Long runs" below).

The power of two therefore stays the default. It is bit-exact, its limit
can be computed in advance, and the failure is sharp. Base 2|Γ| is only
worth it to get the last ~15 % of tape length out of a register.

## Step 2: realistic machines

`experiments/numerics_step2.py`, 692 runs in total:
- BB(4);
- the binary counter, the adder and the palindrome decider, with inputs
  of length 2 to 30;
- 300 random machines with 2–4 states, 2–3 symbols, inputs of length
  0–24 and 150 steps each.

Everything runs in float32 and float64, with the default construction
(power-of-two base, the papers' ramps). Here "length" is the length of
the tape word without trailing blanks. Trailing blank digits may round
away harmlessly, because a blank tail means the same thing.

### Does the failure depend only on the register length? Yes.

| base | dtype | input already too long | grew beyond the bound during the run | of these failed | length at failure − bound |
|---|---|---|---|---|---|
| 4 | float32 | 70 (all fail) | 19 | 19 | +1: 18, +2: 1 |
| 4 | float64 | 0 | 27 | 27 | +1: 13, +2: 14 |
| 8 | float32 | 129 (all fail) | 23 | 20 | +1: 19, +2: 1 |
| 8 | float64 | 62 (all fail) | 32 | 31 | +1: 8, +2: 22, +3: 1 |

* **Not one of the 692 runs failed while every register stayed within the
  worst-case bound** ⌊p/m⌋ − 1. This holds for BB(4), counter, adder,
  palindrome and the random machines alike. What a machine does with its
  tape (direction changes, blanks in the middle, state changes) plays no
  role.
* If a register grows beyond the bound, the run fails at bound +1 or +2
  (once +3), depending on the digits involved, as in step 1.
* Four runs reached bound +1 without failing; their registers never became
  inexact. ⌊p/m⌋ digits still fit into the significand
  (e.g. 8 × 3 = 24 bits in float32), as long as the register is only read
  and not extended.
* 91 of the 97 failures during a run are abrupt: the first inexact step is
  the failing step. In the others, an early rounding was still read
  correctly at first, typically a rounded trailing blank digit.

### Which pass rounds first?

For every run, the step before the first deviation is computed again,
layer by layer, exactly and in float:

| first deviating value | runs |
|---|---|
| input already rounded (the tape was too long from the start) | 261 |
| pass 2, κ·r for moves L/S (work neuron) | 71 |
| pass 2, κ·l for moves R (work neuron) | 25 |
| pass 3, κ·r for moves L (work neuron) | 6 |
| pass 1, register r += d′ + β (push for a move L) | 4 |

In 102 of the 106 cases that start inside the network, the gated scaling
κ·x is the first operation to round. This is exactly the bottleneck from
the capacity formula. The remaining 4 cases come from the push addition
r + d′ + β of left moves, whose magnitude (up to 2β) costs one more bit.
**Possible improvement:** a gate that computes x/β directly, instead of
x − κx, would gain about one digit per register. That would be a change
to the construction; I have not done it.

### Which pass amplifies errors? (base 6 and 10, zigzag, float64)

The table shows the average factor by which a pass enlarges the largest
error in one step, over all steps before the failure:

| base | ramps | read | table | scale | left | state | **whole step** |
|---|---|---|---|---|---|---|---|
| 6 | edge | 1.30 | **4.77** | **2.65** | 0.89 | 0.84 | **11.4** |
| 6 | centre | 0.99 | 1.22 | 0.96 | 1.11 | 0.92 | 1.21 |
| 10 | edge | 1.61 | **7.06** | **5.63** | 0.73 | 0.75 | **43.1** |
| 10 | centre | 1.00 | 1.44 | 0.92 | 0.93 | 0.79 | 1.01 |

The mechanism of result 2 is confirmed:
1. The error *enters* in the read pass. A value just off a ramp edge gives
   a fractional increment, so the state register is no longer an integer.
2. It is *amplified* in the table pass, where the hat functions mix two
   rows with large code jumps, and in the scale pass, where the gates
   with large factors κX are partly open.
3. With centred ramps the state register stays an integer. The
   amplification disappears, and the whole step has a factor of about 1.

## Theory versus practice, in one sentence

The unbounded tape of the theorem becomes about 50 bits per register in
float64, but only in a base that is a power of two. Within this limit the
simulation is exact for every machine, because the only operation that
can round, the gated scaling, does not depend on what the machine does.
In any other base, the network additionally needs margins in its digit
tests, or it amplifies its own rounding errors within a few steps.

## Long runs: exact dyadic integers, and do errors accumulate?

**The exact backend.** The exact evaluation has no length limit; it only
gets slower as the numbers grow with the tape. In a power-of-two base every
value is a dyadic rational. `relu_tm.dyadic` therefore stores the
registers as integers over a shared power of two, `X / 2^K`, and the
weights as integers over one power of two per layer. A layer is then
integer multiply-adds plus a shift:
- if the shift would drop a set bit, `K` grows instead;
- after every step, common factors of two are divided out again.

The results are bit for bit those of `fractions.Fraction` (tested on
random machines). Use it with `run(tm, backend="dyadic")` or
`relu-tm run … -b dyadic`; `precision_profile` uses it automatically as
the reference.

| run | `Fraction` | `dyadic` | speed-up | bits needed at the end (K) |
|---|---|---|---|---|
| filler, 2000 steps (2000 cells) | 1.19 s | 0.16 s | 7.2× | 4000 |
| zigzag, 10 000 steps | 5.22 s | 0.63 s | 8.3× | 262 |
| binary counter, 20 000 steps | 14.06 s | 1.89 s | 7.4× | 36 |
| BB(4), 107 steps | 0.08 s | 0.01 s | 5.7× | 26 |

Formats with a fixed precision move the float limit (⌊p/m⌋ − 1 digits for
p bits), but gain nothing over exact dyadic arithmetic. Examples are int64
with 63 bits, i.e. 5 more base-4 digits than float64, or 1000-bit MPFR
floats.

**Do rounding errors accumulate?** `experiments/numerics_longrun.py` runs
the binary counter for 20 000 steps; its tape grows to 13 cells. The values
are the largest register error, and the failure step:

| base | ramps | dtype | after 100 | after 1000 | after 10 000 | after 20 000 | failure |
|---|---|---|---|---|---|---|---|
| 6 | edge | float64 | 2.6e-4 | | | | step 9 (2 cells) |
| 6 | centre | float32 | 1.2e-3 | 4.4e-2 | | | step 482 (6 cells) |
| 6 | centre | float64 | 4.6e-12 | 7.9e-9 | 1.8e-4 | 2.2e-3 | none yet |
| 8 | edge | float32 | 0 | | | | step 503 (8 cells = capacity + 1) |
| 8 | edge | float64 | 0 | 0 | 0 | 0 | none |

Yes, they accumulate. Centred ramps keep the state register an integer,
but they do not correct the tape registers. Every sweep of the counter
reads deep digits back, and the rounding errors of the weights (1/6, 5/6)
pile up from sweep to sweep. In base 6, float64 the error grows by more
than eight orders of magnitude over 20 000 steps on a tape of at most 13
cells. Base 8 stays bit-exact until its capacity is reached (7 cells in
float32) and is never inexact in float64. **This confirms the power of two
as the default.**

## Possible next steps

* A scaling gate that computes x/β directly (about +1 digit), and margins
  for the hat functions on the state register.
* Margins for the tape registers themselves (re-snapping digits), which
  would stop the accumulation in other bases.
