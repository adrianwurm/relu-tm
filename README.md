# relu-tm

[![CI](https://github.com/adrianwurm/relu-tm/actions/workflows/ci.yml/badge.svg)](https://github.com/adrianwurm/relu-tm/actions/workflows/ci.yml)

**A fixed width-4 ReLU network that simulates a Turing machine, tested step by step.**

Give `relu-tm` the transition table of a deterministic Turing machine and it
*computes* (no training) the weights of a ReLU network $N:\mathbb R^3\to\mathbb R^3$.
Every hidden layer of $N$ has exactly **four neurons**. $N$ maps the
encoding $(\bar l, \bar r, q)$ of a configuration (left tape, head and right
tape, state) to the encoding of the next configuration.

$N$ is used as a **recurrent network**: its output is fed back as the next
input, $x_{t+1} = N(x_t)$. This is an RNN without external input, whose cell
is a deep feed-forward network of width 4. Unrolled for $t$ steps it becomes
the feed-forward network $N^{[t]}$ of width 4. These iterated width-4
networks are the subject of the NeurIPS 2026 paper by Bournez, Cohen & Wurm
(see [below](#relation-to-the-neurips-2026-paper)).

A classical emulator checks the network after **every** step, in exact
rational arithmetic (fractions, or fast dyadic integers) and in
float32/float64.

> **Scope.** `relu-tm` implements the *simulation* only: part (i) of
> Theorem 3 of the paper, in a width-4 variant. It does **not** implement
> the certificate of Theorem 3 (part (ii)), and it makes no claim about
> inputs that are not encodings of configurations.

![BB(4) simulated by the network: the tape decoded from the network after every step, and the state register and work neuron through the 52 layers of one step](docs/figures/core.png)

*Left: the 4-state busy beaver, with its tape decoded from the network's registers after each of
its 107 steps (identical to the emulator). Right: one application of the network. The state register
takes in the head symbol, jumps to the code of the transition that applies, and returns to the new
state. A guided tour is in [notebooks/tour.ipynb](notebooks/tour.ipynb).*

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: small CPU-only PyTorch
pip install "git+https://github.com/adrianwurm/relu-tm"
# or, from a clone:  pip install -e ".[dev]"
```

## Try it

```console
$ relu-tm run bb2
 step  match  emulator                                  network
    0   ok    A: [0]                                    A: [0]
    1   ok    B: 1 [0]                                  B: 1 [0]
    2   ok    A: [1] 1                                  A: [1] 1
    3   ok    B: [0] 1 1                                B: [0] 1 1
    4   ok    A: [0] 1 1 1                              A: [0] 1 1 1
    5   ok    B: 1 [1] 1 1                              B: 1 [1] 1 1
    6   ok    Z: 1 1 [1] 1                              Z: 1 1 [1] 1

All 6 steps agree (width-4 network, exact).
```

```bash
relu-tm list                                   # example machines
relu-tm run bb4 --every 10                     # busy beaver, 107 steps
relu-tm run 1RB1LB_1LA0LC_1RZ1LD_1RD0RA        # any machine in bbchallenge format
relu-tm run adder -i 101+11                    # binary addition
relu-tm run filler -n 30 -b float32            # watch float32 run out of digits
relu-tm run counter -n 20000 -b dyadic --every 5000   # long exact run with dyadic integers
relu-tm info bb2 --program                     # the network, layer by layer
```

## Your own machine in Python

```python
from relu_tm import TuringMachine, build_network, run

# flips every bit, then halts on the first blank
tm = TuringMachine(
    {("flip", "0"): ("flip", "1", "R"),
     ("flip", "1"): ("flip", "0", "R"),
     ("flip", "_"): ("done", "_", "S")},
    start="flip", blank="_", halt_states={"done"},
)

result = run(tm, "0110")             # emulator vs. network, exact arithmetic
print(result.table())
assert result.agree

net = build_network(tm)              # exact rational weights
print(net.width, net.depth)          # 4, number of hidden layers
module = net.to_torch()              # torch.nn.Module (float64 by default)
```

Finite automata (`DFA`) and deterministic pushdown automata (`DPDA`) share
a small interface with Turing machines. `to_turing_machine()` compiles them
into a TM that the network can run, and the automaton serves as an
independent oracle.

## Import and export

Machines can be read from the files of other tools. They are translated
into `relu-tm`'s own types (`TuringMachine`, `DFA`, `DPDA`), and
`build_network` then works as usual. Export goes the other way.

| format | file | machines | extra software |
|---|---|---|---|
| [bbchallenge](https://bbchallenge.org) text, e.g. `1RB1LB_1LA1RZ` | `.txt` or on the command line | Turing machines (symbols 0–9, L/R) | none |
| [JFLAP](https://www.jflap.org) | `.jff` | Turing machines, finite automata, pushdown automata | none |
| [turingmachine.io](https://turingmachine.io) | `.yaml` | Turing machines (L/R) | PyYAML (`pip install pyyaml`, included in `.[test]`/`.[dev]`) |

```bash
relu-tm run machine.jff -i 0110               # a JFLAP file (DFAs/DPDAs are compiled to a TM first)
relu-tm run palindrome.yaml                   # turingmachine.io: the input stored in the file is used
relu-tm convert bb4 --to turingmachine.io -o bb4.yaml
relu-tm convert machine.jff --to bbchallenge
```

```python
from relu_tm import load, run
mf = load("palindrome.yaml")      # MachineFile(machine, input, format)
print(run(mf.turing_machine(), mf.input).agree)
```

Where the formats differ from this package:

- **JFLAP.** It accepts as soon as a final state is reached, so final states become halting accept states. The wildcards `~` (any symbol, or keep the symbol when writing) and `!a` (any symbol except `a`) are expanded; they range over the symbols that appear in the file. Nondeterminism, multi-tape machines and building blocks are rejected with an error.
- **turingmachine.io.** It has no notion of acceptance, so halting states named `accept`/`accepted`/`yes` become accept states.
- **bbchallenge.** The format has no stay moves and no accept states.

All 15 example machines of turingmachine.io import correctly and run identically on the network.

## How it works (short version)

* **Encoding** (as in Bournez, Cohen & Wurm, MFCS 2025): a tape word becomes a
  base-$\beta$ fraction with odd digits $1,3,5,\dots$ and the blank tail is $0$.
  The encoded values therefore have gaps, and a *continuous* piecewise-linear
  function can read the leading digit. Unlike the paper, the blank also gets
  its own digit, and $0$ is reserved for the infinite blank tail. For
  machines with one non-blank symbol (e.g. busy beavers) the registers are
  $\{1,3\}$-quadradics in base 4.
* **Registers + one work neuron**: three neurons carry `l, r, q` unchanged
  (their values stay $\ge0$), and the fourth evaluates nested ReLU
  expressions. One step of the machine consists of five passes: read the
  head symbol, look up the transition (one "hat" function per case), scale
  the tape registers, move a digit for left moves, restore the state.
* **No training**: all weights are dyadic rationals computed from the
  transition table. The network has $5|\Gamma| + 2|Q|(|\Gamma|+1) + 12$
  hidden layers (BB(4): 52).

The full construction, every inequality it uses, and the deviations from
the papers are in [docs/construction.md](docs/construction.md).

## Relation to the NeurIPS 2026 paper

Theorem 3 of Bournez, Cohen & Wurm (NeurIPS 2026) builds, for every
Turing machine $M$, a network $N_M:[0,1]^4\to[0,1]^4$ of width 4 with two
properties: (i) its iterates simulate $M$ exactly on encoded tapes, and
(ii) a fourth coordinate, the *certificate*, irreversibly flags every point
of $[0,1]^4$ whose simulation reads an invalid digit. The certificate is
what makes the hardness results of the paper hold on convex input regions.

| | Theorem 3 of the paper | `relu-tm` |
|---|---|---|
| simulation, part (i) | yes: a width-3 step (Stage 1, after MFCS 2025) | **yes**, tested step by step against an emulator |
| certificate, part (ii) | yes: $x_4 \leftarrow \max\lbrace x_4, \varepsilon(x_1,x_2)\rbrace$ | **no** |
| the four neurons | $x_1$ left tape, $x_2$ state ∥ right tape, $x_3$ scratch, $x_4$ certificate | $\bar l$ left tape, $\bar r$ right tape, $q$ state, $w$ work neuron |
| machines | deterministic, binary alphabet, infinite tapes | deterministic, any finite alphabet, finite tapes |
| encoding | base 4, digits $\lbrace 1,3\rbrace$ only | odd digits for all symbols; digit $0$ for the blank tail |

So the width 4 of `relu-tm` is not the width 4 of Theorem 3. The paper
needs width 3 for the simulation alone and pays the fourth neuron for the
certificate. `relu-tm` keeps the state in its own register instead of
packing it into the right tape, which costs the fourth neuron. A
certificate as in the paper would also flag the digit 0 that `relu-tm` uses
for the blank tail, so adding it would require the paper's encoding.

## Notebook

[notebooks/tour.ipynb](notebooks/tour.ipynb) covers:
- your own machine;
- the tape over time;
- the five passes inside one step;
- which row of the transition table fires at each step;
- the numerics plots;
- a short section on how the simulation relates to the complexity of verification (NP / NEXP / RE),
  with a brute-force verifier that runs the network on all 2ⁿ inputs at once.

To run it: `pip install -e ".[notebook]"` and `jupyter lab notebooks/tour.ipynb`. The figures come from `relu_tm.viz`.

## Tests

```bash
pytest                       # ~180 tests, about 20 s
HYPOTHESIS_PROFILE=ci pytest # more random examples, as in CI
```

The central property test generates random machines (up to 4 states and 3
symbols) and random inputs. It checks that after every one of 30 steps the
decoded network registers equal the emulator's configuration. Further tests
cover the busy beavers BB(2)–BB(4), a binary counter, a palindrome
decider, a binary adder, and DFAs/DPDAs compiled to TMs. CI runs on Python
3.10–3.13 for every push.

## Exact versus floating point

With exact rationals the simulation never fails. With the same weights in
floating point ([docs/numerics.md](docs/numerics.md), `relu-tm precision ...`):

* **Base 2^m** (default): all weights are dyadic, so float is *bit-exact* as
  long as every register fits into the mantissa. With p = 24 or 53
  significand bits, every register may hold ⌊p/m⌋ − 1 digits, whatever the
  machine does:

  | base | float32 | float64 |
  |---|---|---|
  | 4 | 11 | 25 |
  | 8 | 7 | 16 |
  | 16 | 5 | 12 |

  These are guaranteed values. In 692 runs (BB(4), counter, adder,
  palindrome, 300 random machines), no run failed while its registers
  stayed within them. Some runs get one digit further, but not reliably:
  the exact limit of a push is ⌊(p−1−h)/m⌋, with 2^h the largest power of
  two in κ(x+d), so it depends on the digit d (in base 4 and float64, 26
  digits when d = 1, 25 when d = 3). Beyond the limit the simulation
  breaks abruptly, and the number of steps does not matter. Example: BB(4)
  needs 14 digits, so float64 runs all 107 steps bit-exactly, while float32
  breaks at step 90.
* **Other bases** (6, 10, 12) round from the first step. If the machine reads
  its tape back, the digit tests of the papers amplify these errors, and the
  simulation fails after a few steps on a tape of only 2–6 cells. With the
  option `build_network(tm, ramps="centre")`, which puts the ramps in the
  middle of the gaps, the network tolerates the rounding and gets about as
  far as in base 2^m.

## Status / roadmap

- [x] emulator (TM, DFA, DPDA), encoding, width-4 construction, exact + float evaluation, tests, CI
- [x] import/export: JFLAP, turingmachine.io, bbchallenge
- [x] numerics step 1: register capacity in float32/float64, role of the base and of the digit tests
- [x] numerics step 2: realistic machines (692 runs). The failure depends only on register length; the gated scaling κ·x rounds first
- [x] notebook: tape and activations over time, numerics plots, relation to verification complexity (NP / NEXP / RE)
- [ ] optional: variants as in MFCS 2025: width 3 ($\bar l{+}q$, $\bar r$, one working neuron) and width 4 via max-min strings ($\bar l{+}q$, $\bar r$, two working neurons)

## Related work

* O. Bournez, J. Cohen, A. Wurm. *A Universal Uniform Approximation Theorem for Neural Networks.*
  MFCS 2025. [doi:10.4230/LIPIcs.MFCS.2025.29](https://doi.org/10.4230/LIPIcs.MFCS.2025.29). Thm 23 is the width-3 TM construction this package follows.
* O. Bournez, J. Cohen, A. Wurm. *Iterated Narrow Networks: Turing Completeness at Width 4 on Convex Input Spaces
  and the Complexity of Verification.* NeurIPS 2026, [poster page](https://neurips.cc/virtual/2026/poster/148965).
  Theorem 3: a width-4 network simulates any Turing machine, with a certificate on all of $[0,1]^4$ (`relu-tm`
  implements the simulation, not the certificate). Theorem 7: reachability for iterated width-4 networks is
  NP-complete (time bound in unary), NEXP-complete (binary) and RE-complete (no bound).
* O. Bournez, J. Cohen, L. Cohen, A. Wurm. *Fractal Gadgets for Neural Networks: The Complexity of the Narrow Regime.*
  FST/TCS 2026, [arXiv:2610.06092](https://arxiv.org/abs/2610.06092). Companion paper: verification is NP-complete
  at width 4 on $[0,1]$ and at width 3 on the encoding set. Its Lemma 3.1 is the compilation discipline used here.
* J. Bates. *A PyTorch Library of Turing-Complete Neural Networks*, 2026,
  [arXiv:2605.08150](https://arxiv.org/abs/2605.08150), [code](https://github.com/jonrbates/turing).
  Implements Siegelmann–Sontag RNNs and a Wei–Chen–Ma transformer in float arithmetic. `relu-tm` differs in using a
  fixed width of 4, exact rational evaluation, and property-based tests against an emulator.
* A. Giannou et al. *Looped Transformers as Programmable Computers.* ICML 2023. [arXiv:2301.13196](https://arxiv.org/abs/2301.13196)
* G. Carmantini et al. *Turing Computation with Recurrent Artificial Neural Networks*, 2015. [arXiv:1511.01427](https://arxiv.org/abs/1511.01427)

## Acknowledgement

Developed with Claude Code as a coding assistant. The construction follows
Bournez, Cohen & Wurm (MFCS 2025, Thm 23) and the compilation discipline of
the FST/TCS 2026 companion paper, and implements the simulation part of
Theorem 3 of the NeurIPS 2026 paper.

## License

MIT, see [LICENSE](LICENSE).
