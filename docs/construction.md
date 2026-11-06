# How the network works

This note explains, layer by layer, how `build_network` turns a transition
table into a ReLU network of width 4, and lists every inequality the
construction relies on. Code: [`src/relu_tm/network.py`](../src/relu_tm/network.py)
(passes) and [`src/relu_tm/program.py`](../src/relu_tm/program.py) (compiler).

## 1. The model

A network $N\in\mathcal{NN}^{3,3,4}_{\mathrm{ReLU}}$ is
$A_{\mathrm{out}}\circ R_L\circ\dots\circ R_1\circ A_{\mathrm{in}}$ with
rational affine maps $A_{\mathrm{in}}:\mathbb{R}^3\to\mathbb{R}^4$,
$A_{\mathrm{out}}:\mathbb{R}^4\to\mathbb{R}^3$, and hidden layers $R_\ell$ of
exactly four ReLU neurons (notation of the NeurIPS 2026 paper, Def. 2.1).
$N$ maps the encoding of a configuration to the encoding of its successor, so
the iterate $N^{[t]}$ simulates $t$ steps.

Applied this way, with $x_{t+1}=N(x_t)$, $N$ is the cell of a **recurrent**
ReLU network. The RNN has no external input: the input word is part of
$x_0$. Its cell is a deep feed-forward network (one step of the machine =
$L$ layers). This is the "fixed recurrent network" of MFCS 2025.
Unrolled, $N^{[t]}$ is a feed-forward network of width 4 and depth $tL$,
the "iterated composition" of the NeurIPS 2026 paper, Def. 2.1.

In PyTorch, $N$ is `Linear(3,4)`, then pairs of `ReLU`/`Linear(4,4)`, then
`Linear(4,3)`. `StepNetwork.iterate` runs the recurrence.

## 2. The encoding

A configuration becomes three numbers $(\bar l, \bar r, q)$, as in
Bournez, Cohen & Wurm (MFCS 2025, Lemma 19):

* $\bar r=\sum_{i\ge1} d(c_{h+i-1})\,\beta^{-i}$ encodes the head cell and the cells to its right,
* $\bar l=\sum_{i\ge1} d(c_{h-i})\,\beta^{-i}$ encodes the cells to the left of the head, nearest first,
* $q$ is the index of the state.

Tape symbol number $i$ (the blank is number 0) has the odd digit
$d=2i+1$. The infinite blank tail is the digit $0$. The base $\beta$ is even
with $\beta\ge 2|\Gamma|$. By default it is the smallest power of two, so
every weight is a dyadic rational. For machines with one non-blank symbol
(blank $\mapsto1$, symbol $\mapsto3$) this gives $\beta=4$, and the
registers are $\{1,3\}$-quadradics as in the paper. The paper's machines
over $\{1,3,B\}$ need $\beta=8$ here (digits $1,3,5$), see below.

**Gap property.** If the leading digit of $x$ is $d$, then $x\in[d/\beta,(d+1)/\beta]$.
The value $x=0$ means a blank tail. Between two cases there is a gap of width
$1/\beta$, so a *continuous* piecewise-linear function can read the leading
digit. A ReLU network can compute such a function.

**Deviation: the blank has its own digit.** If the blank were the digit 0
inside a word, a configuration such as "blank, then 1" would encode as
$0.01_4 = 1/16$, which lies in the gap $(0, 1/4)$. The gap property would
fail. This happens, for example, when the head moves left past the
written part of the tape (MFCS 2025, Thm 23, case $\bar l=0$). Here the
digit 0 only appears in the infinite tail, so every value the network ever
produces stays in the gapped set. The price is one extra digit, which can
double the base (e.g. $\{1,3,B\}$: base 8 instead of 4).

## 3. The register discipline

Three neurons are **registers** `l, r, q`. They hold non-negative values,
so the ReLU leaves them unchanged. The fourth neuron `w` is the **work
neuron**. Each hidden layer computes

* new registers $=$ affine combinations of `l, r, q, w` from the previous layer,
* new `w` $=\mathrm{ReLU}$(affine combination of the same).

Chaining `w` through several layers evaluates a nested expression such as
$\mathrm{ReLU}(1-\mathrm{ReLU}(4\bar r-1))$. Adding a multiple of `w` to a
register commits the result. Each `ReLU` costs one layer. Affine register
updates are free because they fold into the next layer. This is the
"keep the inputs alive, compute in one extra neuron" principle of Lemma 3.1
in the NeurIPS 2026 paper. `ProgramBuilder` implements it in about 60 lines.
`relu-tm info bb2 --program` prints the resulting layer listing.

## 4. One step = five passes

Notation: $k=|\Gamma|$, $K=k+1$ (head codes per state), $n=|Q|$,
$C=nK$ (case codes), done ranges $D_L=C$, $D_S=C+n$, $D_R=C+2n$,
$\kappa=1-1/\beta$. Digits are $e_0=0$ (tail), $e_g=2g-1$ for $g=1..k$.

### Pass 0 - read the head symbol (no gating)

Start with $q\leftarrow Kq$. For each gap $g=1,\dots,k$ ($\theta_1=0$,
$\Delta_1=1$, and $\theta_g=1$, $\Delta_g=2$ for $g\ge2$):

$$w\leftarrow \mathrm{ReLU}(\beta\bar r-\theta_g),\qquad w\leftarrow\mathrm{ReLU}(1-w),\qquad
q\mathrel{+}=1-w,\qquad \bar r\mathrel{-}=\tfrac{\Delta_g}{\beta}(1-w).$$

*Invariant.* Before gap $g$, the register holds $\bar r-e_{j}/\beta$ with
$j=\min(g-1,\text{digit index})$. Hence $\beta\bar r-\theta_g\le 0$ if the
digit is below $e_g$, and $\ge 1$ otherwise. It never lands in the open gap
$(0,1)$, so $1-w=\min(1,\mathrm{ReLU}(\cdot))\in\{0,1\}$ is exact. Finally
$\bar r\leftarrow\beta\bar r$. Result: $q=Kq+\sigma$ with $\sigma\in\{0..k\}$
the head code (0 = tail, $i+1$ = symbol $i$), and $\bar r=\beta\bar r-\mathrm{lead}(\bar r)$.

*Option `ramps="centre"`.* Instead of $\mathrm{ReLU}(\beta\bar r-\theta_g)$
the test uses $\mathrm{ReLU}(2(\beta\bar r-\theta_g)-\tfrac12)$, a ramp on
the middle half of the gap only. Exact values give the same results. Values
within 1/4 of an admissible case still give exactly 0 or 1, which keeps
floating-point rounding from being amplified ([numerics.md](numerics.md)).
The same applies to the gap tests of pass 3.

### Pass 1 - the transition table (one hat per case code)

For every code $c=0..C-1$, i.e. every pair (state, head code):

$$w\leftarrow\mathrm{ReLU}(c-q),\qquad w\leftarrow \mathrm{ReLU}(1-q+c-2w) = [q=c]\ \text{on integers}.$$

Then $q\mathrel{+}=(\mathrm{done}(c)-c)\,w$ with $\mathrm{done}(c)=D_m+q'$, which
records the new state $q'$ *and* the move $m$. The written digit $d'$ is also
pushed:

* R: $\bar l\mathrel{+}=d'w$
* S: $\bar r\mathrel{+}=d'w$
* L: $\bar r\mathrel{+}=(d'+\beta)w$. The $+\beta$ turns into $+1$ after pass 2 and supplies the "blank tail becomes blank digit" of pass 3.

Halting states and undefined transitions are treated as "write the same
symbol, stay", so halted configurations are fixed points. Done codes are
$\ge C$, so the hats of later codes ignore them. Each code is rewritten
exactly once.

### Pass 2 - gated scaling

**Gate lemma.** Let $0\le x\le X$ and $\mathrm{dist}\in\{0\}\cup[1,\infty)$.
Then $\mathrm{ReLU}(\kappa x-\kappa X\cdot\mathrm{dist})$ equals $\kappa x$ if
$\mathrm{dist}=0$ and $0$ otherwise. On integers $q$, a one-sided range test
is a single ReLU: $\mathrm{dist}=\mathrm{ReLU}(D_R-q)$ is 0 exactly on the
R-range.

* R: $\bar l\leftarrow\bar l/\beta$ (here $\bar l\le 1+d'\le\beta$, so $X=\beta$).
* L, S: $\bar r\leftarrow\bar r/\beta$ ($\bar r\le 2\beta$, so $X=2\beta$).

For the codes whose gate is closed, the registers are $\le1$ and the
subtracted term $\kappa X\,\mathrm{dist}\ge\kappa X$ dominates.

### Pass 3 - moves L: transfer the leading digit of $\bar l$ to $\bar r$

These are the gap tests of pass 0, applied to $\bar l$ and gated by
$\mathrm{dist}=\mathrm{ReLU}(q-D_S+1)$:
$w\leftarrow\mathrm{ReLU}(\beta\bar l-\theta_g-\beta\,\mathrm{dist})$. Each
increment subtracts $\Delta_g/\beta$ from $\bar l$ and adds $\rho_g$ to
$\bar r$, with $\rho_1=0$ and $\rho_g=2$. Because $\rho_1=0$ and pass 1
already added the constant 1, a blank tail (digit 0) arrives in $\bar r$ as
the blank digit 1. Then two gated rescalings follow: $\bar l\leftarrow\beta\bar l$
(with factor $\beta-1$ and bound $\beta-1$) and $\bar r\leftarrow\bar r/\beta$.

### Pass 4 - back to the plain state index

$q\mathrel{-}=n\,[q\ge D_R]$, then $q\mathrel{-}=n\,[q\ge D_S]$, then $q\leftarrow q-D_L$.
Each indicator is $1-\mathrm{ReLU}(1-\mathrm{ReLU}(q-D+1))$.

### Size

$$\text{hidden layers} = \underbrace{2k}_{\text{pass 0}}+\underbrace{2C}_{\text{pass 1}}+\underbrace{4}_{\text{pass 2}}
+\underbrace{3k+4}_{\text{pass 3}}+\underbrace{4}_{\text{pass 4}} = 5k+2nK+12,$$

all of width exactly 4. Examples: BB(2) 40, BB(4) 52, binary adder 92 hidden layers.

## 5. Exact versus floating point

All weights are dyadic rationals with small denominators. All operations
are exact in binary floating point as long as every register value fits
into the mantissa: about $(\text{digits})\cdot\log_2\beta$ bits plus a few
bits of headroom, against 53 bits for float64 and 24 bits for float32.
First measurement with the `filler` machine (one new base-4 digit per step):
float64 is exact up to 25 digits and float32 up to 11. Beyond that, digits
are rounded away and the decoded configuration is wrong. A systematic
study is step 5 of the project.

## 6. Relation to the papers

| | MFCS 2025, Lemma 19 / Thm 23 | this package |
|---|---|---|
| width | 3: neurons for $\bar l$, $\bar r$ and one working neuron; the state is folded into a tape register (Lemma 21). Remark 22: max-min strings (Prop. 17) would give width 4, i.e. $\bar l{+}q$, $\bar r$ and two working neurons | 4: registers $\bar l$, $\bar r$, $q$ (the state keeps its own register) and one working neuron |
| machines | $\Gamma=\{1,3,B\}$, moves L/R | any finite $\Gamma$, moves L/R/S |
| encoding | base 4, digits $1,3$ for the symbols, blank $B=0$ | odd digits $1,3,5,\dots$ for all symbols **including** the blank; $0$ only for the infinite tail; base $2^m\ge2\lvert\Gamma\rvert$. Machines with one non-blank symbol (busy beavers) give $\{1,3\}$-quadradics in base 4; a machine over $\{1,3,B\}$ needs base 8 here |
| blank at the end of the tape | $\bar l=0$: $\bar l\leftarrow4\bar l$, $\bar r\leftarrow\bar r/4$ (pushes a digit 0) | a blank tail is pushed as the blank digit, so every register stays in the gapped set |
| granularity | $g_M$ is one TM step | the same: one application of $N$ = exactly one TM step |
| arithmetic | exact rationals | exact rationals (`fractions`) **and** float32/float64 (PyTorch) |

Tail blank and written blank remain distinguishable for the network: after
pass 0 they have different head codes ($\sigma=0$ and $\sigma=1$). They
only lead to the same transition because both mean "blank" for the machine.
A construction that needs to tell them apart (e.g. to detect the end of the
used tape) can simply give the two codes different rows in pass 1.

The NeurIPS 2026 paper (arXiv:2610.06092) uses the same digit gadgets and
the same compilation discipline. It proves that *verifying* such narrow
networks is NP-complete at width 4 on all of $[0,1]$, and at width 3 on the
encoding set.
