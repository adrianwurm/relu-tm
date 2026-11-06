"""Run a machine and its network side by side and compare after every step."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .encoding import DecodeError, decode, encode
from .machine import Configuration, Symbol, TuringMachine
from .network import CompiledNetwork, build_network

BACKENDS = ("exact", "dyadic", "float64", "float32")


@dataclass
class RunResult:
    machine: TuringMachine
    backend: str
    emulator: list[Configuration]
    network: list[Configuration | None]  # None: registers no longer decode
    vectors: list[tuple]

    @property
    def steps(self) -> int:
        return len(self.emulator) - 1

    @property
    def agree(self) -> bool:
        return all(e == n for e, n in zip(self.emulator, self.network, strict=True))

    @property
    def first_mismatch(self) -> int | None:
        for t, (e, n) in enumerate(zip(self.emulator, self.network, strict=True)):
            if e != n:
                return t
        return None

    def table(self, every: int = 1) -> str:
        rows = [f"{'step':>5}  {'match':^5}  {'emulator':<40}  network"]
        for t, (e, n) in enumerate(zip(self.emulator, self.network, strict=True)):
            if t % every and t != self.steps:
                continue
            mark = "ok" if e == n else "FAIL"
            rows.append(f"{t:>5}  {mark:^5}  {e.render():<40}  {n.render() if n else '(not decodable)'}")
        return "\n".join(rows)


def run(
    tm: TuringMachine,
    word: Sequence[Symbol] | str = (),
    steps: int | None = None,
    backend: str = "exact",
    network: CompiledNetwork | None = None,
    base: int | None = None,
    max_steps: int = 10_000,
) -> RunResult:
    """Simulate ``tm`` on ``word`` with the classical emulator **and** with the
    ReLU network (``backend``: exact rationals, exact dyadic integers - faster,
    for power-of-two bases -, float64 or float32) and decode
    the network's registers after every step.

    ``steps=None`` runs until the emulator halts (at most ``max_steps``).
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if steps is None:
        steps = len(tm.run(word, max_steps)) - 1
    net = network or build_network(tm, base)
    emulator = tm.trace(word, steps)
    x0 = encode(emulator[0], net.layout)

    if backend == "exact":
        vectors = net.iterate_exact(x0, steps)
    elif backend == "dyadic":
        vectors = net.iterate_dyadic(x0, steps)
    else:
        import torch

        dtype = torch.float64 if backend == "float64" else torch.float32
        vectors = net.iterate_torch(x0, steps, dtype)

    decoded: list[Configuration | None] = []
    for v in vectors:
        try:
            decoded.append(decode(v, net.layout))
        except DecodeError:
            decoded.append(None)
    return RunResult(tm, backend, emulator, decoded, vectors)
