"""PyTorch version of the constructed network (weights are set, not trained)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover
    from .network import CompiledNetwork


class StepNetwork(nn.Module):
    """The recurrent cell ``N : R^3 -> R^3``: ``Linear(3,4)``, then ``ReLU`` /
    ``Linear(4,4)`` pairs, and a final ``Linear(4,3)`` read-out.  Every
    hidden layer has exactly four neurons.  :meth:`iterate` runs the
    recurrence ``x_{t+1} = N(x_t)`` (an RNN without external input)."""

    def __init__(self, layers: nn.Sequential) -> None:
        super().__init__()
        self.layers = layers

    @classmethod
    def from_compiled(cls, net: CompiledNetwork, dtype: torch.dtype = torch.float64) -> StepNetwork:
        mods: list[nn.Module] = []
        for W, b in net.hidden:
            lin = nn.Linear(len(W[0]), len(W), dtype=dtype)
            mods += [lin, nn.ReLU()]
            _assign(lin, W, b, dtype)
        W, b = net.output
        lin = nn.Linear(len(W[0]), len(W), dtype=dtype)
        _assign(lin, W, b, dtype)
        mods.append(lin)
        module = cls(nn.Sequential(*mods))
        module.requires_grad_(False)
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @torch.no_grad()
    def iterate(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        """Trajectory ``[x, N(x), ..., N^steps(x)]`` stacked along dim 0."""
        out = [x]
        for _ in range(steps):
            x = self(x)
            out.append(x)
        return torch.stack(out)

    @torch.no_grad()
    def activations(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs of all hidden ReLU layers for one application, shape
        ``(depth, *batch, 4)`` - handy for plotting."""
        acts = []
        for mod in self.layers:
            x = mod(x)
            if isinstance(mod, nn.ReLU):
                acts.append(x)
        return torch.stack(acts)


def _assign(lin: nn.Linear, W, b, dtype: torch.dtype) -> None:
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[float(v) for v in row] for row in W], dtype=dtype))
        lin.bias.copy_(torch.tensor([float(v) for v in b], dtype=dtype))
