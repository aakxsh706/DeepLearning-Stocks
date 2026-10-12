"""Instrumented recurrent cells: training and telemetry use the exact same computation."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class InstrumentedLSTMCell(nn.Module):
    """PyTorch i, f, g, o gate order and bias convention, with inspectable states."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        for p in self.parameters():
            nn.init.uniform_(p, -1 / math.sqrt(hidden_size), 1 / math.sqrt(hidden_size))

    def forward(self, x, state):
        h, c = state
        i, f, g, o = (F.linear(x, self.weight_ih, self.bias_ih)
                       + F.linear(h, self.weight_hh, self.bias_hh)).chunk(4, dim=-1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c, {"input": i, "forget": f, "candidate": g, "output": o}


class InstrumentedGRUCell(nn.Module):
    """PyTorch r, z, n convention, including reset after the hidden affine map."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        for p in self.parameters():
            nn.init.uniform_(p, -1 / math.sqrt(hidden_size), 1 / math.sqrt(hidden_size))

    def forward(self, x, h):
        ir, iz, inn = F.linear(x, self.weight_ih, self.bias_ih).chunk(3, dim=-1)
        hr, hz, hn = F.linear(h, self.weight_hh, self.bias_hh).chunk(3, dim=-1)
        r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        n = torch.tanh(inn + r * hn)
        h = (1 - z) * n + z * h
        return h, {"reset": r, "update": z, "candidate": n}


class QuantileRNN(nn.Module):
    def __init__(self, input_size=8, hidden_size=16, kind="lstm"):
        super().__init__()
        if kind not in {"lstm", "gru"}:
            raise ValueError("Model kind must be lstm or gru")
        self.kind, self.hidden_size = kind, hidden_size
        self.cell = (InstrumentedLSTMCell if kind == "lstm" else InstrumentedGRUCell)(input_size, hidden_size)
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, x, record=False):
        h = x.new_zeros(x.shape[0], self.hidden_size)
        c = torch.zeros_like(h)
        hidden, cells, gates = [], [], {}
        for xt in x.unbind(dim=1):
            if self.kind == "lstm":
                h, c, gate = self.cell(xt, (h, c))
            else:
                h, gate = self.cell(xt, h)
            if record:
                hidden.append(h)
                if self.kind == "lstm":
                    cells.append(c)
                for name, value in gate.items():
                    gates.setdefault(name, []).append(value)
        raw = self.head(h)
        # Monotone parameterization in percentage points, then convert to returns.
        median = raw[:, 0]
        quantiles = torch.stack((median - F.softplus(raw[:, 1]), median,
                                median + F.softplus(raw[:, 2])), dim=-1) / 100
        telemetry = None
        if record:
            telemetry = {"hidden": torch.stack(hidden, dim=1),
                         "gates" if self.kind == "lstm" else "gru_gates":
                             {key: torch.stack(value, dim=1) for key, value in gates.items()},
                         "head_raw": raw}
            if cells:
                telemetry["cell"] = torch.stack(cells, dim=1)
        return quantiles, telemetry


def quantile_loss(prediction, target):
    error = (target[:, None] - prediction) * 100
    q = prediction.new_tensor([0.1, 0.5, 0.9])
    return torch.maximum(q * error, (q - 1) * error).mean()
