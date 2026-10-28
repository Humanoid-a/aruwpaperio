"""The Q network: one PyTorch module, used for training and during a match.

Architecture (the brief, "Network structure and output", with the probability map of
docs/Q_Probability_Map_Input.md as the third input). Three branches, each reading one input group, feed a shared
head with five outputs. All convolutions are 3 x 3 with padding 1, and every layer except the last uses ReLU:

    grid        7 x 21 x 21  conv 16 -> conv 32 s2 -> conv 32 s2 -> conv 32 s2 -> flatten 288 -> dense 64
                             (spatial sizes 21 -> 21 -> 11 -> 6 -> 3)
    scalars     23           dense 64
    probmap     10 x 10 x 10 conv 16 -> conv 32 s2 -> conv 32 s2 -> flatten 288 -> dense 32
                             (spatial sizes 10 -> 10 -> 5 -> 3)
    head        160          dense 64 -> dense 5      (inputs concatenated: grid, scalars, probmap)

79,478 parameters at width 1. The six outputs are in State order (EXPAND, EXPAND2, RETURN, HUNT, TRACK, PATROL) and
in cells of score difference.

Width. `width` multiplies the channels of both convolutional branches, their dense outputs and the head's hidden
layer (the scalar branch stays at 64): width 2 is 32/64/64/64 channels on the grid, 32/64/64 on the probability map,
a 128-wide head, about 299,000 parameters. A weights file records its width; files without one are width 1.

Design choices
  * Branch switches. The brief trains in stages: the scalars branch alone, then with the grid, then with the
    probability map, each stage starting from the previous weights. A switched-off branch outputs zeros of its usual
    width, so the head always takes 160 inputs and its weights carry over from stage to stage unchanged.
  * One file format: torch.save of {"state_dict", "branches", "meta"}, loaded with weights_only=True (plain tensors and
    values only, no pickled code). meta records what the network was trained on.
  * The device. pick_device() picks the GPU only if PyTorch can really run a kernel on it: a PyTorch build without
    kernels for the card still reports CUDA as available and fails on the first operation (seen here with a cu126
    build on an RTX 50-series card). Otherwise the CPU, which is fast enough for this network (well under a
    millisecond a frame).
  * Robustness of the submitted controller. Nothing that runs without a trained network imports this module, and the
    Candidate Controller falls back to the hand-set stand-in if PyTorch or the weights file is missing (see
    candidate_controller.load_q).
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from candidate.agent.behavior.states import N_STATES
from candidate.agent.q.features import N_SCALARS

BRANCHES = ("grid", "scalars", "probmap")
WIDTH = {"grid": 64, "scalars": 64, "probmap": 32}       # each branch's output width; the head takes their sum


def _convs(channels: list[int], strides: list[int]) -> list[nn.Module]:
    """3 x 3 convolutions with padding 1, each followed by a ReLU."""
    layers: list[nn.Module] = []
    for cin, cout, stride in zip(channels, channels[1:], strides):
        layers += [nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.ReLU()]
    return layers


def widths(width: int = 1) -> dict:
    """Each branch's output width at a given width multiplier (the head takes their sum)."""
    return {"grid": WIDTH["grid"] * width, "scalars": WIDTH["scalars"], "probmap": WIDTH["probmap"] * width}


class QNet(nn.Module):
    def __init__(self, branches=BRANCHES, width: int = 1) -> None:
        super().__init__()
        w = int(width)
        self.width = w
        self.widths = widths(w)
        c1, c2 = 16 * w, 32 * w
        self.grid = nn.Sequential(*_convs([7, c1, c2, c2, c2], [1, 2, 2, 2]), nn.Flatten(), nn.Linear(c2 * 3 * 3, self.widths["grid"]),
                                  nn.ReLU())
        self.scalars = nn.Sequential(nn.Linear(N_SCALARS, 64), nn.ReLU())
        self.probmap = nn.Sequential(*_convs([10, c1, c2, c2], [1, 2, 2]), nn.Flatten(), nn.Linear(c2 * 3 * 3, self.widths["probmap"]),
                                     nn.ReLU())
        self.head = nn.Sequential(nn.Linear(sum(self.widths.values()), 64 * w), nn.ReLU(), nn.Linear(64 * w, N_STATES))
        self.set_branches(branches)

    def set_branches(self, branches) -> None:
        unknown = set(branches) - set(BRANCHES)
        if unknown:
            raise ValueError(f"unknown branches {sorted(unknown)}")
        self.branches = tuple(b for b in BRANCHES if b in branches)

    @property
    def output_layer(self) -> nn.Linear:
        return self.head[-1]

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor, probmap: torch.Tensor) -> torch.Tensor:
        """Batched: grid (N, 7, 21, 21), scalars (N, 23), probmap (N, 10, 10, 10), all float -> (N, 6) values."""
        inputs = {"grid": grid, "scalars": scalars, "probmap": probmap}
        parts = [getattr(self, b)(inputs[b]) if b in self.branches else scalars.new_zeros(len(scalars), self.widths[b])
                 for b in BRANCHES]
        return self.head(torch.cat(parts, dim=1))


def parameter_count(model: nn.Module | None = None) -> int:
    return sum(p.numel() for p in (model or QNet()).parameters())


# ---- files -----------------------------------------------------------------------------------------------------------


def save_network(model: QNet, path: str | Path, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "branches": list(model.branches), "width": int(getattr(model, "width", 1)),
                "meta": json.dumps(meta or {})}, path)
    return path


def load_network(path: str | Path, device: str | torch.device = "cpu") -> QNet:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = QNet(saved["branches"], width=int(saved.get("width", 1)))
    model.load_state_dict(saved["state_dict"])
    meta = saved.get("meta", "{}")
    model.meta = json.loads(meta) if isinstance(meta, str) else dict(meta)
    return model.to(device).eval()


# ---- the device --------------------------------------------------------------------------------------------------------


def pick_device(requested: str | torch.device | None = None) -> torch.device:
    """The device to run on: `requested` if given; else the GPU if a kernel really runs on it; else the CPU."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        try:
            (torch.zeros(4, device="cuda") + 1).sum().item()
            return torch.device("cuda")
        except Exception:  # noqa: BLE001 - any failure means the GPU cannot be used
            pass
    return torch.device("cpu")


# ---- running it ----------------------------------------------------------------------------------------------------------


@torch.inference_mode()
def predict(model: QNet, grid: np.ndarray, scalars: np.ndarray, probmap: np.ndarray, batch: int = 2048) -> np.ndarray:
    """(N, 6) float64 values for N frames given as NumPy arrays (the grid may be uint8, the probability map float16)."""
    device = next(model.parameters()).device
    model.eval()
    out = []
    for start in range(0, len(scalars), batch):
        part = slice(start, start + batch)
        as_tensor = lambda a: torch.as_tensor(np.asarray(a[part], dtype=np.float32), device=device)  # noqa: E731
        out.append(model(as_tensor(grid), as_tensor(scalars), as_tensor(probmap)).double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, N_STATES))


class NetworkQ:
    """The trained Q as QStrategy uses it: values(frame, inputs) -> (6,). It reads Q's inputs, so the strategy must build
    them (QStrategy(build_inputs=True), the default)."""

    needs_inputs = True

    def __init__(self, model: QNet, device: str | torch.device | None = None) -> None:
        from candidate.agent.q.version import inputs_version

        self.device = pick_device(device)
        self.model = model.to(self.device).eval()
        meta = getattr(model, "meta", {})
        self.name = meta.get("name", "network")
        # the weights were trained on inputs built by some version of the code; it should be the current one
        self.inputs_version = meta.get("inputs_version")
        self.current_version = inputs_version()
        self.inputs_match = self.inputs_version == self.current_version

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "NetworkQ":
        return cls(load_network(path), device)

    def values(self, frame, inputs) -> np.ndarray:
        if inputs is None:
            raise ValueError("the network needs Q's inputs; build them (QStrategy(build_inputs=True))")
        return predict(self.model, inputs.grid[None], inputs.scalars[None], inputs.probmap[None])[0]
