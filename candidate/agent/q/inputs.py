"""Q's three inputs for one frame, built together."""

from dataclasses import dataclass

import numpy as np

from candidate.agent.q import features, grid, probmap
from candidate.agent.q.config import Q, QConfig
from candidate.agent.behavior.snapshot import Snapshot
from candidate.agent.behavior.states import State
from candidate.agent.runtime import TurnView


@dataclass(frozen=True, eq=False)
class QInputs:
    grid: np.ndarray          # (7, 21, 21) uint8
    scalars: np.ndarray       # (22,) float32
    probmap: np.ndarray       # (10, 10, 10) float32 (logged as float16)


def build_inputs(view: TurnView, snap: Snapshot, current: State, cfg: QConfig = Q) -> QInputs:
    return QInputs(
        grid=grid.build_grid(view.obs, cfg),
        scalars=features.build_scalars(view, snap, current, cfg),
        probmap=probmap.build_probmap(view, snap, cfg),
    )
