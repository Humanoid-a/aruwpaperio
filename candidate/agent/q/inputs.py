"""Q's three inputs for one frame, built together."""

from dataclasses import dataclass

import numpy as np

from candidate.agent.q import confidence, features, grid
from candidate.agent.q.config import Q, QConfig
from candidate.agent.q.snapshot import Snapshot
from candidate.agent.q.types import State
from candidate.agent.runtime import TurnView


@dataclass(frozen=True, eq=False)
class QInputs:
    grid: np.ndarray          # (7, 21, 21) uint8
    scalars: np.ndarray       # (22,) float32
    confidence: np.ndarray    # (8, 10, 10) float32 (logged as float16)


def build_inputs(view: TurnView, snap: Snapshot, current: State, cfg: QConfig = Q) -> QInputs:
    return QInputs(
        grid=grid.build_grid(view.obs, cfg),
        scalars=features.build_scalars(view, snap, current, cfg),
        confidence=confidence.build_confidence(view, snap, cfg),
    )
