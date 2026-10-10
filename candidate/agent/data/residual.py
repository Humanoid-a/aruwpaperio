"""The residual: probability mass whose history has been thrown away.

Pruning keeps a bounded number of hypotheses per position. What it drops is not
forgotten: it goes into the residual of the position it was at, so the probability
of the opponent being at any position is never changed by pruning. The residual
knows only where, not how: no trail, no land, no count. It is moved along (one
step per frame, and a little of it back to the respawn, since it may have died)
and cut by evidence about position, but it cannot be tested against anything that
needs a history, which is why part of it is handed back to the hypotheses when
they hold too little of the belief (belief.settle.rehydrate).
"""

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH


class ResidualGrid:
    """Probability mass per cell, as (50, 50) float64."""

    def __init__(self) -> None:
        self.mass = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float64)

    def total(self) -> float:
        return float(self.mass.sum())

    def teleport(self, cell: int, fraction: float) -> None:
        """Move `fraction` of every cell's mass to `cell` (an opponent of unknown history may
        have died and respawned there, which no step-by-step walk can reach)."""
        total = self.total()
        self.mass = self.mass * (1.0 - fraction)
        self.mass.flat[cell] += fraction * total

    def spread(self, legal_destination: np.ndarray) -> None:
        """One opponent move: each cell's mass goes in equal shares to its legal
        neighbours (on the board and where `legal_destination` is True). The
        opponent must move, so none of it stays unless it has nowhere to go."""
        legal = legal_destination
        degree = np.zeros(self.mass.shape, dtype=np.float64)
        degree[1:, :] += legal[:-1, :]    # source (y, x) can go north to (y-1, x)
        degree[:-1, :] += legal[1:, :]
        degree[:, 1:] += legal[:, :-1]
        degree[:, :-1] += legal[:, 1:]
        stuck = degree == 0
        share = np.where(stuck, 0.0, self.mass / np.maximum(degree, 1.0))
        out = np.where(stuck, self.mass, 0.0)
        out[:-1, :] += share[1:, :] * legal[:-1, :]   # from the row below, moving north
        out[1:, :] += share[:-1, :] * legal[1:, :]    # from the row above, moving south
        out[:, :-1] += share[:, 1:] * legal[:, :-1]   # from the right, moving west
        out[:, 1:] += share[:, :-1] * legal[:, 1:]    # from the left, moving east
        self.mass = out
