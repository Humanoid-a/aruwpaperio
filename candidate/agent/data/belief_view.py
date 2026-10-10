"""What the belief publishes each turn: a read-only snapshot a strategy can use."""

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH


class BeliefEvent(Enum):
    SIGHTED = auto()          # the opponent is in view
    LOST_SIGHT = auto()       # it was in view last turn and is not now
    RESEEDED = auto()         # every hypothesis was contradicted, so the belief was rebuilt
    OPPONENT_RESET = auto()   # I cut its trail: it is certainly at its respawn


@dataclass(frozen=True)
class BeliefSummary:
    """What the belief holds and what this frame's evidence did to it, as plain numbers for describing it.
    Computed once per frame when the view is published."""

    # what is stored
    opp_counts: tuple[int, ...]         # the distinct opponent territory counts among the hypotheses
    exposed_share: float | None         # share of hypothesis mass with a trail out (None: no hypotheses)
    trail_mean: float                   # mean trail length, by mass
    trail_longest: int
    opp_land_known: float               # share of mass whose land cells add up to its count (the rest has unseen land)
    my_views: int                       # distinct views of my land across the hypotheses
    my_views_exact: float               # share of mass whose view of my land has exactly the engine's count of cells
    my_count: int                       # the engine's count for me
    opp_land_diffs: int                 # entries in the opponent's territory-difference table
    my_land_diffs: int                  # ... in mine
    # what this frame did (since the previous view: my move at the end of last frame, then this observation)
    successors: int                     # moves tried that position evidence did not rule out
    cut_by_position: int
    cut_by_count: int
    cut_by_window: int
    cut_by_my_count: int
    merged: int
    pruned_mass: float                  # probability moved into the residual by pruning
    rehydrated: int
    resets: int


@dataclass(frozen=True, eq=False)
class BeliefDetail:
    """The whole belief as plain arrays, for consumers that need more than the position marginal (the Q
    model's confidence map). Hypotheses carry their lands as an index into a small table of distinct lands,
    because many hypotheses share one."""

    weight: np.ndarray            # (n,) float64: probability mass of each hypothesis
    pos: np.ndarray               # (n,) int32: the opponent's cell under each
    trail_bb: np.ndarray          # (n, 50) int64: its trail
    opp_land_index: np.ndarray    # (n,) int64: row of opp_lands
    opp_lands: np.ndarray         # (u, 50) int64: the distinct enemy lands
    my_land_index: np.ndarray     # (n,) int64: row of my_lands
    my_lands: np.ndarray          # (v, 50) int64: the distinct views of my land
    residual: np.ndarray          # (50, 50) float64: mass whose history is unknown, per cell

    @property
    def n(self) -> int:
        return int(self.weight.shape[0])

    @property
    def residual_mass(self) -> float:
        return float(self.residual.sum())

    @classmethod
    def empty(cls) -> "BeliefDetail":
        z = np.zeros
        return cls(z(0), z(0, dtype=np.int32), z((0, BOARD_HEIGHT), dtype=np.int64), z(0, dtype=np.int64),
                   z((0, BOARD_HEIGHT), dtype=np.int64), z(0, dtype=np.int64), z((0, BOARD_HEIGHT), dtype=np.int64),
                   z((BOARD_HEIGHT, BOARD_WIDTH)))


@dataclass(frozen=True, eq=False)
class BeliefView:
    """A snapshot of the belief for one turn. Read-only; stays valid after the
    next update because the filter builds new arrays each time."""

    turn: int
    position: np.ndarray            # (50, 50) float32: probability the opponent is on each cell
    support: np.ndarray             # (50, 50) bool: cells with non-zero probability
    n_hypotheses: int
    hypothesis_mass: float          # share of the belief that still has a history
    residual_mass: float            # share whose history is unknown
    entropy: float                  # of the position marginal, in nats
    sighted_at: int | None          # the cell the opponent is seen on this turn
    events: tuple[BeliefEvent, ...]
    _pos: np.ndarray = field(repr=False)
    _weight: np.ndarray = field(repr=False)
    _trail_bb: np.ndarray = field(repr=False)
    _trail_len: np.ndarray = field(repr=False)
    _opp_count: np.ndarray = field(repr=False)
    summary: "BeliefSummary | None" = None   # what the belief holds and what this frame did to it
    detail: "BeliefDetail | None" = None     # the hypotheses and the residual as arrays

    def probability_at(self, cell: int) -> float:
        return float(self.position.reshape(-1)[cell])

    def top_positions(self, k: int = 5) -> list[tuple[int, float]]:
        """The k most probable cells as (flat cell, probability), most probable first, ties
        broken by cell index. Cells with no probability are never listed, so there can be
        fewer than k (a single one when the opponent is in view)."""
        flat = self.position.reshape(-1)
        possible = int(self.support.sum())
        k = min(k, possible)
        if k <= 0:
            return []
        order = np.lexsort((np.arange(flat.size), -flat))[:k]
        return [(int(c), float(flat[c])) for c in order]

    def most_likely(self) -> tuple[int, float]:
        """(cell, probability) of the single most probable position."""
        flat = self.position.reshape(-1)
        cell = int(np.argmax(flat))
        return cell, float(flat[cell])

    def trail_marginal(self) -> np.ndarray:
        """(50, 50) float32: the weight of hypotheses whose trail covers each
        cell. In units of belief mass, so it sums over cells to at most
        hypothesis_mass times the trail length; the residual says nothing
        about trails and contributes nothing."""
        out = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float64)
        if self._weight.shape[0]:
            for x in range(BOARD_WIDTH):
                out[:, x] = (((self._trail_bb >> x) & 1) * self._weight[:, None]).sum(axis=0)
        return out.astype(np.float32)

    def exposed_probability(self) -> float | None:
        """Among hypotheses with a history, the share that have a trail out; None if there are none."""
        if self._weight.shape[0] == 0 or self.hypothesis_mass <= 0:
            return None
        return float(self._weight[self._trail_len > 0].sum() / self._weight.sum())

    def count_distribution(self) -> dict[int, float]:
        """Territory size -> share of hypothesis mass."""
        out: dict[int, float] = {}
        total = float(self._weight.sum())
        for count, weight in zip(self._opp_count.tolist(), self._weight.tolist()):
            out[count] = out.get(count, 0.0) + weight / total if total > 0 else 0.0
        return out
