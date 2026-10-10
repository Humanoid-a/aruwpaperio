"""The static facts about a match that the move rules need.

The board contents change every turn (WorldState / BitWorld). What does not
change is where each side respawns and which land it gets back when it dies.
A MatchRules is built once per match from the two respawn cells; the
controller knows its own from the API and has to assume the opponent's (the
default config mirrors it across the board, see MatchRules.mirrored).
"""

from dataclasses import dataclass

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT, STARTING_TERRITORY_SIZE
from candidate.agent.data.cells import MY_TERRITORY, OPP_TERRITORY
from candidate.agent.data.types import Side
from candidate.agent.data.world import WorldState
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat, in_bounds, manhattan, to_yx


def starting_cells(respawn: int, size: int = STARTING_TERRITORY_SIZE) -> tuple[int, ...]:
    """The size x size square of cells centred on `respawn`, clipped to the
    board, in row-major order (the framework's starting territory)."""
    cy, cx = to_yx(respawn)
    half = size // 2
    return tuple(
        flat(y, x)
        for y in range(cy - half, cy + half + 1)
        for x in range(cx - half, cx + half + 1)
        if in_bounds(y, x)
    )


def _alternate_respawn(respawn: int, start_cells: tuple[int, ...]) -> int:
    """Where an agent respawns when the other agent is standing on its
    respawn cell: the nearest other cell of its starting territory, or if it
    has none, the nearest other cell of the board. Ties go to the smaller
    (y, x), exactly as the engine does."""

    def key(c: int) -> tuple[int, int, int]:
        y, x = to_yx(c)
        return manhattan(c, respawn), y, x

    candidates = [c for c in start_cells if c != respawn] or [c for c in range(CELL_COUNT) if c != respawn]
    return min(candidates, key=key)


@dataclass(frozen=True, eq=False)
class RoleRules:
    """Respawn facts for one side."""

    respawn: int
    start_cells: tuple[int, ...]
    start_bb: np.ndarray        # the starting territory as a single bitboard (read-only)
    alt_respawn: int            # used instead of `respawn` while the other agent stands on it

    @classmethod
    def build(cls, respawn: int, size: int = STARTING_TERRITORY_SIZE) -> "RoleRules":
        cells = starting_cells(respawn, size)
        start_bb = bitboard.from_cells(cells)
        start_bb.flags.writeable = False
        return cls(respawn, cells, start_bb, _alternate_respawn(respawn, cells))


@dataclass(frozen=True, eq=False)
class MatchRules:
    me: RoleRules
    opp: RoleRules

    def __post_init__(self) -> None:
        if set(self.me.start_cells) & set(self.opp.start_cells):
            raise ValueError("the two starting territories overlap")

    def role(self, side: Side) -> RoleRules:
        return self.me if side == Side.ME else self.opp

    @classmethod
    def from_respawns(
        cls, my_respawn: int, opp_respawn: int, size: int = STARTING_TERRITORY_SIZE
    ) -> "MatchRules":
        return cls(RoleRules.build(my_respawn, size), RoleRules.build(opp_respawn, size))

    @classmethod
    def mirrored(cls, my_respawn: int, size: int = STARTING_TERRITORY_SIZE) -> "MatchRules":
        """Assume the opponent respawns at the left-right mirror image of mine.
        The default config does ((4, 25) and (45, 25)), but nothing in the API
        confirms it; treat it as an assumption."""
        y, x = to_yx(my_respawn)
        return cls.from_respawns(my_respawn, flat(y, BOARD_WIDTH - 1 - x), size)

    def initial_world(self) -> WorldState:
        """The board at the start of a match: each side on its own starting territory."""
        world = WorldState.empty(self.me.respawn, self.opp.respawn)
        for c in self.me.start_cells:
            world.territory.flat[c] = MY_TERRITORY
        for c in self.opp.start_cells:
            world.territory.flat[c] = OPP_TERRITORY
        world.my_count = len(self.me.start_cells)
        world.opp_count = len(self.opp.start_cells)
        return world
