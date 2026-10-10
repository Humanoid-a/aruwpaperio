"""WorldState: one complete, true board from "my" point of view.

This is the dense, easy-to-read form used for reference transitions, oracle
tests, and logged ground truth. The controller never sees a true WorldState
during a match (it only gets an Observation); self-play builds one from the
engine for the log. Fast paths use the bitboard forms from bitboards().

Representation
    territory   (50, 50) int8, [y][x]: FREE / MY_TERRITORY / OPP_TERRITORY
    trail       (50, 50) int8, [y][x]: FREE / MY_TRAIL / OPP_TRAIL
    my_pos, opp_pos           flat indices (see geometry.grid)
    my_trail, opp_trail       flat indices in walking order, oldest first
    my_count, opp_count       territory sizes

Two layers rather than the single cell code of data.cells.combine(), because
the engine lets a trail lie on top of the other player's territory.

Trails include the cell the agent stands on as their last element whenever
they are non-empty (the engine sets the position and appends the same cell).
An empty trail does not mean the agent is on its own land: after the other
player captures a region around it, an agent can stand on enemy territory
with no trail.
"""

from dataclasses import dataclass

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.data.cells import (
    FREE,
    MY_TERRITORY,
    MY_TRAIL,
    OPP_TERRITORY,
    OPP_TRAIL,
    LayerBitboards,
    combine,
    layer_bitboards,
    layer_problems,
)
from candidate.agent.geometry.grid import manhattan


@dataclass(eq=False)
class WorldState:
    territory: np.ndarray
    trail: np.ndarray
    my_pos: int
    opp_pos: int
    my_trail: list[int]
    opp_trail: list[int]
    my_count: int
    opp_count: int

    @classmethod
    def empty(cls, my_pos: int, opp_pos: int) -> "WorldState":
        """A blank board with both agents placed and no territory or trails."""
        shape = (BOARD_HEIGHT, BOARD_WIDTH)
        return cls(
            np.full(shape, FREE, dtype=np.int8),
            np.full(shape, FREE, dtype=np.int8),
            my_pos,
            opp_pos,
            [],
            [],
            0,
            0,
        )

    @property
    def cells(self) -> np.ndarray:
        """The single-code view (trail over territory). Lossy; for display."""
        return combine(self.territory, self.trail)

    def bitboards(self) -> LayerBitboards:
        """The four layers as single-board bitboards."""
        return layer_bitboards(self.territory, self.trail)

    def copy(self) -> "WorldState":
        return WorldState(
            self.territory.copy(),
            self.trail.copy(),
            self.my_pos,
            self.opp_pos,
            list(self.my_trail),
            list(self.opp_trail),
            self.my_count,
            self.opp_count,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorldState):
            return NotImplemented
        return (
            np.array_equal(self.territory, other.territory)
            and np.array_equal(self.trail, other.trail)
            and self.my_pos == other.my_pos
            and self.opp_pos == other.opp_pos
            and self.my_trail == other.my_trail
            and self.opp_trail == other.opp_trail
            and self.my_count == other.my_count
            and self.opp_count == other.opp_count
        )

    def problems(self, *, check_counts: bool = True) -> list[str]:
        """Every broken invariant, as readable messages; empty when sound.
        Pass check_counts=False for a world assembled from memory, where
        unseen cells are guesses and the stored counts are the better source."""
        out = layer_problems(self.territory, self.trail, (BOARD_HEIGHT, BOARD_WIDTH), allow_unknown=False)
        if out:
            return out
        for name, pos in (("my_pos", self.my_pos), ("opp_pos", self.opp_pos)):
            if not 0 <= pos < CELL_COUNT:
                out.append(f"{name} {pos} is off the board")
        if out:
            return out
        if self.my_pos == self.opp_pos:
            out.append("both agents are on the same cell")
        if check_counts:
            for name, stored, code in (
                ("my_count", self.my_count, MY_TERRITORY),
                ("opp_count", self.opp_count, OPP_TERRITORY),
            ):
                actual = int((self.territory == code).sum())
                if stored != actual:
                    out.append(f"{name} is {stored} but the board has {actual} such cells")
        out += self._trail_problems("my", self.my_trail, self.my_pos, MY_TRAIL, MY_TERRITORY)
        out += self._trail_problems("opp", self.opp_trail, self.opp_pos, OPP_TRAIL, OPP_TERRITORY)
        return out

    def validate(self, *, check_counts: bool = True) -> None:
        problems = self.problems(check_counts=check_counts)
        if problems:
            raise ValueError("invalid WorldState: " + "; ".join(problems))

    def _trail_problems(
        self, owner: str, trail: list[int], pos: int, trail_code: int, own_territory_code: int
    ) -> list[str]:
        name = f"{owner}_trail"
        if any(not 0 <= c < CELL_COUNT for c in trail):
            return [f"{name} has a cell off the board"]
        out: list[str] = []
        if len(set(trail)) != len(trail):
            out.append(f"{name} repeats a cell")
        for a, b in zip(trail, trail[1:]):
            if manhattan(a, b) != 1:
                out.append(f"{name} jumps from {a} to {b}, which are not adjacent")
                break
        if set(np.flatnonzero(self.trail == trail_code).tolist()) != set(trail):
            out.append(f"{name} does not match the cells marked {owner} trail on the board")
        if trail and trail[-1] != pos:
            out.append(f"{name} does not end at its agent's cell")
        if any(self.territory.flat[c] == own_territory_code for c in trail):
            out.append(f"{name} lies on its owner's own territory")
        return out
