"""One-move lookahead from the controller's own point of view.

predict_move() says what the engine will do with a move, using only what the
controller can see: the destination is always an adjacent cell, so it is
always inside the visible window. It uses the same rules as rules.reference,
specialised to the one fact the controller does not have to guess, namely its
own trail, and it asks the memory only for the shape of its own territory (to
size a capture).

The runtime compares every prediction with the result the framework actually
returns, which is a live check of this module, the adapter and the rules.
"""

from dataclasses import dataclass

import numpy as np

from candidate.agent.data.cells import MY_TERRITORY, MY_TRAIL, OPP_TRAIL
from candidate.agent.data.observation import Observation
from candidate.agent.data.types import Direction, MoveResult
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.fill import capture_region
from candidate.agent.geometry.grid import step
from candidate.agent.rules.match import MatchRules


@dataclass(frozen=True, eq=False)
class Prediction:
    direction: Direction
    dest: int | None                # the cell stepped onto; None if off the board
    result: MoveResult
    killed: bool                    # the step cuts the opponent's trail
    new_pos: int                    # where my agent will be
    new_trail: tuple[int, ...]      # my trail afterwards, oldest first
    region: np.ndarray | None       # CAPTURED: bitboard of the newly enclosed cells
    gain: int | None                # CAPTURED: expected change in my territory count
    respawn_displaced: bool         # DIED: the opponent is on my respawn cell


def predict_move(obs: Observation, my_territory: np.ndarray, rules: MatchRules, direction: Direction) -> Prediction:
    """`my_territory` is a single bitboard of the cells believed to be mine
    (normally MemoryMap.bitboards().my_territory)."""
    trail = tuple(int(c) for c in obs.my_trail)
    dest = step(obs.my_pos, direction)
    if dest is None or dest == obs.opp_seen_at:
        return Prediction(direction, dest, MoveResult.INVALID, False, obs.my_pos, trail, None, None, False)

    territory_code, trail_code = obs.window_cell(dest)

    if trail_code == MY_TRAIL:
        displaced = obs.opp_seen_at == rules.me.respawn
        new_pos = rules.me.alt_respawn if displaced else rules.me.respawn
        return Prediction(direction, dest, MoveResult.DIED, False, new_pos, (), None, None, displaced)

    killed = trail_code == OPP_TRAIL
    if territory_code == MY_TERRITORY and trail:
        trail_bb = bitboard.from_cells(trail)
        region = capture_region(my_territory, trail_bb)
        gain = len(trail) + bitboard.popcount(region)
        return Prediction(direction, dest, MoveResult.CAPTURED, killed, dest, (), region, gain, False)

    new_trail = trail if territory_code == MY_TERRITORY else trail + (dest,)
    return Prediction(direction, dest, MoveResult.NORMAL, killed, dest, new_trail, None, None, False)
