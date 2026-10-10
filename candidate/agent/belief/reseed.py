"""Building a belief from nothing but an observation.

The ordinary start of a match is belief.state.start_state(), which knows everything.
This is for the other case: every hypothesis was contradicted by an observation (or
turns were skipped, or the controller was switched on in the middle of a match), so
the belief starts again from what is certain now.
"""

import numpy as np

from candidate.agent.belief.state import BeliefState
from candidate.agent.belief.cut import in_window
from candidate.agent.config import BOARD_WIDTH, CELL_COUNT
from candidate.agent.data.cells import OPP_TERRITORY, OPP_TRAIL
from candidate.agent.data.hypotheses import HypothesisStore
from candidate.agent.data.observation import Observation
from candidate.agent.data.residual import ResidualGrid
from candidate.agent.data.territory import TerritoryDiffs
from candidate.agent.geometry import bitboard


def reseed(state: BeliefState, obs: Observation, opp_base_new: np.ndarray) -> None:
    """Rebuild the belief from this observation.

    If I can see the opponent there is one hypothesis: where it is, with the trail cells I can see if it is out
    of its land, its land as memory has it, and the engine's count. If not, there are no hypotheses, only the
    residual spread over every cell it could be on: outside my window, and within reach of the last sighting."""
    state.opp_diffs = TerritoryDiffs(state.keys)
    state.my_diffs = TerritoryDiffs(state.keys)
    state.residual = ResidualGrid()
    state.opp_base = opp_base_new
    seen = obs.opp_seen_at
    if seen is not None:
        on_own_land = obs.window_cell(seen)[0] == OPP_TERRITORY
        trail = bitboard.empty()
        if not on_own_land:
            y0, x0 = obs.window_origin
            for dy, dx in np.argwhere((obs.window_trail == OPP_TRAIL) & ~obs.hidden_mask):
                bitboard.set_cell(trail, (y0 + int(dy)) * BOARD_WIDTH + x0 + int(dx))
            bitboard.set_cell(trail, seen)
        state.store = state.single(seen, opp_base_new, trail, obs.opp_count)
        return
    state.store = HypothesisStore.empty()
    cells = np.arange(CELL_COUNT, dtype=np.int64)
    allowed = ~in_window(cells, obs)
    allowed[obs.my_pos] = False
    if state.sighting is not None:
        sy, sx = divmod(state.sighting[0], BOARD_WIDTH)
        y, x = np.divmod(cells, BOARD_WIDTH)
        reach = abs(y - sy) + abs(x - sx) <= max(1, obs.turn - state.sighting[1])
        if (allowed & reach).any():
            allowed &= reach
    if not allowed.any():
        allowed = np.ones(CELL_COUNT, dtype=bool)
        allowed[obs.my_pos] = False
    state.residual.mass = (allowed / allowed.sum()).reshape(-1, BOARD_WIDTH).astype(np.float64)
