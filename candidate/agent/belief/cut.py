"""Step 2 of a frame: cut what the game rules and this observation make impossible.

Evidence removes a hypothesis only when it is contradicted outright. It never
reweights: that is K's job, later, and K never rules anything out. If the model
of the game is right, the true state is therefore never removed, at worst pruned
into the residual, which keeps its position.

What the observation says, and where it is applied:

  move level   (before any child is built, so nothing is copied for a ruled-out move)
    * position   I can see the opponent on a cell: only moves that end there survive.
                 I cannot: only moves that end outside my window survive.
    * my death   I was killed by the opponent this turn: only moves onto my trail
                 survive. I was not: those moves are out.

  child level  (needs the child, because the evidence is about its history)
    * its count  the child's territory count must equal the engine's count for it.
    * the window the child's trail and land must agree with every cell I can see
                 (the trail under an agent is hidden by the framework and is not compared).
    * my count   the engine's count for me after my move, minus what this child's move takes
                 from its view of my land, must equal the count I see now. That needs the view
                 to add up to the engine's count to begin with (`my_exact`); a hypothesis whose
                 view does not is not judged by it. When the opponent has killed me my whole land
                 is gone and my count is the start size for every hypothesis, so there is nothing
                 to tell them apart by and nothing is checked.

  residual     position and my death only: it has no history to test.
"""

from dataclasses import dataclass

import numpy as np

from candidate.agent.belief.generate import Children, Expansion, Situation, final_positions
from candidate.agent.belief.state import BeliefState, BeliefStats
from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.data.cells import OPP_TERRITORY, OPP_TRAIL
from candidate.agent.data.observation import Observation
from candidate.agent.geometry import bitboard


@dataclass(frozen=True, eq=False)
class Evidence:
    """What this frame's observation says about the opponent's move."""

    obs: Observation
    killed_by_opponent: bool        # I died this turn by the opponent stepping onto my trail


def _rows(mask: np.ndarray) -> np.ndarray:
    """(h, w) boolean -> (h,) int64, bit x of row y set where mask[y, x]."""
    return (mask.astype(np.int64) << np.arange(mask.shape[1], dtype=np.int64)).sum(axis=1)


def in_window(cells: np.ndarray, obs: Observation) -> np.ndarray:
    y0, x0 = obs.window_origin
    h, w = obs.window_territory.shape
    y, x = np.divmod(cells, BOARD_WIDTH)
    return (y >= y0) & (y < y0 + h) & (x >= x0) & (x < x0 + w)


def position_ok(cells: np.ndarray, obs: Observation) -> np.ndarray:
    """Is the opponent allowed to be on these cells, given whether I can see it?"""
    if obs.opp_seen_at is not None:
        return cells == obs.opp_seen_at
    return ~in_window(cells, obs)


# ---- move level -------------------------------------------------------------------------------------------


def cut_moves(exp: Expansion, ev: Evidence, stats: BeliefStats) -> np.ndarray:
    """(n, 4) boolean: the moves that are legal in the game and agree with where the opponent is seen (or not
    seen) and with whether I was killed."""
    batch = exp.batch
    wanted = batch.kills_me if ev.killed_by_opponent else ~batch.kills_me
    keep = batch.legal & position_ok(final_positions(exp), ev.obs) & wanted
    stats.cut_by_position += int(batch.legal.sum() - keep.sum())
    return keep


def position_ok_unmoved(exp: Expansion, ev: Evidence) -> np.ndarray:
    """Indices of the hypotheses that could be where they are now, if the opponent has not moved at all."""
    return np.nonzero(position_ok(exp.store.pos.astype(np.int64), ev.obs))[0]


# ---- child level ------------------------------------------------------------------------------------------


def window_ok(kids: Children, sit: Situation) -> np.ndarray:
    """Do each child's trail and land agree with every cell I can see?"""
    store, obs = kids.store, sit.obs
    if store.n == 0:
        return np.zeros(0, dtype=bool)
    y0, x0 = obs.window_origin
    h, w = obs.window_territory.shape
    wmask = np.int64((1 << w) - 1)
    certain = _rows(~obs.hidden_mask)
    seen_trail = _rows((obs.window_trail == OPP_TRAIL) & ~obs.hidden_mask)
    predicted = (store.trail_bb[:, y0:y0 + h] >> x0) & wmask
    trail_ok = (((predicted ^ seen_trail) & certain) == 0).all(axis=1)

    seen_land = _rows(obs.window_territory == OPP_TERRITORY)
    unique, inverse = np.unique(store.opp_land_id, return_inverse=True)
    diffs = sit.opp_diffs
    land = (sit.opp_base[y0:y0 + h] & ~diffs.removed[unique, y0:y0 + h]) | diffs.added[unique, y0:y0 + h]
    land_ok = ((((land >> x0) & wmask) ^ seen_land) == 0).all(axis=1)
    return trail_ok & land_ok[inverse.reshape(-1)]


def cut_children(kids: Children, sit: Situation, ev: Evidence, stats: BeliefStats) -> np.ndarray:
    """(m,) boolean: the children whose history agrees with both territory counts and with every visible cell."""
    obs = ev.obs
    ok = kids.store.opp_count == obs.opp_count
    stats.cut_by_count += int((~ok).sum())
    window = window_ok(kids, sit)
    stats.cut_by_window += int((ok & ~window).sum())
    ok &= window
    if sit.my_count is not None and not ev.killed_by_opponent:
        mine = ~kids.my_exact | (sit.my_count - kids.my_loss == obs.my_count)
        stats.cut_by_my_count += int((ok & ~mine).sum())
        ok &= mine
    return ok


# ---- the residual -------------------------------------------------------------------------------------------


def gate_residual(state: BeliefState, sit: Situation, ev: Evidence) -> None:
    """Zero the residual where the opponent cannot be: wrong for where it is (not) seen, or wrong for whether
    it stepped onto my trail."""
    cells = np.arange(CELL_COUNT, dtype=np.int64).reshape(BOARD_HEIGHT, BOARD_WIDTH)
    ok = position_ok(cells, ev.obs)
    on_my_trail = np.zeros(CELL_COUNT, dtype=bool)
    on_my_trail[bitboard.to_cells(sit.my_trail_bb)] = True
    on_my_trail = on_my_trail.reshape(BOARD_HEIGHT, BOARD_WIDTH)
    ok &= on_my_trail if ev.killed_by_opponent else ~on_my_trail
    state.residual.mass = np.where(ok, state.residual.mass, 0.0)
