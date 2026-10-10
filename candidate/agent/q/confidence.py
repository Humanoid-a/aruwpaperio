"""Q's confidence map: the whole belief as one full-board world state, plus how sure we are of each part.

For three items (enemy land, enemy trail, my land) every cell gets a probability:

    probability = the weighted vote of the hypotheses (votes.py)  +  a share for the residual mass R

and the residual's share depends on the reach region, the diamond around the last sighting whose radius is the
turns since then, minus the current view. Residual mass has no known history, so inside the reach region it
counts as a coin flip; outside it, it falls back on what memory says:

    item            from hypotheses                       residual share
    enemy land      sum of weights of those whose land    R x 0.5 in the reach region, else R x memory's value
                    contains the cell
    enemy trail     sum of weights of those whose trail   R x 0.5 in the reach region, else 0
                    contains the cell
    my land         sum of weights of those that still    R x 0.5 in the reach region where memory says mine,
                    have the cell as mine (= memory       else R x memory's value
                    minus what an enemy took from me)
    enemy position  the belief's position marginal        (already included)

Visible cells override everything with the observed 0 or 1. For an item, value = 1 if probability >= 0.5 else 0,
and confidence = |2 p - 1| (0 is a coin flip, 1 is certain). The enemy position uses the probability itself,
because when the belief is spread out every cell's value would be 0 with high confidence, which hides where the
enemy probably is. My position is included so a branch can relate uncertainty to where I am.

Eight channels on the 50 x 50 board, pooled in 5 x 5 blocks to 10 x 10:

    0-2  value: enemy land, enemy trail, my land           mean   (fraction of the block believed occupied)
    3-5  confidence of those three                         mean   (how sure we are about the block)
    6    enemy position probability                        sum    (probability the enemy is in the block)
    7    my position, one-hot                              sum

Value and confidence stay separate so that, after pooling, a block that is half enemy and half free but certain
still looks different from a block we know nothing about.
"""

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.cells import MY_TERRITORY, OPP_TERRITORY, OPP_TRAIL
from candidate.agent.data.observation import Observation
from candidate.agent.q.config import Q, QConfig
from candidate.agent.q.snapshot import Snapshot
from candidate.agent.runtime import TurnView

N_CHANNELS = 8
POOLED = BOARD_HEIGHT // Q.confidence_block


def reach_region(center: int, radius: int, obs: Observation) -> np.ndarray:
    """(50, 50) bool: the cells within `radius` steps (a diamond) of `center` that are not in the current view."""
    cy, cx = divmod(center, BOARD_WIDTH)
    ys, xs = np.mgrid[0:BOARD_HEIGHT, 0:BOARD_WIDTH]
    diamond = np.abs(ys - cy) + np.abs(xs - cx) <= radius
    diamond[obs.window_slices] = False
    return diamond


def confidence_board(
    obs: Observation,
    *,
    memory_my_land: np.ndarray,
    memory_opp_land: np.ndarray,
    land_votes: np.ndarray,
    trail_votes: np.ndarray,
    my_land_votes: np.ndarray,
    residual_mass: float,
    position: np.ndarray,
    reach: np.ndarray,
) -> np.ndarray:
    """The (8, 50, 50) float32 map, before pooling. Pure: everything it uses is an argument."""
    r = float(residual_mass)
    half = np.where(reach, 0.5, 0.0)
    p_land = land_votes + r * np.where(reach, 0.5, memory_opp_land)
    p_trail = trail_votes + r * half
    p_mine = my_land_votes + r * np.where(reach & memory_my_land, 0.5, memory_my_land)
    p_position = position.astype(np.float64).copy()

    rows, cols = obs.window_slices
    p_land[rows, cols] = obs.window_territory == OPP_TERRITORY
    p_trail[rows, cols] = obs.window_trail == OPP_TRAIL
    p_mine[rows, cols] = obs.window_territory == MY_TERRITORY
    p_position[rows, cols] = 0.0
    if obs.opp_seen_at is not None:
        p_position.flat[obs.opp_seen_at] = 1.0

    probabilities = [np.clip(p, 0.0, 1.0) for p in (p_land, p_trail, p_mine)]
    out = np.zeros((N_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    for i, p in enumerate(probabilities):
        out[i] = p >= 0.5
        out[3 + i] = np.abs(2.0 * p - 1.0)
    out[6] = p_position
    out[7].flat[obs.my_pos] = 1.0
    return out


def pool(board: np.ndarray, cfg: QConfig = Q) -> np.ndarray:
    """(8, 50, 50) -> (8, 10, 10): mean for the value and confidence channels, sum for the two position channels."""
    b = cfg.confidence_block
    blocks = board.reshape(N_CHANNELS, BOARD_HEIGHT // b, b, BOARD_WIDTH // b, b)
    pooled = np.empty((N_CHANNELS, BOARD_HEIGHT // b, BOARD_WIDTH // b), dtype=np.float32)
    pooled[:6] = blocks[:6].mean(axis=(2, 4))
    pooled[6:] = blocks[6:].sum(axis=(2, 4))
    return pooled


def build_confidence_board(view: TurnView, snap: Snapshot, cfg: QConfig = Q) -> np.ndarray:
    """The full-board map for a frame. The reach region is centred on the last sighting; before the first
    sighting the enemy is known to have started on its respawn, so that is the centre and the turns played
    are the radius."""
    obs, belief = view.obs, view.belief
    if view.opp_last_seen is not None:
        center, radius = view.opp_last_seen[0], obs.turn - view.opp_last_seen[1]
    else:
        center, radius = view.rules.opp.respawn, obs.turn
    reach = reach_region(center, radius, obs)
    return confidence_board(
        obs,
        memory_my_land=view.memory.territory == MY_TERRITORY,
        memory_opp_land=view.memory.territory == OPP_TERRITORY,
        land_votes=snap.land_votes, trail_votes=snap.trail_votes, my_land_votes=snap.my_land_votes,
        residual_mass=belief.detail.residual_mass, position=belief.position, reach=reach,
    )


def build_confidence(view: TurnView, snap: Snapshot, cfg: QConfig = Q) -> np.ndarray:
    """Q's third input: (8, 10, 10) float32."""
    return pool(build_confidence_board(view, snap, cfg), cfg)
