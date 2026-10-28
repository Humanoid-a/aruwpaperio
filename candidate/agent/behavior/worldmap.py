"""The world map: the whole belief as per-cell probabilities over the full world state (docs/Q_Probability_Map_Input.md).

Built once per frame by the snapshot (Snapshot.world). The states read it (HUNT goes for likely enemy trail, TRACK
TARGET for what is uncertain) and Q's third input is it, pooled (q/probmap.py).

For every cell, two groups of three mutually exclusive categories, each summing to 1 at the cell:

    ownership   free, mine, enemy
    trail       none, my trail, enemy trail       (two trails never share a cell: stepping on one kills its owner)

and two position layers, each summing to 1 over the board: the enemy's position (the belief's position marginal) and
mine (one-hot). Layers rather than one value with a confidence, because the layers keep the fact that the categories
exclude each other: a cell at 0.45 mine / 0.45 enemy / 0.10 free reads as a coin flip between mine and enemy, not as
free.

How a cell's probabilities are made:

  * hypotheses: a category's probability is the total weight of the hypotheses that put the cell in it (votes.py).
    Each hypothesis picks exactly one category per group (its land is enemy, its view of my land is mine, the rest is
    free; its trail is enemy trail), so the layers add up.
  * residual mass (weight whose history was pruned) uses the reach region: the diamond around the last sighting whose
    radius is the turns since then, minus the current view.
        ownership, inside it, by what memory says: free -> 1/2 free, 1/2 enemy; mine -> 1/2 mine, 1/2 enemy;
                   enemy -> enemy. Outside it: memory's category (a cell never seen counts as free).
        trail, inside it: 1/2 none, 1/2 enemy trail (none on my trail cells). Outside it: none.
  * my trail is always exact (I know it), and the enemy trail is 0 on it.
  * cells in view are what is seen: one-hot.

Confidence of each group at a cell: 1 - entropy / log 3, which is 1 for a certain cell and 0 for an even three-way
split. Pooling averages, so without it a block half certainly mine and half certainly free would look like a block of
50/50 guesses.

Ten channels on the 50 x 50 board:

    0-2  ownership: free, mine, enemy
    3    ownership confidence
    4-6  trail: none, my trail, enemy trail
    7    trail confidence
    8    enemy position probability
    9    my position
"""

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.cells import MY_TERRITORY, MY_TRAIL, OPP_TERRITORY, OPP_TRAIL
from candidate.agent.data.observation import Observation

N_CHANNELS = 10
FREE, MINE, ENEMY, OWN_CONF, NONE, MY_TRAIL_CH, ENEMY_TRAIL, TRAIL_CONF, ENEMY_POS, MY_POS = range(N_CHANNELS)
_LOG3 = float(np.log(3.0))


def reach_region(center: int, radius: int, obs: Observation) -> np.ndarray:
    """(50, 50) bool: the cells within `radius` steps (a diamond) of `center` that are not in the current view."""
    cy, cx = divmod(center, BOARD_WIDTH)
    ys, xs = np.mgrid[0:BOARD_HEIGHT, 0:BOARD_WIDTH]
    diamond = np.abs(ys - cy) + np.abs(xs - cx) <= radius
    diamond[obs.window_slices] = False
    return diamond


def confidence(p: np.ndarray) -> np.ndarray:
    """1 - entropy / log 3 over the first axis (three probabilities per cell that sum to 1)."""
    p = np.clip(p, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return 1.0 + terms.sum(axis=0) / _LOG3


def probability_board(
    obs: Observation,
    *,
    memory_territory: np.ndarray,
    land_votes: np.ndarray,
    trail_votes: np.ndarray,
    my_land_votes: np.ndarray,
    hypothesis_mass: float,
    residual_mass: float,
    position: np.ndarray,
    reach: np.ndarray,
) -> np.ndarray:
    """The (10, 50, 50) float32 map, before pooling. Pure: everything it uses is an argument. The votes and masses are
    in the belief's own units; they are normalised here so that hypotheses and residual together weigh 1."""
    total = float(hypothesis_mass + residual_mass)
    scale = 1.0 / total if total > 0 else 0.0
    h, r = (hypothesis_mass * scale, residual_mass * scale) if total > 0 else (0.0, 1.0)   # no mass: nothing known
    enemy_h = np.clip(land_votes * scale, 0.0, h)
    mine_h = np.clip(my_land_votes * scale, 0.0, h - enemy_h)            # a cell a hypothesis calls its land is not mine
    free_h = h - enemy_h - mine_h

    remembered_mine = memory_territory == MY_TERRITORY
    remembered_enemy = memory_territory == OPP_TERRITORY
    remembered_free = ~remembered_mine & ~remembered_enemy                # free, or never seen
    half = 0.5 * r
    free = free_h + np.where(reach, np.where(remembered_free, half, 0.0), np.where(remembered_free, r, 0.0))
    mine = mine_h + np.where(reach, np.where(remembered_mine, half, 0.0), np.where(remembered_mine, r, 0.0))
    enemy = enemy_h + np.where(reach, np.where(remembered_enemy, r, half), np.where(remembered_enemy, r, 0.0))

    my_trail = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float64)
    my_trail.flat[np.asarray(obs.my_trail, dtype=np.int64)] = 1.0
    on_my_trail = my_trail > 0
    enemy_trail = np.clip(trail_votes * scale, 0.0, h) + np.where(reach, half, 0.0)
    enemy_trail[on_my_trail] = 0.0
    none = 1.0 - my_trail - enemy_trail

    rows, cols = obs.window_slices                                        # what is in view is what is seen
    terr, trail = obs.window_territory, obs.window_trail
    free[rows, cols] = (terr != MY_TERRITORY) & (terr != OPP_TERRITORY)
    mine[rows, cols] = terr == MY_TERRITORY
    enemy[rows, cols] = terr == OPP_TERRITORY
    none[rows, cols] = (trail != MY_TRAIL) & (trail != OPP_TRAIL)
    my_trail[rows, cols] = trail == MY_TRAIL
    enemy_trail[rows, cols] = trail == OPP_TRAIL

    ownership = _normalised(np.stack([free, mine, enemy]))
    trails = _normalised(np.stack([none, my_trail, enemy_trail]))
    p_position = position.astype(np.float64).copy()
    p_position[rows, cols] = 0.0
    if obs.opp_seen_at is not None:
        p_position.flat[obs.opp_seen_at] = 1.0

    out = np.empty((N_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    out[FREE:ENEMY + 1] = ownership
    out[OWN_CONF] = confidence(ownership)
    out[NONE:ENEMY_TRAIL + 1] = trails
    out[TRAIL_CONF] = confidence(trails)
    out[ENEMY_POS] = p_position
    out[MY_POS] = 0.0
    out[MY_POS].flat[obs.my_pos] = 1.0
    return out


def _normalised(p: np.ndarray) -> np.ndarray:
    """Clip to [0, 1] and make the three layers sum to exactly 1 at every cell (rounding only; they already do)."""
    p = np.clip(p, 0.0, None)
    return p / np.maximum(p.sum(axis=0, keepdims=True), 1e-12)


def world_map(view, land_votes: np.ndarray, trail_votes: np.ndarray, my_land_votes: np.ndarray) -> np.ndarray:
    """The map for a frame (`view` is a TurnView; the votes are the snapshot's). The reach region is centred on the last sighting; before the first sighting
    the enemy is known to have started on its respawn, so that is the centre and the turns played are the radius."""
    obs, belief = view.obs, view.belief
    if view.opp_last_seen is not None:
        center, radius = view.opp_last_seen[0], obs.turn - view.opp_last_seen[1]
    else:
        center, radius = view.rules.opp.respawn, obs.turn
    detail = belief.detail
    return probability_board(
        obs,
        memory_territory=view.memory.territory,
        land_votes=land_votes, trail_votes=trail_votes, my_land_votes=my_land_votes,
        hypothesis_mass=float(detail.weight.sum()), residual_mass=detail.residual_mass, position=belief.position,
        reach=reach_region(center, radius, obs),
    )
