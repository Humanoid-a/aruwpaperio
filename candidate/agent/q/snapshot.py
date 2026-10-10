"""The per-frame snapshot: distance maps and safety margins, built once and shared by the inputs and the states.

build_snapshot() is a pure function of a TurnView (observation, memory, belief). It computes, in the brief's
terms, the quantities D1..D8:

    D1  enemy distance to my trail     worst case: taxicab from the possibility mask to my nearest trail cell
    D2  my distance to the enemy trail shortest path avoiding my trail to the nearest known enemy trail cell
    D3  enemy distance to its land     exact if it is visible, else averaged over the position marginal
    D4  my distance to my land         shortest path avoiding my trail
    D5  my gain if I return now        flood fill along my shortest way home, enemy cells counting double
    D6  enemy gain if it returns now   the same for the enemy, my cells counting double
    D7  safety margin                  D1 - D4  (+ margin_offset)
    D8  kill race                      D2 - D3

All distances are clipped to cfg.distance_cap, which also means "none". D5 and D6 are 0 without a trail.
The raw (unscaled, unclipped D7/D8) values are kept, because the training label needs them.
"""

from dataclasses import dataclass

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.data.cells import MY_TERRITORY, OPP_TERRITORY, OPP_TRAIL
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.distance import taxicab_distance
from candidate.agent.geometry.fill import BOARD
from candidate.agent.q import paths, votes
from candidate.agent.q.config import Q, QConfig
from candidate.agent.runtime import TurnView


@dataclass(frozen=True, eq=False)
class Snapshot:
    turn: int
    my_pos: int
    trail_len: int
    on_land: bool                     # my agent stands on my own territory
    my_trail_bb: np.ndarray           # (50,)
    my_land_bb: np.ndarray            # (50,) as memory has it
    opp_land_bb: np.ndarray           # (50,) as memory has it
    enemy_cell_bb: np.ndarray         # (50,) the visible enemy agent, or empty (it blocks my moves)
    passable_bb: np.ndarray           # (50,) cells I can walk through: not my trail, not the visible enemy
    possible: np.ndarray              # (50, 50) bool: the possibility mask
    enemy_arrival: np.ndarray         # (2500,) int16: worst-case turns for the enemy to reach each cell
    enemy_to_land: np.ndarray         # (2500,) int16: taxicab from each cell to the enemy's land
    enemy_seen: bool
    land_votes: np.ndarray            # (50, 50) float64: hypotheses' vote for enemy land
    trail_votes: np.ndarray           # (50, 50) float64: ... enemy trail
    my_land_votes: np.ndarray         # (50, 50) float64: ... my land
    known_trail_bb: np.ndarray        # (50,) enemy trail cells that are visible or likely
    hunt_targets_bb: np.ndarray       # (50,) the known ones I could step onto (not under the enemy agent)
    hunt_dist: np.ndarray | None      # (2500,) int16: distance from those cells through passable ones; None if none
    home_dist: np.ndarray             # (2500,) int16: distance to my land through passable cells
    home_path: list[int] | None       # shortest way home (cells to step onto); [] on land; None if none
    turns_since_sighting: int
    d1: int
    d2: int
    d3: float
    d4: int
    d5: int                           # gain in cells
    d6: int
    d7: float                         # raw, unclipped
    d8: float

    @property
    def exposed(self) -> bool:
        return self.trail_len > 0

    @property
    def enemy_trail_known(self) -> bool:
        return bool(self.known_trail_bb.any())


def _window_mask(obs, selector) -> np.ndarray:
    """A (50, 50) boolean from a function of the window layers."""
    out = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=bool)
    rows, cols = obs.window_slices
    out[rows, cols] = selector(obs)
    return out


def _clip(distance: int | None, cap: int) -> int:
    return cap if distance is None else min(int(distance), cap)


def _clip_map(raw: np.ndarray, cap: int) -> np.ndarray:
    return np.minimum(raw, cap).astype(np.int16)


def enemy_hypothesis(view: TurnView):
    """The top-weighted hypothesis as (pos, trail_bb, land_bb), or None when the belief holds only residual."""
    detail = view.belief.detail if view.belief is not None else None
    if detail is None or detail.n == 0:
        return None
    h = int(np.argmax(detail.weight))
    return int(detail.pos[h]), detail.trail_bb[h], detail.opp_lands[detail.opp_land_index[h]]


def build_snapshot(view: TurnView, cfg: QConfig = Q) -> Snapshot:
    obs, cap = view.obs, cfg.distance_cap
    belief = view.belief
    if belief is None or belief.detail is None:
        raise ValueError("the Q model needs a belief with detail (BeliefView.detail)")
    detail = belief.detail

    my_trail_bb = view.my_trail_bb
    my_land_bb = view.my_territory_bb
    opp_land_bb = bitboard.from_mask(view.memory.territory == OPP_TERRITORY)
    enemy_cell_bb = bitboard.from_cells([obs.opp_seen_at]) if obs.opp_seen_at is not None else bitboard.empty()
    passable = BOARD & ~my_trail_bb & ~enemy_cell_bb
    on_land = bitboard.has_cell(my_land_bb, obs.my_pos)
    trail_len = int(obs.my_trail.size)

    # ---- the enemy: where it could be, how fast it could get anywhere ---------------------------------------
    possible = belief.position > cfg.possible_min_probability
    if not possible.any():
        possible = ~_window_mask(obs, lambda o: np.ones(o.window_territory.shape, dtype=bool))
    possible_bb = bitboard.from_mask(possible)
    enemy_arrival = _clip_map(taxicab_distance(possible_bb).reshape(CELL_COUNT), cap)
    enemy_to_land = _clip_map(taxicab_distance(opp_land_bb).reshape(CELL_COUNT), cap)

    land_votes = votes.enemy_land_votes(detail)
    trail_votes = votes.enemy_trail_votes(detail)
    my_land_votes = votes.my_land_votes(detail)
    visible_enemy_trail = _window_mask(obs, lambda o: o.window_trail == OPP_TRAIL)
    known_trail_bb = bitboard.from_mask(visible_enemy_trail | (trail_votes > cfg.known_trail_probability))

    # ---- my side -----------------------------------------------------------------------------------------------
    home_dist = paths.distance_map(my_land_bb, passable)
    d4_raw = paths.distance_from(home_dist, obs.my_pos, my_land_bb)
    home_path = paths.path_from_map(home_dist, obs.my_pos) if my_land_bb.any() else None
    d4 = _clip(d4_raw, cap)

    if trail_len:
        arrival_on_trail = enemy_arrival[bitboard.to_cells(my_trail_bb)]
        d1 = int(arrival_on_trail.min())
    else:
        d1 = cap

    hunt_targets_bb = known_trail_bb & ~enemy_cell_bb
    if hunt_targets_bb.any():
        hunt_dist = paths.distance_map(hunt_targets_bb, passable)
        d2 = _clip(paths.distance_from(hunt_dist, obs.my_pos, hunt_targets_bb), cap)
    else:
        hunt_dist = None
        d2 = cap

    if obs.opp_seen_at is not None:
        d3 = float(enemy_to_land[obs.opp_seen_at])
    else:
        d3 = float((belief.position.astype(np.float64).reshape(-1) * enemy_to_land).sum())

    # ---- gains ---------------------------------------------------------------------------------------------------
    d5 = paths.best_return_gain(my_land_bb, opp_land_bb, my_trail_bb, home_dist, obs.my_pos) if trail_len else 0
    d6 = _enemy_gain(view, my_land_bb, opp_land_bb)

    if view.opp_last_seen is not None:
        since = obs.turn - view.opp_last_seen[1]
    else:
        since = obs.turn
    since = int(min(since, cfg.since_sighting_cap))

    return Snapshot(
        turn=obs.turn, my_pos=obs.my_pos, trail_len=trail_len, on_land=on_land,
        my_trail_bb=my_trail_bb, my_land_bb=my_land_bb, opp_land_bb=opp_land_bb, enemy_cell_bb=enemy_cell_bb,
        passable_bb=passable, possible=possible, enemy_arrival=enemy_arrival, enemy_to_land=enemy_to_land,
        enemy_seen=obs.opp_seen_at is not None, land_votes=land_votes, trail_votes=trail_votes,
        my_land_votes=my_land_votes, known_trail_bb=known_trail_bb, hunt_targets_bb=hunt_targets_bb, hunt_dist=hunt_dist, home_dist=home_dist, home_path=home_path,
        turns_since_sighting=since, d1=d1, d2=d2, d3=d3, d4=d4, d5=d5, d6=d6,
        d7=float(d1 - d4 + cfg.margin_offset), d8=float(d2) - d3,
    )


def _enemy_gain(view: TurnView, my_land_bb: np.ndarray, opp_land_bb: np.ndarray) -> int:
    """D6: what the enemy would capture by going straight home now, my cells counting double. 0 without a trail.
    Uses the top-weighted hypothesis (it already agrees with everything visible); with only residual, the
    visible enemy trail."""
    obs = view.obs
    hypothesis = enemy_hypothesis(view)
    if hypothesis is not None:
        pos, trail, land = hypothesis
    elif obs.opp_seen_at is not None:
        pos, land = obs.opp_seen_at, opp_land_bb
        mask = _window_mask(obs, lambda o: o.window_trail == OPP_TRAIL)
        trail = bitboard.from_mask(mask)
    else:
        return 0
    if not trail.any():
        return 0
    home = paths.distance_map(land, BOARD & ~trail)         # its own trail blocks it; mine does not
    return paths.best_return_gain(land, my_land_bb, trail, home, pos)
