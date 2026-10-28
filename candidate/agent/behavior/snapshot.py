"""The per-frame snapshot: distance maps and safety margins, built once and shared by the inputs and the states.

build_snapshot() is a pure function of a TurnView (observation, memory, belief). It first builds the world map
(worldmap.py: the whole belief as per-cell probabilities, full resolution) and reads D1..D8 from it, using only the
cells whose probability is at least a threshold (BehaviorConfig):

    enemy position  P >= position_threshold (1e-4)      the cells the enemy is likely to be on
    enemy trail     P >= trail_threshold (0.5)          its trail, as far as it is likely
    enemy land      P(enemy) >= land_threshold (0.5)    my land: P(mine) >= land_threshold

    D1  enemy distance to my trail     taxicab from the likely enemy cells to my nearest trail cell
    D2  my distance to the enemy trail shortest path avoiding my trail to the nearest likely enemy trail cell
    D3  enemy distance to its land     from its seen cell, else averaged over the likely cells (renormalised)
    D4  my distance to my land         shortest path avoiding my trail, to the likely land of mine
    D5  my gain if I return now        flood fill along that way home, the likely enemy land counting double
    D6  enemy gain if it returns now   the same for the enemy (its likely trail, its most likely cell), mine double
    D7  margin                         D1 - D4  (+ margin_offset)
    D8  kill race                      D2 - D3

These are what Q is told (agent/q/features.py). The hard safety rules do NOT use them: a threshold on the enemy's
position drops its true cell on many turns (measured out of view: 12% of turns below 1e-6, 28% below 1e-4, half
against the hunter), and a safety rule must not be wrong exactly then. So the snapshot also keeps the worst case,
under its own names: `possible` (every cell with any probability), `enemy_arrival` (how soon it could reach each cell),
`safety_d1` and `safety_margin` (= safety_d1 - the way home + margin_offset), and the way home over the land memory
holds (`home_dist`, `home_path`), which RETURN takes and the look-ahead scores.

All distances are clipped to cfg.distance_cap, which also means "none". D5 and D6 are 0 without a trail.
The raw (unscaled, unclipped D7/D8) values are kept, because the training label needs them.
"""

from dataclasses import dataclass

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.data.cells import OPP_TERRITORY, OPP_TRAIL
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.distance import taxicab_distance
from candidate.agent.geometry.fill import BOARD
from candidate.agent.behavior import paths, votes, worldmap
from candidate.agent.behavior.config import BEHAVIOR, BehaviorConfig
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
    possible: np.ndarray              # (50, 50) bool: worst case, every cell the enemy could be on (safety)
    enemy_arrival: np.ndarray         # (2500,) int16: worst-case turns for the enemy to reach each cell (safety)
    likely: np.ndarray                # (50, 50) bool: the cells it is likely on (P >= position_threshold; D1, D3)
    enemy_trail_bb: np.ndarray        # (50,) its likely trail (P >= trail_threshold; D2, D6)
    enemy_land_map_bb: np.ndarray     # (50,) its likely land (P >= land_threshold; D3, D5, D6)
    my_land_map_bb: np.ndarray        # (50,) my likely land (D4, D5, D6)
    enemy_to_land: np.ndarray         # (2500,) int16: taxicab from each cell to the enemy's land
    enemy_seen: bool
    enemy_respawn: int                # where the enemy reappears if it dies now ...
    enemy_alt_respawn: int            # ... or here, if my agent stands on that cell
    land_votes: np.ndarray            # (50, 50) float64: hypotheses' vote for enemy land
    trail_votes: np.ndarray           # (50, 50) float64: ... enemy trail
    my_land_votes: np.ndarray         # (50, 50) float64: ... my land
    world: np.ndarray                 # (10, 50, 50) float32: the world map (worldmap.py): per-cell probabilities
    known_trail_bb: np.ndarray        # (50,) enemy trail cells seen or > known_trail_probability: a step onto one kills
    home_dist: np.ndarray             # (2500,) int16: distance to my land (memory's) through passable cells
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
    safety_d1: int                    # worst case D1: from every cell the enemy could be on (the hard rules use this)
    safety_margin: float              # worst case D7: safety_d1 - my way home (+ margin_offset)

    @property
    def exposed(self) -> bool:
        return self.trail_len > 0

    @property
    def enemy_trail_known(self) -> bool:
        return bool(self.enemy_trail_bb.any())


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


def build_snapshot(view: TurnView, cfg: BehaviorConfig = BEHAVIOR) -> Snapshot:
    obs, cap = view.obs, cfg.distance_cap
    belief = view.belief
    if belief is None or belief.detail is None:
        raise ValueError("the snapshot needs a belief with detail (BeliefView.detail)")
    detail = belief.detail

    my_trail_bb = view.my_trail_bb
    my_land_bb = view.my_territory_bb
    opp_land_bb = bitboard.from_mask(view.memory.territory == OPP_TERRITORY)
    enemy_cell_bb = bitboard.from_cells([obs.opp_seen_at]) if obs.opp_seen_at is not None else bitboard.empty()
    passable = BOARD & ~my_trail_bb & ~enemy_cell_bb
    on_land = bitboard.has_cell(my_land_bb, obs.my_pos)
    trail_len = int(obs.my_trail.size)
    trail_cells = bitboard.to_cells(my_trail_bb)

    # ---- the world map, and the cells above the thresholds -------------------------------------------------------
    land_votes = votes.enemy_land_votes(detail)
    trail_votes = votes.enemy_trail_votes(detail)
    my_land_votes = votes.my_land_votes(detail)
    world = worldmap.world_map(view, land_votes, trail_votes, my_land_votes)
    position = world[worldmap.ENEMY_POS].astype(np.float64)
    likely = position >= cfg.position_threshold
    if not likely.any():                                         # nothing that likely: its most likely cell(s)
        likely = position >= position.max()
    enemy_trail_bb = bitboard.from_mask(world[worldmap.ENEMY_TRAIL] >= cfg.trail_threshold)
    enemy_land_map_bb = bitboard.from_mask(world[worldmap.ENEMY] >= cfg.land_threshold)
    my_land_map_bb = bitboard.from_mask(world[worldmap.MINE] >= cfg.land_threshold)

    # ---- the worst case, for the hard safety rules ------------------------------------------------------------------
    possible = belief.position > cfg.possible_min_probability
    if not possible.any():
        possible = ~_window_mask(obs, lambda o: np.ones(o.window_territory.shape, dtype=bool))
    enemy_arrival = _clip_map(taxicab_distance(bitboard.from_mask(possible)).reshape(CELL_COUNT), cap)
    visible_enemy_trail = _window_mask(obs, lambda o: o.window_trail == OPP_TRAIL)
    known_trail_bb = bitboard.from_mask(visible_enemy_trail | (trail_votes > cfg.known_trail_probability))
    home_targets = my_land_bb & passable                       # memory's land: what RETURN heads for
    home_dist = paths.distance_map(home_targets, passable)
    home_path = [] if on_land else (paths.path_from_map(home_dist, obs.my_pos) if home_targets.any() else None)
    way_home = 0 if on_land else _clip(paths.distance_from(home_dist, obs.my_pos, home_targets), cap)
    safety_d1 = int(enemy_arrival[trail_cells].min()) if trail_len else cap

    # ---- D1..D8, from the map ----------------------------------------------------------------------------------------
    d1 = int(min(int(_clip_map(taxicab_distance(bitboard.from_mask(likely)).reshape(CELL_COUNT), cap)[trail_cells].min()),
                 cap)) if trail_len else cap
    hunt_targets = enemy_trail_bb & ~enemy_cell_bb & ~my_trail_bb
    d2 = cap
    if hunt_targets.any():
        d2 = _clip(paths.distance_from(paths.distance_map(hunt_targets, passable), obs.my_pos, hunt_targets), cap)
    to_enemy_land = (_clip_map(taxicab_distance(enemy_land_map_bb).reshape(CELL_COUNT), cap) if enemy_land_map_bb.any()
                     else np.full(CELL_COUNT, cap, dtype=np.int16))
    if obs.opp_seen_at is not None:
        d3 = float(to_enemy_land[obs.opp_seen_at])
    else:
        weights = np.where(likely, position, 0.0).reshape(-1)
        total = float(weights.sum())
        d3 = float((weights * to_enemy_land).sum() / total) if total > 0 else float(cap)
    my_home_targets = my_land_map_bb & passable
    my_home_dist = paths.distance_map(my_home_targets, passable)
    on_my_map_land = bitboard.has_cell(my_land_map_bb, obs.my_pos)
    d4 = 0 if on_my_map_land else _clip(paths.distance_from(my_home_dist, obs.my_pos, my_home_targets), cap)
    d5 = (paths.best_return_gain(my_land_map_bb, enemy_land_map_bb, my_trail_bb, my_home_dist, obs.my_pos)
          if trail_len and my_home_targets.any() else 0)
    d6 = _enemy_gain(obs, position, enemy_trail_bb, enemy_land_map_bb, my_land_map_bb)

    if view.opp_last_seen is not None:
        since = obs.turn - view.opp_last_seen[1]
    else:
        since = obs.turn
    since = int(min(since, cfg.since_sighting_cap))

    return Snapshot(
        turn=obs.turn, my_pos=obs.my_pos, trail_len=trail_len, on_land=on_land,
        my_trail_bb=my_trail_bb, my_land_bb=my_land_bb, opp_land_bb=opp_land_bb, enemy_cell_bb=enemy_cell_bb,
        passable_bb=passable, possible=possible, enemy_arrival=enemy_arrival, likely=likely,
        enemy_trail_bb=enemy_trail_bb, enemy_land_map_bb=enemy_land_map_bb, my_land_map_bb=my_land_map_bb,
        enemy_to_land=to_enemy_land, enemy_seen=obs.opp_seen_at is not None, enemy_respawn=view.rules.opp.respawn,
        enemy_alt_respawn=view.rules.opp.alt_respawn, land_votes=land_votes, trail_votes=trail_votes,
        my_land_votes=my_land_votes, world=world, known_trail_bb=known_trail_bb, home_dist=home_dist,
        home_path=home_path, turns_since_sighting=since, d1=d1, d2=d2, d3=d3, d4=d4, d5=d5, d6=d6,
        d7=float(d1 - d4 + cfg.margin_offset), d8=float(d2) - d3,
        safety_d1=safety_d1, safety_margin=float(safety_d1 - way_home + cfg.margin_offset),
    )


def _enemy_gain(obs, position: np.ndarray, trail: np.ndarray, land: np.ndarray, my_land: np.ndarray) -> int:
    """D6: what the enemy would capture by going straight home now, my cells counting double, on the map: its likely
    trail, its likely land, and its cell (seen, else the most likely one). 0 without a likely trail."""
    if not trail.any() or not land.any():
        return 0
    pos = obs.opp_seen_at if obs.opp_seen_at is not None else int(np.argmax(position))
    home = paths.distance_map(land, BOARD & ~trail)         # its own trail blocks it; mine does not
    return paths.best_return_gain(land, my_land, trail, home, pos)
