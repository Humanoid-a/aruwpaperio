"""PATROL's exit score: where on the edge of my territory it is best to go out.

For every boundary cell of my territory (a land cell with a neighbour that is not mine) three things are combined,
each scaled to 0..1, and the cost of getting there is taken off:

    room    free cells just outside it: the density of cells that are neither mine nor the enemy's in a square
            around the neighbour it would step to (room to expand)
    safety  the worst-case distance from the enemy to the cell / 50 (the enemy's arrival time over the
            possibility mask)
    info    how long ago the cells beyond it were last seen / 100 (a cell never seen counts as the cap)
    walk    steps from me to the cell, inside my territory / 50

    score = w_room*room + w_safety*safety + w_info*info - w_walk*walk

Each exit also has an outward direction: the side with the most room. EXPAND starts from the best exit and heads
outward from it. The weights are in the config.
"""

from dataclasses import dataclass

import numpy as np

from candidate.agent.behavior import maps, paths
from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard


@dataclass(frozen=True, eq=False)
class Exits:
    score: np.ndarray            # (50, 50) float64; -inf where the cell is not a usable exit
    outward: np.ndarray          # (50, 50) int8: the Direction to step out in, -1 where there is none
    best: int | None             # flat cell of the best exit, or None
    walk: np.ndarray             # (2500,) steps from me to each cell inside my territory (UNREACHABLE elsewhere)


def exit_scores(frame) -> Exits:
    snap, cfg, view = frame.snap, frame.cfg, frame.view
    land = bitboard.to_mask(snap.my_land_bb)
    enemy_land = bitboard.to_mask(snap.opp_land_bb)
    free = ~land & ~enemy_land

    r = cfg.exit_room_radius
    density = maps.box_sum(free, r) / float((2 * r + 1) ** 2)

    age = np.where(view.memory.last_seen < 0, cfg.info_age_cap,
                   np.minimum(view.obs.turn - view.memory.last_seen, cfg.info_age_cap)).astype(np.float64)
    staleness = maps.box_sum(age, r) / float((2 * r + 1) ** 2) / cfg.info_age_cap

    room = np.full((4, BOARD_HEIGHT, BOARD_WIDTH), -1.0)
    stale = np.zeros((4, BOARD_HEIGHT, BOARD_WIDTH))
    for d in Direction:
        beyond_density = maps.at_neighbor(density, int(d), -1.0)
        beyond_land = maps.at_neighbor(land, int(d), True)       # off the board counts as "mine": not a way out
        room[d] = np.where(beyond_land, -1.0, beyond_density)
        stale[d] = maps.at_neighbor(staleness, int(d), 0.0)

    outward = room.argmax(axis=0).astype(np.int8)
    best_room = room.max(axis=0)
    is_exit = land & (best_room >= 0)
    outward = np.where(is_exit, outward, -1).astype(np.int8)

    rows, cols = np.indices(land.shape)
    chosen_stale = stale[np.maximum(outward, 0), rows, cols]
    safety = snap.enemy_arrival.reshape(land.shape).astype(np.float64) / cfg.distance_cap

    walk = paths.distance_map(bitboard.from_cells([view.obs.my_pos]), snap.my_land_bb & snap.passable_bb)
    walk_map = walk.reshape(land.shape).astype(np.float64)
    reachable = walk_map < 9999
    if not snap.on_land:                                      # not standing on my land: no inside walk to measure
        reachable = np.ones_like(reachable)
        walk_map = np.zeros_like(walk_map)
    walk_cost = np.minimum(walk_map, cfg.distance_cap) / cfg.distance_cap

    score = (cfg.exit_weight_room * np.maximum(best_room, 0.0) + cfg.exit_weight_safety * safety
             + cfg.exit_weight_info * chosen_stale - cfg.exit_weight_walk * walk_cost)
    if frame.style is not None:                               # a training game's own preference (style.py)
        score = score + frame.style.exit_bias
    usable = is_exit & reachable
    score = np.where(usable, score, -np.inf)
    best = int(score.argmax()) if usable.any() else None
    return Exits(score, outward, best, walk)
