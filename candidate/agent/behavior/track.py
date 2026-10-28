"""TRACK TARGET: find the enemy. Move towards where the probability map says it most likely is.

"Where it most likely is" is read at the scale of a look, which is the map downsampled to what a view covers: for
every cell, how much of the enemy's position probability (the world map, worldmap.py; zero inside my current view, where
it is not) an 11 x 11 view from there would cover. So a spread-out belief points at its densest region, not at one
noisy cell, and getting there is the way to see it. Each step of walking multiplies a place's worth by
`track_discount`: of two equally promising places the nearer wins, but however far the likely place is, it still
beats places where the enemy cannot be (before this, a fixed reach of 30 steps left a far-away enemy out of sight,
and the agent paced at home). The safety look-ahead still guards every step out.

Valid when the enemy is out of view (when it is seen, there is nothing to find).
"""

import numpy as np

from candidate.agent.behavior import maps, paths, worldmap
from candidate.agent.behavior.common import first, pace, toward
from candidate.agent.behavior.frame import Frame
from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.distance import UNREACHABLE


def valid(frame: Frame) -> bool:
    return bool(frame.safe) and not frame.snap.enemy_seen


def coverage(frame: Frame) -> np.ndarray:
    """(50, 50): the enemy's position probability a view centred on each cell would cover."""
    return maps.box_sum(frame.snap.world[worldmap.ENEMY_POS].astype(np.float64), frame.cfg.track_view_radius)


def target(frame: Frame) -> int | None:
    """The cell to head for: the most coverage, discounted by the walk there, among the cells I can reach."""
    snap, cfg = frame.snap, frame.cfg
    value_of = coverage(frame)
    dist = paths.distance_map(bitboard.from_cells([frame.pos]), snap.passable_bb).reshape(value_of.shape)
    reachable = dist < UNREACHABLE
    reachable.flat[frame.pos] = True
    steps = np.where(reachable, dist, 0).astype(np.float64)
    steps.flat[frame.pos] = 0.0
    value = np.where(reachable, value_of * cfg.track_discount ** steps, -np.inf)
    if not np.isfinite(value).any() or value.max() <= 0.0:
        return None
    return int(value.argmax())


def move(frame: Frame) -> Direction | None:
    if not frame.safe:
        return None
    cell = target(frame)
    if cell is None or cell == frame.pos:
        return first(pace(frame), frame.safe[0])
    dist = paths.distance_map(bitboard.from_cells([cell]), frame.snap.passable_bb)
    return first(toward(frame, paths.path_from_map(dist, frame.pos)), frame.safe[0])
