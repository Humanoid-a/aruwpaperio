"""RETURN: bring the trail home by the path that gains the most, among those the enemy cannot cut off. (The module is
not called return.py: `return` is a keyword.)

Where to close the loop matters: closing on a different cell of my land encloses a different area. So RETURN looks at
a series of places to come back to and works out, for each, what it is worth and whether it is safe:

  candidates  every cell of my land I can reach without first entering my land elsewhere (one search from my head
              through the cells outside my land, avoiding my trail and the visible enemy), up to `return_max_extra`
              steps longer than the nearest one, at most `return_max_candidates` of them (the shortest first)
  path        the shortest such path to that cell
  gain        the cells the loop would capture if closed there: the trail, the path, and everything they enclose,
              the enemy's land counting double (it swings the score twice) -- computed for all candidates in one batch
  margin      the enemy's worst-case arrival on any of my exposed cells (my trail and every cell of the path) minus the
              steps to get home, plus margin_offset: how many turns ahead of the enemy I would be
  choice      the highest gain among the paths with margin >= `return_margin` (the shorter of equal gains); if no path
              is that safe, the one with the largest margin

Valid whenever I have a trail.
"""

import numpy as np

from candidate.agent.behavior import maps, paths
from candidate.agent.behavior.common import toward
from candidate.agent.behavior.frame import Frame
from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.distance import UNREACHABLE


def candidates(frame: Frame) -> list[list[int]]:
    """The candidate paths home (cells to step onto, the last on my land), the shortest first."""
    snap, cfg = frame.snap, frame.cfg
    land = bitboard.to_mask(snap.my_land_bb)
    outside = snap.passable_bb & ~snap.my_land_bb
    dist = paths.distance_map(bitboard.from_cells([frame.pos]), outside).reshape(BOARD_HEIGHT, BOARD_WIDTH)
    nearest_outside = np.full((BOARD_HEIGHT, BOARD_WIDTH), UNREACHABLE, dtype=np.int64)
    for d in Direction:
        nearest_outside = np.minimum(nearest_outside, maps.at_neighbor(dist.astype(np.int64), int(d), UNREACHABLE))
    entry = np.where(land & bitboard.to_mask(snap.passable_bb) & (nearest_outside < UNREACHABLE), nearest_outside + 1,
                     UNREACHABLE).reshape(CELL_COUNT)
    if (entry >= UNREACHABLE).all():
        return []
    best = int(entry.min())
    cells = np.flatnonzero(entry <= best + cfg.return_max_extra)
    cells = cells[np.argsort(entry[cells], kind="stable")][: cfg.return_max_candidates]
    flat = dist.reshape(CELL_COUNT)
    out = []
    for cell in cells:
        back = paths.path_from_map(flat, int(cell))          # from the land cell back to my head
        if back is None:
            continue
        out.append(list(reversed(back[:-1])) + [int(cell)])
    return out


def score_paths(frame: Frame, paths_home: list[list[int]]):
    """(gain, steps, margin) of every candidate path, as arrays (see the module notes)."""
    snap, cfg = frame.snap, frame.cfg
    trails = np.stack([paths.with_cells(snap.my_trail_bb, path[:-1]) for path in paths_home])
    gains = np.asarray(paths.gain(snap.my_land_bb, snap.opp_land_bb, trails))
    steps = np.array([len(path) for path in paths_home])
    trail_cells = bitboard.to_cells(snap.my_trail_bb)
    margins = np.array([int(snap.enemy_arrival[trail_cells + path].min()) - int(n) + cfg.margin_offset
                        for path, n in zip(paths_home, steps)])
    return gains, steps, margins


def choose_path(frame: Frame) -> list[int] | None:
    """The path to take home (see the module notes), or None without a way home."""
    options = candidates(frame)
    if not options:
        return frame.snap.home_path or None
    gains, steps, margins = score_paths(frame, options)
    safe = margins >= frame.cfg.return_margin
    if safe.any():
        i = min(np.flatnonzero(safe), key=lambda k: (-gains[k], steps[k], k))
    else:
        i = min(range(len(options)), key=lambda k: (-margins[k], -gains[k], steps[k], k))
    return options[int(i)]


def valid(frame: Frame) -> bool:
    return frame.snap.exposed


def move(frame: Frame) -> Direction | None:
    direction = toward(frame, choose_path(frame))
    return direction if direction in frame.safe else None
