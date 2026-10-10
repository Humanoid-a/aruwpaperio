"""Distance maps on the board.

Distance maps are int16 arrays indexed [y][x]. A cell that cannot be reached
holds UNREACHABLE (9999, far above any real distance on a 50 x 50 board and
small enough that adding a few steps to it cannot overflow int16).
"""

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.fill import BOARD, dilate
from candidate.agent.geometry.grid import neighbors

UNREACHABLE = 9999


def bfs_distance(sources: np.ndarray, passable: np.ndarray) -> np.ndarray:
    """Steps from the nearest source to every cell, moving 4-connected
    through `passable` cells. Sources have distance 0 whether or not they
    are passable; a cell that is not passable is never entered, so it keeps
    UNREACHABLE unless it is itself a source.

    Because moves are symmetric this is also the distance *to* the nearest
    source from each cell, as long as the start cell is passable or the
    caller reads it from its neighbours (see shortest_path). Single board.
    """
    sources = bitboard.as_single(sources) & BOARD
    passable = bitboard.as_single(passable)
    dist = np.full((BOARD_HEIGHT, BOARD_WIDTH), UNREACHABLE, dtype=np.int16)
    seen = sources.copy()
    front = sources
    dist[bitboard.to_mask(seen)] = 0
    step = 0
    while True:
        step += 1
        grown = dilate(front, passable) & ~seen
        if not grown.any():
            return dist
        dist[bitboard.to_mask(grown)] = step
        seen |= grown
        front = grown


def shortest_path(start: int, targets: np.ndarray, passable: np.ndarray) -> list[int] | None:
    """A shortest route from `start` into `targets` through `passable` cells,
    as the flat indices of the cells to step onto, in order, ending on a
    target. [] if start is already a target; None if no route exists.

    Ties are broken in Direction order (north, east, south, west), so the
    result is deterministic. `start` need not be passable (an exposed head
    stands on its own trail, which is blocked)."""
    if bitboard.has_cell(targets, start):
        return []
    flat_dist = bfs_distance(targets, passable).reshape(CELL_COUNT)
    path: list[int] = []
    cur = start
    while True:
        options = neighbors(cur)
        best = min(options, key=lambda n: (flat_dist[n], options.index(n)))
        if flat_dist[best] >= UNREACHABLE:
            return None
        path.append(best)
        if flat_dist[best] == 0:
            return path
        cur = best


def taxicab_distance(sources: np.ndarray) -> np.ndarray:
    """Manhattan distance from every cell to the nearest source, ignoring
    obstacles (two sweeps along each axis, no loops over cells in Python).
    UNREACHABLE everywhere if there are no sources.

    `sources` is one bitboard (50,) giving a map (50, 50), or a batch (N, 50)
    giving (N, 50, 50) so many paths are handled in one call. The opponent is
    only blocked by its own trail, so for it this is a safe lower bound on the
    true travel time."""
    d = np.where(bitboard.to_mask(sources), 0, UNREACHABLE).astype(np.int16)
    for x in range(1, BOARD_WIDTH):
        np.minimum(d[..., x], d[..., x - 1] + 1, out=d[..., x])
    for x in range(BOARD_WIDTH - 2, -1, -1):
        np.minimum(d[..., x], d[..., x + 1] + 1, out=d[..., x])
    for y in range(1, BOARD_HEIGHT):
        np.minimum(d[..., y, :], d[..., y - 1, :] + 1, out=d[..., y, :])
    for y in range(BOARD_HEIGHT - 2, -1, -1):
        np.minimum(d[..., y, :], d[..., y + 1, :] + 1, out=d[..., y, :])
    return np.minimum(d, UNREACHABLE)
