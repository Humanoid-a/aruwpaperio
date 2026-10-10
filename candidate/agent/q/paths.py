"""Paths and what they are worth. Pure functions on bitboards and flat distance maps.

A distance map here is the flat (2500,) int16 array from geometry.distance.bfs_distance, indexed by flat cell.
"""

import numpy as np

from candidate.agent.config import BOARD_WIDTH, CELL_COUNT
from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.distance import UNREACHABLE, bfs_distance
from candidate.agent.geometry.fill import BOARD, capture_region
from candidate.agent.geometry.grid import neighbors, step


def distance_map(targets: np.ndarray, passable: np.ndarray) -> np.ndarray:
    """Steps from the nearest target to every cell through `passable`, flat. UNREACHABLE where there is no way."""
    return bfs_distance(targets, passable).reshape(CELL_COUNT)


def distance_from(dist: np.ndarray, start: int, targets: np.ndarray) -> int | None:
    """Steps from `start` into the targets given a distance map to them, or None if it cannot get there.
    `start` need not be passable (an exposed head stands on its own trail)."""
    if bitboard.has_cell(targets, start):
        return 0
    best = min((int(dist[n]) for n in neighbors(start)), default=UNREACHABLE)
    return None if best >= UNREACHABLE else best + 1


DEFAULT_ORDER = (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)

# The eight tie-break orders that turn into each other when the board is rotated or mirrored: the four
# rotations of (N, E, S, W) and of its mirror image (N, W, S, E). Taking the best over all eight makes a
# result that depends on which of several equally short paths is followed independent of the board's
# orientation.
ORDERS = tuple(
    tuple(base[(i + k) % 4] for i in range(4))
    for base in (DEFAULT_ORDER, (Direction.NORTH, Direction.WEST, Direction.SOUTH, Direction.EAST))
    for k in range(4)
)


def path_from_map(dist: np.ndarray, start: int, order: tuple = DEFAULT_ORDER) -> list[int] | None:
    """The cells to step onto, steepest descent of a distance map, ties broken in the given direction order
    (default N, E, S, W). [] if `start` is a target (distance 0); None if the map cannot be descended."""
    if dist[start] == 0:
        return []
    path: list[int] = []
    cur = start
    for _ in range(CELL_COUNT):
        options = [n for n in (step(cur, d) for d in order) if n is not None]
        best = min(options, key=lambda n: (int(dist[n]), options.index(n)))
        if dist[best] >= UNREACHABLE:
            return None
        path.append(best)
        if dist[best] == 0:
            return path
        cur = best
    return None


def captured_cells(land: np.ndarray, trails: np.ndarray) -> np.ndarray:
    """The cells that become `land` when `trails` (one board or a batch) close onto it: the trail cells that
    were not already land, and everything they enclose."""
    return (trails & ~land) | capture_region(land, trails)


def gain(land: np.ndarray, rival_land: np.ndarray, trails: np.ndarray) -> np.ndarray | int:
    """What closing `trails` onto `land` is worth, in cells, with the rival's cells counting double.
    One board gives an int, a batch an array."""
    taken = captured_cells(land, trails)
    return bitboard.popcount(taken) + bitboard.popcount(taken & rival_land)


def best_return_gain(land: np.ndarray, rival_land: np.ndarray, trail: np.ndarray, dist: np.ndarray, start: int) -> int:
    """What going straight home from `start` is worth, as the best over every way of breaking ties between equally
    short paths (see ORDERS), so the answer does not depend on the board's orientation. 0 if there is no way."""
    if not trail.any():
        return 0
    trails = []
    for order in ORDERS:
        way = path_from_map(dist, start, order)
        if way is not None:
            trails.append(with_cells(trail, way))
    if not trails:
        return 0
    return int(gain(land, rival_land, np.stack(trails)).max())


def with_cells(trail: np.ndarray, cells) -> np.ndarray:
    """A copy of the trail bitboard with the given flat cells added."""
    out = trail.copy()
    for c in cells:
        bitboard.set_cell(out, c)
    return out


def direction_of_step(a: int, b: int) -> Direction | None:
    for d in Direction:
        if step(a, d) == b:
            return d
    return None


def walk(start: int, direction: Direction, steps: int) -> list[int]:
    """The cells met walking `steps` cells from `start` in `direction`, stopping at the board edge."""
    out: list[int] = []
    cur = start
    for _ in range(steps):
        nxt = step(cur, direction)
        if nxt is None:
            break
        out.append(nxt)
        cur = nxt
    return out


def board_passable(blocked: np.ndarray) -> np.ndarray:
    return BOARD & ~blocked


def cell_xy(c: int) -> tuple[int, int]:
    return c % BOARD_WIDTH, c // BOARD_WIDTH
