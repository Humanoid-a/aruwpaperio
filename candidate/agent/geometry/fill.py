"""Flood fill and enclosure on bitboards (see geometry.bitboard for the layout).

Everything here works on a single board of shape (50,) or on a batch of
shape (N, 50), and returns the same shape. Connectivity is 4-neighbourhood
(no diagonals), the same as movement.

holes() and capture_region() reproduce the engine's capture rule exactly: a
cell is enclosed when it is not the capturer's territory and cannot be
reached from the board edge through cells that are not the capturer's
territory. A trail only counts as a wall once it has been converted to
territory, and regions that were already enclosed before the trail closed
are not captured (see capture_region).
"""

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.geometry import bitboard

ROW_MASK = np.int64(bitboard.ROW_MASK)


def _build_edge() -> np.ndarray:
    edge = bitboard.empty()
    edge[0] = ROW_MASK
    edge[BOARD_HEIGHT - 1] = ROW_MASK
    edge |= np.int64(1) | np.int64(1 << (BOARD_WIDTH - 1))  # first and last column of every row
    edge.flags.writeable = False
    return edge


EDGE = _build_edge()
"""Every cell on the border of the board."""

BOARD = bitboard.full()
BOARD.flags.writeable = False


def dilate(bb: np.ndarray, passable: np.ndarray | None = None) -> np.ndarray:
    """The set grown by one step: every cell in bb plus its four neighbours,
    clipped to the board and, if given, to `passable`. Shifts never wrap
    between rows."""
    g = bb | (bb << 1) | (bb >> 1)  # horizontal neighbours, inside each word
    g[..., 1:] |= bb[..., :-1]      # vertical neighbours: the adjacent row words
    g[..., :-1] |= bb[..., 1:]
    g &= ROW_MASK                   # a left shift can spill into bit 50
    if passable is not None:
        g &= passable
    return g


def flood(seed: np.ndarray, passable: np.ndarray) -> np.ndarray:
    """Every cell of `passable` connected to a seed cell through passable
    cells. Seed cells outside `passable` are ignored."""
    cur = seed & passable
    while True:
        grown = dilate(cur, passable)
        if np.array_equal(grown, cur):
            return cur
        cur = grown


def holes(blocked: np.ndarray) -> np.ndarray:
    """Cells that are not blocked and cannot reach the board edge without
    crossing a blocked cell: the regions `blocked` fully surrounds."""
    passable = ~blocked & BOARD
    reached = flood(passable & EDGE, passable)
    return passable & ~reached


def capture_region(territory: np.ndarray, trail: np.ndarray) -> np.ndarray:
    """The cells newly captured when `trail` closes onto `territory`.

    The trail is converted to territory first, then everything the larger
    territory encloses is captured, except regions that were already
    enclosed before the trail closed. The converted trail cells themselves
    are not part of the result: the gain in cells is
    popcount(trail & ~territory) + popcount(capture_region(...)).

    Territory and trail may be single boards or matching batches.
    """
    before = holes(territory)
    after = holes(territory | trail)
    return after & ~before
