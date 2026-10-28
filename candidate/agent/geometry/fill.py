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


_SPANS = (1, 2, 4, 8, 16, 32)       # 1 + 2 + ... + 32 = 63 >= 49: any run inside a row


def _grow(cur: np.ndarray, passable: np.ndarray) -> np.ndarray:
    """One round of flood: `cur` (inside `passable`) run along its rows to the ends of their passable runs, then one
    step up and down. Towards higher bits, adding the seeds to the passable runs carries through each run; towards
    lower bits it is an occluded fill (Kogge-Stone): log2(50) shifts instead of one per cell. `passable` has no bits
    above 49, so nothing spills past a row's end."""
    g = cur | (((passable + cur) ^ passable) & passable)
    p = passable
    for k in _SPANS:
        g |= p & (g >> k)
        p = p & (p >> k)
    g[..., 1:] |= g[..., :-1] & passable[..., 1:]
    g[..., :-1] |= g[..., 1:] & passable[..., :-1]
    return g


def flood(seed: np.ndarray, passable: np.ndarray) -> np.ndarray:
    """Every cell of `passable` connected to a seed cell through passable
    cells. Seed cells outside `passable` are ignored. A round runs along whole
    rows, so the rounds needed are about the rows a path crosses, not its
    length; in a batch, a board drops out of the rounds once it stops growing."""
    shape = np.broadcast_shapes(seed.shape, passable.shape)          # a single board may go with a batch, as before
    passable = np.broadcast_to(passable & ROW_MASK, shape)
    cur = seed & passable
    if cur.ndim == 1:
        while True:
            grown = _grow(cur, passable)
            if np.array_equal(grown, cur):
                return cur
            cur = grown
    out = cur.copy()
    rows = np.arange(cur.shape[0])
    while rows.size:
        grown = _grow(cur, passable)
        moving = (grown != cur).any(axis=1)
        out[rows] = grown
        if moving.all():
            cur = grown
            continue
        rows, cur, passable = rows[moving], grown[moving], passable[moving]
    return out


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
