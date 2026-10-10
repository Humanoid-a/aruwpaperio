"""Row-word bitboards: a set of board cells stored as one int64 word per row.

Layout
    A single board is an int64 array of shape (BOARD_HEIGHT,). Bit x of word
    y is set exactly when cell (y, x) is in the set, so bit x of word y
    corresponds to flat index BOARD_WIDTH * y + x. Only bits 0 .. BOARD_WIDTH-1
    are ever used (50 of 64), so a word is never negative.
    A batch of N boards is an int64 array of shape (N, BOARD_HEIGHT).
    A board is 50 * 8 = 400 bytes.

Rules that keep it correct
    * Always int64, never uint64 (PyTorch's unsigned 64-bit support is thin).
    * Never use ~ directly: it also sets bits 50 .. 63. Use complement().
    * A left shift can spill into bit 50; mask with ROW_MASK afterwards.
    * Horizontal neighbours are shifts within a word; vertical neighbours are
      the adjacent word. Shifts never wrap between rows, unlike flat indices
      (see geometry.grid).
    * set_cell and clear_cell mutate in place. Everything else returns a new
      array. Copy with .copy() before mutating a board you do not own.

Only the representation lives here (conversion, membership, counting). Flood
fill and distance transforms are separate algorithms built on top of it.
"""

from typing import Iterable

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT

ROW_MASK = (1 << BOARD_WIDTH) - 1

_SHIFTS = np.arange(BOARD_WIDTH, dtype=np.int64)
_BIT_WEIGHTS = np.left_shift(np.int64(1), _SHIFTS)
_SHIFTS.flags.writeable = False
_BIT_WEIGHTS.flags.writeable = False


def as_board(bb: np.ndarray) -> np.ndarray:
    if not isinstance(bb, np.ndarray) or bb.dtype != np.int64:
        raise TypeError("a bitboard must be a numpy int64 array")
    if bb.ndim < 1 or bb.shape[-1] != BOARD_HEIGHT:
        raise ValueError(f"a bitboard must have shape (..., {BOARD_HEIGHT}), got {bb.shape}")
    return bb


def as_single(bb: np.ndarray) -> np.ndarray:
    bb = as_board(bb)
    if bb.ndim != 1:
        raise ValueError(f"expected a single board of shape ({BOARD_HEIGHT},), got a batch {bb.shape}")
    return bb


def _split(c: int) -> tuple[int, int]:
    if not 0 <= c < CELL_COUNT:
        raise ValueError(f"flat index {c} is outside 0..{CELL_COUNT - 1}")
    return divmod(c, BOARD_WIDTH)


def empty(*batch: int) -> np.ndarray:
    """No cells set. empty() is one board; empty(n) is a batch of n."""
    return np.zeros(batch + (BOARD_HEIGHT,), dtype=np.int64)


def full(*batch: int) -> np.ndarray:
    """Every board cell set (and no bit beyond them)."""
    return np.full(batch + (BOARD_HEIGHT,), ROW_MASK, dtype=np.int64)


def from_mask(mask: np.ndarray) -> np.ndarray:
    """Pack a boolean array of shape (..., BOARD_HEIGHT, BOARD_WIDTH), indexed
    [y][x], into bitboards of shape (..., BOARD_HEIGHT)."""
    mask = np.asarray(mask)
    if mask.dtype != np.bool_:
        raise TypeError(f"expected a boolean mask, got {mask.dtype}")
    if mask.shape[-2:] != (BOARD_HEIGHT, BOARD_WIDTH):
        raise ValueError(f"expected shape (..., {BOARD_HEIGHT}, {BOARD_WIDTH}), got {mask.shape}")
    return np.where(mask, _BIT_WEIGHTS, np.int64(0)).sum(axis=-1, dtype=np.int64)


def to_mask(bb: np.ndarray) -> np.ndarray:
    """Unpack bitboards of shape (..., BOARD_HEIGHT) into a boolean array of
    shape (..., BOARD_HEIGHT, BOARD_WIDTH), indexed [y][x]."""
    bb = as_board(bb)
    return ((bb[..., None] >> _SHIFTS) & 1).astype(np.bool_)


def from_cells(cells: Iterable[int]) -> np.ndarray:
    """One board containing the given flat indices."""
    bb = empty()
    for c in cells:
        set_cell(bb, c)
    return bb


def to_cells(bb: np.ndarray) -> list[int]:
    """The flat indices set in a single board, ascending."""
    ys, xs = np.nonzero(to_mask(as_single(bb)))
    return (ys * BOARD_WIDTH + xs).tolist()


def has_cell(bb: np.ndarray, c: int) -> bool:
    y, x = _split(c)
    return bool((int(as_single(bb)[y]) >> x) & 1)


def test_cells(bb: np.ndarray, cells) -> np.ndarray:
    """Vectorised membership: is each flat index in `cells` set in the board?

    bb is one board (50,) with `cells` of any shape, or a batch (N, 50) with
    `cells` of shape (N,) or (N, K) giving cell(s) to test per board. Returns a
    boolean array shaped like `cells`. Indices must be on the board."""
    bb = as_board(bb)
    cells = np.asarray(cells)
    rows, cols = np.divmod(cells, BOARD_WIDTH)
    if bb.ndim == 1:
        words = bb[rows]
    elif cells.ndim == 1:
        words = bb[np.arange(bb.shape[0]), rows]
    else:
        words = np.take_along_axis(bb, rows, axis=1)
    return ((words >> cols.astype(np.int64)) & 1).astype(np.bool_)


def set_cell(bb: np.ndarray, c: int) -> None:
    """Add cell c to a single board, in place."""
    y, x = _split(c)
    as_single(bb)[y] |= np.int64(1 << x)


def clear_cell(bb: np.ndarray, c: int) -> None:
    """Remove cell c from a single board, in place."""
    y, x = _split(c)
    as_single(bb)[y] &= np.int64(ROW_MASK & ~(1 << x))


def complement(bb: np.ndarray) -> np.ndarray:
    """Every board cell not in bb. The result never has bits beyond the board,
    which a bare ~ would."""
    return ~as_board(bb) & np.int64(ROW_MASK)


def popcount(bb: np.ndarray) -> int | np.ndarray:
    """Number of cells set. An int for a single board; for a batch, an array
    of counts over the leading dimensions."""
    bb = as_board(bb)
    raw = np.ascontiguousarray(bb).view(np.uint8)
    counts = np.unpackbits(raw, axis=-1).sum(axis=-1, dtype=np.int64)
    return int(counts) if bb.ndim == 1 else counts


def is_valid(bb: np.ndarray) -> bool:
    """True if bb is an int64 array of the right shape with no bit set beyond
    the board (no stray bits 50 .. 63, which also rules out negative words)."""
    if not isinstance(bb, np.ndarray) or bb.dtype != np.int64:
        return False
    if bb.ndim < 1 or bb.shape[-1] != BOARD_HEIGHT:
        return False
    return bool(((bb >= 0) & (bb <= ROW_MASK)).all())
