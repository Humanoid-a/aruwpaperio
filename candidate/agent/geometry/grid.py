"""Flat cell indexing.

A cell (y, x) has the flat index c = BOARD_WIDTH * y + x, with y the row and
x the column (the framework's GridPosition is (x, y); convert once, in the
adapter). Flat indices are what every agent module stores and passes around.

Hazard: adding FLAT_OFFSETS to a flat index is only valid when the move
stays on the board. East from the last column lands on the first cell of the
next row, and west from the first column lands on the last cell of the row
above. Use step() for single moves; vectorised code must bounds-check x
before applying the +1 / -1 offsets.
"""

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.data.types import Direction

# Indexed by Direction: N, E, S, W.
FLAT_OFFSETS = (-BOARD_WIDTH, +1, +BOARD_WIDTH, -1)
DY = (-1, 0, 1, 0)
DX = (0, 1, 0, -1)


def in_bounds(y: int, x: int) -> bool:
    return 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH


def flat(y: int, x: int) -> int:
    if not in_bounds(y, x):
        raise ValueError(f"cell ({y}, {x}) is off the {BOARD_HEIGHT}x{BOARD_WIDTH} board")
    return BOARD_WIDTH * y + x


def to_yx(c: int) -> tuple[int, int]:
    if not 0 <= c < CELL_COUNT:
        raise ValueError(f"flat index {c} is outside 0..{CELL_COUNT - 1}")
    return divmod(c, BOARD_WIDTH)


def step(c: int, direction: Direction) -> int | None:
    """The flat index one move away, or None if that move leaves the board."""
    y, x = to_yx(c)
    ny, nx = y + DY[direction], x + DX[direction]
    if not in_bounds(ny, nx):
        return None
    return BOARD_WIDTH * ny + nx


def neighbors(c: int) -> tuple[int, ...]:
    """The on-board cardinal neighbours of c, in Direction order (N, E, S, W)."""
    moved = (step(c, d) for d in Direction)
    return tuple(n for n in moved if n is not None)


def direction_to(a: int, b: int) -> Direction | None:
    """The move that takes cell a to the adjacent cell b, or None if they are
    not neighbours."""
    for direction in Direction:
        if step(a, direction) == b:
            return direction
    return None


def manhattan(a: int, b: int) -> int:
    ay, ax = to_yx(a)
    by, bx = to_yx(b)
    return abs(ay - by) + abs(ax - bx)
