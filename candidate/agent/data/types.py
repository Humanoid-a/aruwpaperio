"""Our own vocabulary types. Everything past the adapter uses these instead
of the framework's, so no module here imports territorygame.

The framework's Direction orders NORTH, SOUTH, EAST, WEST and its MoveResult
calls a normal step MOVED; ours orders directions N, E, S, W (the order of
the flat-index offsets) and calls it NORMAL. The adapter maps between them.
"""

from enum import Enum, IntEnum


class Direction(IntEnum):
    """Cardinal moves, numbered 0 to 3 so they can index offset tables."""

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    @property
    def opposite(self) -> "Direction":
        return Direction((self + 2) % 4)


class Side(IntEnum):
    """Which player: ME is the controller's own player, OPP the other one."""

    ME = 0
    OPP = 1

    @property
    def other(self) -> "Side":
        return Side(1 - self)


class MoveResult(Enum):
    NORMAL = 0
    CAPTURED = 1
    DIED = 2
    INVALID = 3
