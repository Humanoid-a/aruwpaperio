"""Small helpers the six states share."""

from candidate.agent.behavior.frame import Frame
from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import direction_to, step


PERPENDICULAR = {
    Direction.NORTH: (Direction.EAST, Direction.WEST),
    Direction.SOUTH: (Direction.EAST, Direction.WEST),
    Direction.EAST: (Direction.NORTH, Direction.SOUTH),
    Direction.WEST: (Direction.NORTH, Direction.SOUTH),
}


def first(*options: Direction | None) -> Direction | None:
    """The first option that is a move. (Not `or`: Direction.NORTH is 0, which is falsy.)"""
    for option in options:
        if option is not None:
            return option
    return None


def toward(frame: Frame, path: list[int] | None) -> Direction | None:
    """The step onto the first cell of `path` (a path starts at the cell after mine), or None without a path."""
    if not path:
        return None
    return direction_to(frame.pos, path[0])


def safe_dest(frame: Frame, direction: Direction | None) -> int | None:
    """The cell a move steps onto, if the move is one I can make without stepping on my own trail."""
    if direction is None or direction not in frame.safe:
        return None
    return step(frame.pos, direction)


def in_land(frame: Frame, cell: int) -> bool:
    return bitboard.has_cell(frame.snap.my_land_bb, cell)


def pace(frame: Frame) -> Direction | None:
    """Step to a neighbouring cell of my land and, next frame, back: gathers information and never leaves my land.
    Which neighbour alternates with the turn so it does not stick in a corner. (PATROL at its exit, and TRACK TARGET
    when there is nowhere better to look from.)"""
    options = [d for d in frame.safe if in_land(frame, step(frame.pos, d))]
    if not options:
        return None
    options.sort(key=lambda d: step(frame.pos, d))
    return options[frame.view.obs.turn % len(options)]
