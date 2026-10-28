"""EXPAND2: the turn. Out in the open, go sideways and keep going that way.

An expedition that only goes straight out and straight back encloses nothing but its own trail. EXPAND2 is the
across leg of a loop: perpendicular to the direction the expedition went out in (Frame.heading), on one side, chosen
at the first turn and kept until the trail is home (Hints.across). The side chosen is the one with more free room
ahead (cells before the board's edge, my trail or my land), so the loop opens towards space. A step sideways onto my
own land closes the loop, which is a capture.

Valid when I have a trail, know the expedition's direction, and can step to the chosen side (or, before choosing, to
either side).
"""

from candidate.agent.behavior.common import PERPENDICULAR, in_land, safe_dest
from candidate.agent.behavior.frame import Frame
from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import step

ROOM_LOOK = 25                         # how far ahead the free room on each side is counted


def _room(frame: Frame, direction: Direction) -> int:
    """Free cells straight ahead in `direction` from where I stand: not my trail, not my land, on the board."""
    snap, cell, n = frame.snap, frame.pos, 0
    for _ in range(ROOM_LOOK):
        cell = step(cell, direction)
        if cell is None or bitboard.has_cell(snap.my_trail_bb, cell) or bitboard.has_cell(snap.my_land_bb, cell):
            break
        n += 1
    return n


def side(frame: Frame) -> Direction | None:
    """The side to go: the one already chosen if it can still be taken; else, of the two perpendicular to the way out
    that can be taken, the one with more free room (ties to the lower direction)."""
    heading = frame.heading()
    if not frame.snap.exposed or heading is None:
        return None
    across = frame.hints.across
    if across is not None and across in PERPENDICULAR[heading]:
        return across if safe_dest(frame, across) is not None else None
    options = [d for d in PERPENDICULAR[heading] if safe_dest(frame, d) is not None]
    if not options:
        return None
    return max(options, key=lambda d: (_room(frame, d), -int(d)))


def valid(frame: Frame) -> bool:
    return side(frame) is not None


def move(frame: Frame) -> Direction | None:
    """The sideways step. The first one also fixes the side for the rest of the expedition (Hints.across)."""
    direction = side(frame)
    if direction is not None and frame.hints.across is None and not in_land(frame, step(frame.pos, direction)):
        frame.hints.across = direction
    return direction
