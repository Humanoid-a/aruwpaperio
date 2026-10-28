"""PATROL: reposition inside my borders. Walk, inside my territory, to the best exit (exits.py: room outside, safety,
how long ago the cells beyond were seen, a short walk) and pace there, so that EXPAND leaves from a good place.

The safe state: it never leaves my land, it is where a death sends the controller, and it is what is left when nothing
else is worth the risk. Valid whenever I have no trail out. Standing on land that is not
mine with no trail (it can happen after the enemy captured the cell) it walks home first.
"""

from candidate.agent.behavior import paths
from candidate.agent.behavior.common import first, pace, toward
from candidate.agent.behavior.frame import Frame
from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard


def valid(frame: Frame) -> bool:
    return not frame.snap.exposed


def move(frame: Frame) -> Direction | None:
    snap = frame.snap
    if not snap.on_land:                                   # standing on someone else's land with no trail: go home
        return toward(frame, snap.home_path)
    exits = frame.exits
    if exits.best is None or frame.pos == exits.best:
        return pace(frame)
    inside = snap.my_land_bb & snap.passable_bb
    dist = paths.distance_map(bitboard.from_cells([exits.best]), inside)
    return first(toward(frame, paths.path_from_map(dist, frame.pos)), pace(frame))
