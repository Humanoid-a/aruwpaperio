"""The slow, obviously-correct move rules on a dense WorldState.

This mirrors the framework's MoveResolver, RespawnService and
TerritoryResolver step for step, using plain Python loops and a plain
breadth-first search, and shares no code with the bitboard version in
rules.transition. It exists to be checked against the real engine and to
check the fast version against; use rules.transition when speed matters.

Resolution order for one move by `mover` (the other player is `other`):

 1. Off the board, or onto the other agent's cell: INVALID, nothing changes.
 2. Onto the mover's own trail: the mover dies and respawns (DIED).
 3. The mover steps onto the destination. The destination's territory
    owner is read now and used for the later checks, before anything below
    changes it.
 4. If that cell was the mover's territory and the mover has a trail, the
    trail closes (a capture).
 5. If the destination held the other player's trail, the other player dies
    and respawns, after any capture. A kill has no result code of its own;
    it is reported separately.
 6. A capture returns CAPTURED. Otherwise, if the destination was not the
    mover's territory, it is appended to the mover's trail. Return NORMAL.

Respawn clears the player's trail and all of their territory, restores their
starting territory (taking those cells from whoever held them) and places the
agent on its respawn cell, or the alternate cell if the other agent stands there.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.cells import FREE, MY_TERRITORY, MY_TRAIL, OPP_TERRITORY, OPP_TRAIL
from candidate.agent.data.types import Direction, MoveResult, Side
from candidate.agent.data.world import WorldState
from candidate.agent.geometry.grid import DX, DY, flat, in_bounds, to_yx
from candidate.agent.rules.match import MatchRules

_CODES = {Side.ME: (MY_TERRITORY, MY_TRAIL), Side.OPP: (OPP_TERRITORY, OPP_TRAIL)}


@dataclass(frozen=True, eq=False)
class Outcome:
    world: WorldState   # the board after the move (a new object; the input is untouched)
    result: MoveResult
    killed: bool        # the mover cut the other player's trail and sent them home


def apply_move(world: WorldState, rules: MatchRules, mover: Side, direction: Direction) -> Outcome:
    other = mover.other
    mover_terr, mover_trail = _CODES[mover]
    _, other_trail = _CODES[other]
    w = world.copy()

    y, x = to_yx(_pos(w, mover))
    ny, nx = y + DY[direction], x + DX[direction]
    if not in_bounds(ny, nx):
        return Outcome(w, MoveResult.INVALID, False)
    destination = flat(ny, nx)
    if destination == _pos(w, other):
        return Outcome(w, MoveResult.INVALID, False)

    trail_at_destination = w.trail[ny, nx]
    territory_at_destination = w.territory[ny, nx]

    if trail_at_destination == mover_trail:
        _respawn(w, rules, mover)
        _recount(w)
        return Outcome(w, MoveResult.DIED, False)

    _set_pos(w, mover, destination)
    captured = territory_at_destination == mover_terr and len(_trail(w, mover)) > 0
    if captured:
        _apply_capture(w, mover)

    killed = bool(trail_at_destination == other_trail)
    if killed:
        _respawn(w, rules, other)

    if captured:
        _recount(w)
        return Outcome(w, MoveResult.CAPTURED, killed)

    if territory_at_destination != mover_terr:
        w.trail[ny, nx] = mover_trail
        _trail(w, mover).append(destination)
    _recount(w)
    return Outcome(w, MoveResult.NORMAL, killed)


# ---- pieces -------------------------------------------------------------


def _pos(w: WorldState, side: Side) -> int:
    return w.my_pos if side == Side.ME else w.opp_pos


def _set_pos(w: WorldState, side: Side, cell: int) -> None:
    if side == Side.ME:
        w.my_pos = cell
    else:
        w.opp_pos = cell


def _trail(w: WorldState, side: Side) -> list[int]:
    return w.my_trail if side == Side.ME else w.opp_trail


def _recount(w: WorldState) -> None:
    w.my_count = int((w.territory == MY_TERRITORY).sum())
    w.opp_count = int((w.territory == OPP_TERRITORY).sum())


def _respawn(w: WorldState, rules: MatchRules, side: Side) -> None:
    own_terr, _ = _CODES[side]
    role = rules.role(side)
    for cell in _trail(w, side):
        w.trail.flat[cell] = FREE
    _trail(w, side).clear()
    w.territory[w.territory == own_terr] = FREE
    for cell in role.start_cells:
        w.territory.flat[cell] = own_terr
    occupied_by_other = _pos(w, side.other) == role.respawn
    _set_pos(w, side, role.alt_respawn if occupied_by_other else role.respawn)


def _apply_capture(w: WorldState, side: Side) -> None:
    own_terr, _ = _CODES[side]
    already_enclosed = _enclosed(w.territory, own_terr)
    for cell in _trail(w, side):
        w.territory.flat[cell] = own_terr
        w.trail.flat[cell] = FREE
    for cell in _enclosed(w.territory, own_terr):
        if cell not in already_enclosed:
            w.territory.flat[cell] = own_terr
    _trail(w, side).clear()


def _enclosed(territory: np.ndarray, own_terr: int) -> set[int]:
    """Cells that are not `own_terr` land and cannot reach the board edge
    without crossing `own_terr` land, by breadth-first search from the edge."""
    reached = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            on_edge = y in (0, BOARD_HEIGHT - 1) or x in (0, BOARD_WIDTH - 1)
            if on_edge and territory[y, x] != own_terr:
                reached[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy, dx in zip(DY, DX):
            ny, nx = y + dy, x + dx
            if in_bounds(ny, nx) and not reached[ny, nx] and territory[ny, nx] != own_terr:
                reached[ny, nx] = True
                queue.append((ny, nx))
    return {
        flat(y, x)
        for y in range(BOARD_HEIGHT)
        for x in range(BOARD_WIDTH)
        if not reached[y, x] and territory[y, x] != own_terr
    }
