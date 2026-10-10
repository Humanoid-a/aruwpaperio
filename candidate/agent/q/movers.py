"""The five action-states' own logic: from the frame to one move.

Q decides which state to be in; these decide how to move. Every state recomputes its target every frame, so a
change of mind happens exactly when the best choice changes. Each state has a validity test (cheap, no search
beyond what the snapshot already has) and a mover returning a Direction, or None when there is no move.

    EXPAND  inside my land: walk to the exit PATROL scores best, then step out the way with the most room.
            Outside: straight on in the direction the expedition set out in. Never turns back (that is RETURN's job).
    RETURN  close the loop with the best balance of gain and safety (see choose_return_path).
    HUNT    step onto the enemy's trail: head for the known trail cell with the best race margin by the shortest
            path that avoids my own trail. The race margin of a cell is (the enemy's time to get home) - (my time to
            reach it); the first is the same for every cell, so the best one is the nearest.
    TRACK   keep the enemy in view (3-5 cells away, preferring my land), or if unseen head for the cell whose
            11 x 11 view would cover the most belief mass.
    PATROL  walk inside my territory to the best exit and pace there. Never leaves territory.

Validity: EXPAND needs a next outward cell on the board that I can step to; RETURN needs a trail; HUNT needs a known
enemy trail cell I can reach; TRACK is always valid; PATROL needs an empty trail.
"""

import numpy as np

from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.distance import UNREACHABLE, shortest_path
from candidate.agent.geometry.fill import BOARD
from candidate.agent.geometry.grid import direction_to, manhattan, step
from candidate.agent.q import maps, paths
from candidate.agent.q.frame import Frame
from candidate.agent.q.types import N_STATES, State

_PERPENDICULAR = {
    Direction.NORTH: (Direction.EAST, Direction.WEST),
    Direction.SOUTH: (Direction.EAST, Direction.WEST),
    Direction.EAST: (Direction.NORTH, Direction.SOUTH),
    Direction.WEST: (Direction.NORTH, Direction.SOUTH),
}


def _first(*options: Direction | None) -> Direction | None:
    """The first option that is a move. (Not `or`: Direction.NORTH is 0, which is falsy.)"""
    for option in options:
        if option is not None:
            return option
    return None


def _toward(frame: Frame, path: list[int] | None) -> Direction | None:
    if not path:
        return None
    return direction_to(frame.pos, path[0])


def _safe_dest(frame: Frame, direction: Direction | None) -> int | None:
    """The cell a move steps onto, if the move is one I can make without stepping on my own trail."""
    if direction is None or direction not in frame.safe:
        return None
    return step(frame.pos, direction)


def _in_land(frame: Frame, cell: int) -> bool:
    return bitboard.has_cell(frame.snap.my_land_bb, cell)


# ---- PATROL -------------------------------------------------------------------------------------------------


def patrol(frame: Frame) -> Direction | None:
    snap = frame.snap
    if not snap.on_land:                                   # standing on someone else's land with no trail: go home
        return _toward(frame, snap.home_path)
    exits = frame.exits
    if exits.best is None:
        return _pace(frame)
    if frame.pos == exits.best:
        return _pace(frame)
    inside = snap.my_land_bb & snap.passable_bb
    dist = paths.distance_map(bitboard.from_cells([exits.best]), inside)
    return _first(_toward(frame, paths.path_from_map(dist, frame.pos)), _pace(frame))


def _pace(frame: Frame) -> Direction | None:
    """Step to a neighbouring cell of my land and, next frame, back: gathers information and never leaves my land.
    Which neighbour alternates with the turn so it does not stick in a corner."""
    options = [d for d in frame.safe if _in_land(frame, step(frame.pos, d))]
    if not options:
        return None
    options.sort(key=lambda d: step(frame.pos, d))
    return options[frame.view.obs.turn % len(options)]


# ---- EXPAND -------------------------------------------------------------------------------------------------


def _exit_step(frame: Frame, cell: int) -> Direction | None:
    """From the exit cell `cell` (where I stand), the step out: its best outward side if that is a move I can make."""
    d = int(frame.exits.outward.flat[cell])
    if d < 0:
        return None
    direction = Direction(d)
    dest = _safe_dest(frame, direction)
    if dest is not None and not _in_land(frame, dest):
        return direction
    for other in frame.safe:                               # the best side is blocked: any other way out
        dest = step(frame.pos, other)
        if not _in_land(frame, dest):
            return other
    return None


def expand_valid(frame: Frame) -> bool:
    snap = frame.snap
    if snap.exposed:
        return _safe_dest(frame, frame.heading()) is not None
    if not snap.on_land:
        return False
    exits = frame.exits
    if exits.best is None:
        return False
    if frame.pos == exits.best:
        return _exit_step(frame, exits.best) is not None
    return True


def expand(frame: Frame) -> Direction | None:
    snap = frame.snap
    if snap.exposed:
        heading = frame.heading()
        return heading if _safe_dest(frame, heading) is not None else None
    if not snap.on_land or frame.exits.best is None:
        return None
    if frame.pos == frame.exits.best:
        return _exit_step(frame, frame.pos)
    inside = snap.my_land_bb & snap.passable_bb
    dist = paths.distance_map(bitboard.from_cells([frame.exits.best]), inside)
    return _toward(frame, paths.path_from_map(dist, frame.pos))


# ---- RETURN -------------------------------------------------------------------------------------------------


def return_candidates(frame: Frame) -> list[list[int]]:
    """The direct shortest way home, plus, for each sideways length w from 1 to W in both sideways directions,
    'go sideways w cells, then take the shortest way home that avoids my trail (and the cells just walked)'."""
    snap, cfg = frame.snap, frame.cfg
    out: list[list[int]] = []
    if snap.home_path:
        out.append(list(snap.home_path))
    heading = frame.heading()
    if heading is None:
        return out
    for sideways in _PERPENDICULAR[heading]:
        leg: list[int] = []
        cur = frame.pos
        for _ in range(cfg.return_max_sideways):
            nxt = step(cur, sideways)
            if nxt is None or bitboard.has_cell(snap.my_trail_bb, nxt) or bitboard.has_cell(snap.enemy_cell_bb, nxt):
                break
            leg.append(nxt)
            cur = nxt
            if bitboard.has_cell(snap.my_land_bb, cur):          # the sideways leg reached home itself
                out.append(list(leg))
                break
            blocked = paths.with_cells(snap.my_trail_bb, leg) | snap.enemy_cell_bb
            home = shortest_path(cur, snap.my_land_bb, BOARD & ~blocked)
            if home is not None:
                out.append(leg + home)
    return out


def score_paths(frame: Frame, candidates: list[list[int]]):
    """(gain, closing time, margin) of every candidate path, as arrays.
    Gain: cells it would capture, the enemy's counting double. Closing time: turns until it reaches my land.
    Margin: the worst-case enemy arrival on any of my exposed cells (my trail and the path) minus the closing time."""
    snap, cfg = frame.snap, frame.cfg
    trails = np.stack([paths.with_cells(snap.my_trail_bb, c) for c in candidates])
    gains = np.asarray(paths.gain(snap.my_land_bb, snap.opp_land_bb, trails))
    times = np.array([len(c) for c in candidates])
    exposed = trails & ~snap.my_land_bb[None, :]
    margins = np.empty(len(candidates), dtype=np.int64)
    for i in range(len(candidates)):
        margins[i] = int(snap.enemy_arrival[bitboard.to_cells(exposed[i])].min()) - int(times[i]) + cfg.margin_offset
    return gains, times, margins


def choose_return_path(frame: Frame) -> list[int] | None:
    """The highest-gain path whose margin is at least the buffer; if none has it, the path with the largest margin."""
    candidates = return_candidates(frame)
    if not candidates:
        return None
    gains, times, margins = score_paths(frame, candidates)
    safe = margins >= frame.cfg.return_buffer
    if safe.any():
        order = sorted(np.nonzero(safe)[0], key=lambda i: (-gains[i], times[i], i))
    else:
        order = sorted(range(len(candidates)), key=lambda i: (-margins[i], -gains[i], times[i], i))
    return candidates[int(order[0])]


def return_valid(frame: Frame) -> bool:
    return frame.snap.exposed


def return_(frame: Frame) -> Direction | None:
    path = choose_return_path(frame)
    direction = _toward(frame, path)
    return direction if direction in frame.safe else None


# ---- HUNT -----------------------------------------------------------------------------------------------------


def hunt_valid(frame: Frame) -> bool:
    return frame.snap.hunt_dist is not None and frame.snap.d2 < frame.cfg.distance_cap


def hunt(frame: Frame) -> Direction | None:
    if not hunt_valid(frame):
        return None
    direction = _toward(frame, paths.path_from_map(frame.snap.hunt_dist, frame.pos))
    return direction if direction in frame.safe else None


# ---- TRACK TARGET ---------------------------------------------------------------------------------------------


def track(frame: Frame) -> Direction | None:
    if not frame.safe:
        return None
    if frame.snap.enemy_seen:
        return _track_visible(frame)
    return _track_unseen(frame)


def _track_visible(frame: Frame) -> Direction | None:
    cfg = frame.cfg
    enemy = frame.view.obs.opp_seen_at
    mid = (cfg.track_min_distance + cfg.track_max_distance) / 2

    def cost(direction: Direction):
        dest = step(frame.pos, direction)
        d = manhattan(dest, enemy)
        off = max(0, cfg.track_min_distance - d, d - cfg.track_max_distance)
        return (off, 0 if _in_land(frame, dest) else 1, abs(d - mid))

    return min(frame.safe, key=cost)


def _track_unseen(frame: Frame) -> Direction | None:
    snap, cfg = frame.snap, frame.cfg
    belief = frame.view.belief
    coverage = maps.box_sum(belief.position, cfg.track_view_radius)
    here = bitboard.from_cells([frame.pos])

    def best_cell(passable: np.ndarray, only_land: bool):
        dist = paths.distance_map(here, passable).reshape(coverage.shape).astype(np.float64)
        reachable = dist <= cfg.track_reach_cap
        if only_land:
            reachable &= bitboard.to_mask(snap.my_land_bb)
        else:                                                # outside my land only where I get there before the enemy could
            arrival = snap.enemy_arrival.reshape(coverage.shape)
            reachable &= bitboard.to_mask(snap.my_land_bb) | (arrival > dist)
        if not reachable.any():
            return None
        value = np.where(reachable, coverage - cfg.track_distance_cost * dist, -np.inf)
        cell = int(value.argmax())
        return cell, float(value.flat[cell])

    inside = best_cell(snap.my_land_bb & snap.passable_bb, True) if snap.on_land else None
    outside = best_cell(snap.passable_bb, False)
    target = inside
    if outside is not None and (inside is None or outside[1] > inside[1] + cfg.track_outside_margin):
        target = outside
    if target is None:
        return _first(_pace(frame), frame.safe[0])
    cell = target[0]
    if cell == frame.pos:
        return _first(_pace(frame), frame.safe[0])
    passable = snap.passable_bb if target is outside else (snap.my_land_bb & snap.passable_bb)
    dist = paths.distance_map(bitboard.from_cells([cell]), passable)
    return _first(_toward(frame, paths.path_from_map(dist, frame.pos)), frame.safe[0])


# ---- the table ----------------------------------------------------------------------------------------------------


def valid_states(frame: Frame) -> np.ndarray:
    """(5,) boolean in the order of Q's outputs. With no move available at all, nothing is valid."""
    out = np.zeros(N_STATES, dtype=bool)
    if not frame.safe:
        return out
    out[State.EXPAND] = expand_valid(frame)
    out[State.RETURN] = return_valid(frame)
    out[State.HUNT] = hunt_valid(frame)
    out[State.TRACK] = True
    out[State.PATROL] = not frame.snap.exposed
    return out


_MOVERS = {
    State.EXPAND: expand, State.RETURN: return_, State.HUNT: hunt, State.TRACK: track, State.PATROL: patrol,
}


def move_for(state: State, frame: Frame) -> Direction | None:
    return _MOVERS[State(state)](frame)
