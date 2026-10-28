"""EXPAND: go out from a nearby edge of my territory, at a wide angle to the border there, and keep going straight.

Inside my land it picks an exit (a cell on my land's edge with room outside it) and a side to step out on:

    candidates   the exits at most `expand_max_walk` (2) steps away inside my land: EXPAND picks a spot near me to go
                 out from; it does not reposition (that is PATROL)
    the border   at each candidate, the line of my land's edge there: the principal direction b of the edge cells within
                 `expand_border_radius` (3) of it (a unit vector; for a corner, where the edge cells have no clear line,
                 its weight a, from 0 for no line to 1 for a straight one, is small)
    the angle    for each side d it can step out on (a unit vector), 1 - a |d . b|: 1 going straight out of the border,
                 0 going along it. A trail that runs along my own border closes almost nothing, the low-gain habit of
                 taking the nearest exit and its roomiest side
    the choice   the largest angle, in quarter steps (nearly equal angles are equal); then the nearest exit; then the
                 better exit score (exits.py). Nearest before score: the score rewards edges not seen for a while, and
                 walking towards one shows it, so with the score first the target kept changing and the agent paced
                 inside its land (848 of 2000 frames in a measured game, against 387 with the nearest exit)

With no exit within reach it walks to the nearest exit and steps out on its roomiest side. Out in the open it goes
straight on in the direction it left in (Frame.heading). It never turns: the turn is EXPAND2, and bringing the trail
home is RETURN; Q decides when each is worth more.

Valid: outside, when the cell straight ahead is one I can step to and not my land; inside, when there is an exit (and,
standing on it, a way out of it).

In a training game with a Style (style.py), out in the open it now and then takes one step sideways instead (a jog),
never back and never onto my land; the expedition's direction is kept, so it carries on outward afterwards.
"""

import numpy as np

from candidate.agent.behavior import paths
from candidate.agent.behavior.common import PERPENDICULAR, in_land, safe_dest, toward
from candidate.agent.behavior.frame import Frame
from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import DX, DY, step

UNIT = {d: np.array([DY[d], DX[d]], dtype=np.float64) for d in Direction}      # (y, x) of each direction


def edge_mask(land: np.ndarray) -> np.ndarray:
    """(50, 50) bool: the cells of `land` with a 4-neighbour that is not land (off the board counts as not land)."""
    padded = np.pad(land, 1, constant_values=False)
    inner = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return land & ~inner


def border_line(edge: np.ndarray, cell: int, radius: int) -> tuple[np.ndarray, float]:
    """The border at `cell`: (b, a), b the unit direction of the edge cells within `radius` (Chebyshev) of it, a how
    clearly they form a line: (l1 - l2) / (l1 + l2) of their coordinates' covariance, 0 for a blob or a single cell."""
    y, x = divmod(cell, BOARD_WIDTH)
    y0, y1, x0, x1 = max(0, y - radius), min(BOARD_HEIGHT, y + radius + 1), max(0, x - radius), min(BOARD_WIDTH, x + radius + 1)
    ys, xs = np.nonzero(edge[y0:y1, x0:x1])
    if ys.size < 2:
        return np.zeros(2), 0.0
    pts = np.stack([ys, xs], axis=1).astype(np.float64)
    cov = np.cov(pts, rowvar=False, bias=True)
    values, vectors = np.linalg.eigh(cov)                       # ascending
    total = float(values.sum())
    if total <= 0:
        return np.zeros(2), 0.0
    return vectors[:, 1], float((values[1] - values[0]) / total)


def angle(direction: Direction, line: tuple[np.ndarray, float]) -> float:
    """1 - a |d . b|: 1 straight out of the border, 0 along a clear border line."""
    b, a = line
    return 1.0 - a * abs(float(UNIT[direction] @ b))


def plan(frame: Frame) -> tuple[int, Direction] | None:
    """The exit to leave from and the side to leave on (see the module notes), or None when no exit is within reach."""
    if "_expand_plan" not in frame.__dict__:
        exits, cfg = frame.exits, frame.cfg
        score = exits.score.reshape(-1)
        walk = exits.walk.reshape(-1).astype(np.float64)
        near = np.flatnonzero(np.isfinite(score) & (walk <= cfg.expand_max_walk))
        result, best_key = None, None
        if near.size:
            land = bitboard.to_mask(frame.snap.my_land_bb)
            edge = edge_mask(land)
            for cell in near:
                cell = int(cell)
                line = border_line(edge, cell, cfg.expand_border_radius)
                y, x = divmod(cell, BOARD_WIDTH)
                for d in Direction:
                    ny, nx = y + DY[d], x + DX[d]
                    if not (0 <= ny < BOARD_HEIGHT and 0 <= nx < BOARD_WIDTH) or land[ny, nx]:
                        continue
                    key = (np.floor(angle(d, line) * 4 + 1e-9) / 4, -walk[cell], float(score[cell]))
                    if best_key is None or key > best_key:
                        best_key, result = key, (cell, d)
        frame.__dict__["_expand_plan"] = result
    return frame.__dict__["_expand_plan"]


def nearest_exit(frame: Frame) -> int | None:
    """The exit nearest to me inside my land; of equally near ones, the best scored. None without one."""
    if "_nearest_exit" not in frame.__dict__:
        exits = frame.exits
        score = exits.score.reshape(-1)
        usable = np.isfinite(score)
        cell = None
        if usable.any():
            walk = np.where(usable, exits.walk.astype(np.float64), np.inf)
            nearest = usable & (walk == walk.min())
            cell = int(np.argmax(np.where(nearest, score, -np.inf)))
        frame.__dict__["_nearest_exit"] = cell
    return frame.__dict__["_nearest_exit"]


def chosen_exit(frame: Frame) -> int | None:
    """The exit EXPAND heads for: the planned one, or with none within reach the nearest exit; None without exits."""
    p = plan(frame)
    return p[0] if p is not None else nearest_exit(frame)


def _step_out(frame: Frame, cell: int) -> Direction | None:
    """Standing on the exit `cell`, the step out: the planned side if I can make it, else the roomiest one, else any
    way out."""
    p = plan(frame)
    if p is not None and p[0] == cell:
        dest = safe_dest(frame, p[1])
        if dest is not None and not in_land(frame, dest):
            return p[1]
    d = int(frame.exits.outward.flat[cell])
    if d >= 0:
        direction = Direction(d)
        dest = safe_dest(frame, direction)
        if dest is not None and not in_land(frame, dest):
            return direction
    for other in frame.safe:
        if not in_land(frame, step(frame.pos, other)):
            return other
    return None


def _straight(frame: Frame) -> Direction | None:
    """Out in the open: the expedition's direction, if the cell ahead can be stepped to and is not my land."""
    heading = frame.heading()
    dest = safe_dest(frame, heading)
    return heading if dest is not None and not in_land(frame, dest) else None


def _jog(frame: Frame, heading: Direction) -> Direction | None:
    """A sideways step out in the open, if the game's Style asks for one this frame and one can be made."""
    style = frame.style
    if style is None or style.jog_rate <= 0.0 or style.rng.random() >= style.jog_rate:
        return None
    options = [d for d in PERPENDICULAR[heading] if (dest := safe_dest(frame, d)) is not None and not in_land(frame, dest)]
    return style.rng.choice(options) if options else None


def valid(frame: Frame) -> bool:
    snap = frame.snap
    if snap.exposed:
        return _straight(frame) is not None
    if not snap.on_land:
        return False
    cell = chosen_exit(frame)
    if cell is None:
        return False
    return frame.pos != cell or _step_out(frame, cell) is not None


def move(frame: Frame) -> Direction | None:
    snap = frame.snap
    if snap.exposed:
        straight = _straight(frame)
        if straight is not None:
            jog = _jog(frame, straight)
            if jog is not None:
                return jog
        return straight
    if not snap.on_land:
        return None
    cell = chosen_exit(frame)
    if cell is None:
        return None
    if frame.pos == cell:
        return _step_out(frame, cell)
    inside = snap.my_land_bb & snap.passable_bb
    return toward(frame, paths.path_from_map(paths.distance_map(bitboard.from_cells([cell]), inside), frame.pos))
