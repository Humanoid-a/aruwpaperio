"""Small hand-built boards and observations for the tests."""

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, WINDOW_SIZE
from candidate.agent.data.cells import FREE, MY_TERRITORY, MY_TRAIL, OPP_TERRITORY, OPP_TRAIL
from candidate.agent.data.observation import Observation
from candidate.agent.data.types import Side
from candidate.agent.data.world import WorldState
from candidate.agent.geometry.grid import flat


def window_bounds(y: int, x: int) -> tuple[int, int, int, int]:
    """(y0, x0, h, w) of the window the framework would show for an agent at (y, x)."""
    half = WINDOW_SIZE // 2
    y0, x0 = max(0, y - half), max(0, x - half)
    y1, x1 = min(BOARD_HEIGHT - 1, y + half), min(BOARD_WIDTH - 1, x + half)
    return y0, x0, y1 - y0 + 1, x1 - x0 + 1


def square(cy: int, cx: int, half: int = 1) -> list[tuple[int, int]]:
    """The (2*half+1)-square of (y, x) cells around (cy, cx): a starting territory by default."""
    return [(y, x) for y in range(cy - half, cy + half + 1) for x in range(cx - half, cx + half + 1)]


def make_observation(
    pos: tuple[int, int] = (25, 4),
    *,
    turn: int = 0,
    territory: dict[tuple[int, int], int] | None = None,
    trail: dict[tuple[int, int], int] | None = None,
    my_trail: list[tuple[int, int]] = (),
    opp_at: tuple[int, int] | None = None,
    respawn: tuple[int, int] = (25, 4),
    turns_left: int = 2000,
    my_count: int = 9,
    opp_count: int = 9,
    last_result=None,
    trail_hidden: tuple[int, ...] = (),
) -> Observation:
    """A consistent Observation. `territory` and `trail` map board (y, x) cells
    to layer codes; my_trail cells are drawn into the trail layer for you."""
    y0, x0, h, w = window_bounds(*pos)
    window_territory = np.zeros((h, w), dtype=np.int8)
    window_trail = np.zeros((h, w), dtype=np.int8)
    layers = ((window_territory, territory or {}), (window_trail, dict(trail or {})))
    for cell in my_trail:
        layers[1][1][cell] = MY_TRAIL
    for array, cells in layers:
        for (y, x), code in cells.items():
            if y0 <= y < y0 + h and x0 <= x < x0 + w:
                array[y - y0, x - x0] = code
    return Observation(
        turn=turn,
        my_pos=flat(*pos),
        respawn=flat(*respawn),
        my_trail=np.array([flat(*c) for c in my_trail], dtype=np.int16),
        turns_left=turns_left,
        my_count=my_count,
        opp_count=opp_count,
        window_territory=window_territory,
        window_trail=window_trail,
        window_origin=(y0, x0),
        opp_seen_at=None if opp_at is None else flat(*opp_at),
        last_result=last_result,
        trail_hidden=trail_hidden,
    )


def put_territory(world: WorldState, side: Side, cells) -> None:
    """Mark (y, x) cells as `side`'s territory and refresh both counts."""
    code = MY_TERRITORY if side == Side.ME else OPP_TERRITORY
    for cell in cells:
        world.territory[cell] = code
    world.my_count = int((world.territory == MY_TERRITORY).sum())
    world.opp_count = int((world.territory == OPP_TERRITORY).sum())


def put_trail(world: WorldState, side: Side, path) -> None:
    """Give `side` a trail along the (y, x) path, oldest first, and put its
    agent on the last cell, as the engine does."""
    code = MY_TRAIL if side == Side.ME else OPP_TRAIL
    cells = [flat(y, x) for y, x in path]
    for cell in path:
        world.trail[cell] = code
    if side == Side.ME:
        world.my_trail, world.my_pos = cells, cells[-1]
    else:
        world.opp_trail, world.opp_pos = cells, cells[-1]


def put_agent(world: WorldState, side: Side, cell: tuple[int, int]) -> None:
    if side == Side.ME:
        world.my_pos = flat(*cell)
    else:
        world.opp_pos = flat(*cell)


def make_world() -> WorldState:
    """A mid-match world that exercises both layers.

    I start in a 3x3 around (25, 4) and have walked out east, ending at (26, 7).
    The opponent started in a 3x3 around (25, 45) and has a trail that crosses
    my territory at (24, 5) and (24, 4): an opponent trail lying on my land.
    """
    world = WorldState.empty(my_pos=flat(26, 7), opp_pos=flat(24, 4))
    for cell in square(25, 4):
        world.territory[cell] = MY_TERRITORY
    for cell in square(25, 45):
        world.territory[cell] = OPP_TERRITORY
    world.my_count = world.opp_count = 9

    my_trail = [(25, 6), (25, 7), (26, 7)]
    opp_trail = [(24, 7), (24, 6), (24, 5), (24, 4)]
    for cell in my_trail:
        world.trail[cell] = MY_TRAIL
    for cell in opp_trail:
        world.trail[cell] = OPP_TRAIL
    world.my_trail = [flat(*c) for c in my_trail]
    world.opp_trail = [flat(*c) for c in opp_trail]
    assert world.territory[24, 4] == MY_TERRITORY and world.trail[24, 4] == OPP_TRAIL
    assert world.territory[25, 6] == FREE
    return world
