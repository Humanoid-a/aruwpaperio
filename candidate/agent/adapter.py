"""Translation between the framework's types and ours.

This is the only module under candidate/agent that imports territorygame.
Everything else works on Observation, Direction and MoveResult from
agent.data, so it can be tested without the framework and survives upstream
changes. It holds no state: the per-match memory lives in agent.runtime.

Conventions that differ between the two sides
  * Cells: the framework's GridPosition is (x, y); ours are flat indices
    c = BOARD_WIDTH * y + x (see geometry.grid).
  * MoveResult: the framework's MOVED is our NORMAL.
  * Direction: the framework orders NORTH, SOUTH, EAST, WEST; ours N, E, S, W.
"""

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, WINDOW_SIZE
from candidate.agent.data.cells import FREE, MY_TERRITORY, MY_TRAIL, OPP_TERRITORY, OPP_TRAIL
from candidate.agent.data.observation import Observation
from candidate.agent.data.types import Direction, MoveResult
from territorygame.api.direction import Direction as FrameworkDirection
from territorygame.api.game_api import GameApi
from territorygame.api.grid_position import GridPosition
from territorygame.api.move_result import MoveResult as FrameworkMoveResult
from territorygame.api.occupant_view import OccupantView
from territorygame.api.territory_view import TerritoryView


class AdapterError(RuntimeError):
    """The framework handed us something our types cannot represent."""


_DIRECTION_TO_FRAMEWORK = {
    Direction.NORTH: FrameworkDirection.NORTH,
    Direction.EAST: FrameworkDirection.EAST,
    Direction.SOUTH: FrameworkDirection.SOUTH,
    Direction.WEST: FrameworkDirection.WEST,
}
_DIRECTION_FROM_FRAMEWORK = {v: k for k, v in _DIRECTION_TO_FRAMEWORK.items()}
_RESULT_FROM_FRAMEWORK = {
    FrameworkMoveResult.MOVED: MoveResult.NORMAL,
    FrameworkMoveResult.CAPTURED: MoveResult.CAPTURED,
    FrameworkMoveResult.DIED: MoveResult.DIED,
    FrameworkMoveResult.INVALID: MoveResult.INVALID,
}


def direction_to_framework(direction: Direction) -> FrameworkDirection:
    return _DIRECTION_TO_FRAMEWORK[Direction(direction)]


def direction_from_framework(direction: FrameworkDirection) -> Direction:
    return _DIRECTION_FROM_FRAMEWORK[direction]


def result_from_framework(result: FrameworkMoveResult) -> MoveResult:
    return _RESULT_FROM_FRAMEWORK[result]


def to_flat(position: GridPosition) -> int:
    return position.y * BOARD_WIDTH + position.x


def from_flat(cell: int) -> GridPosition:
    y, x = divmod(cell, BOARD_WIDTH)
    return GridPosition(x, y)


def check_board(game: GameApi) -> None:
    """Fail loudly if the framework's board is not the size our constants
    assume (see agent.config)."""
    size = (game.get_board_width(), game.get_board_height())
    if size != (BOARD_WIDTH, BOARD_HEIGHT):
        raise AdapterError(f"board is {size[0]}x{size[1]}, the agent is built for {BOARD_WIDTH}x{BOARD_HEIGHT}")


def observation_from_game(game: GameApi, turn: int, last_result: MoveResult | None) -> Observation:
    """Everything the framework tells us this turn, as an Observation.

    `turn` and `last_result` are not available from the framework (it only
    reports turns left, and returns each result from move()), so the caller,
    which keeps the match state, supplies them.

    The trail layer under an agent is hidden by the framework. Under my own
    agent it is rebuilt from my active trail; under the opponent it is a
    documented guess (see Observation): a trail is assumed there exactly when
    the opponent is not standing on its own territory.
    """
    grid = game.get_visible_grid()
    rows, cols = len(grid), len(grid[0])
    if not (0 < rows <= WINDOW_SIZE and 0 < cols <= WINDOW_SIZE):
        raise AdapterError(f"unexpected visible window of {rows}x{cols} cells")
    origin = grid[0][0].position
    my_trail = [to_flat(p) for p in game.get_active_trail()]
    my_trail_cells = set(my_trail)

    territory = np.zeros((rows, cols), dtype=np.int8)
    trail = np.zeros((rows, cols), dtype=np.int8)
    opp_seen_at = None
    hidden: list[int] = []
    for dy, row in enumerate(grid):
        for dx, cell in enumerate(row):
            if cell.territory == TerritoryView.SELF:
                territory[dy, dx] = MY_TERRITORY
            elif cell.territory == TerritoryView.OPPONENT:
                territory[dy, dx] = OPP_TERRITORY
            occupant = cell.occupant
            if occupant == OccupantView.EMPTY:
                continue
            if occupant == OccupantView.SELF_TRAIL:
                trail[dy, dx] = MY_TRAIL
            elif occupant == OccupantView.OPPONENT_TRAIL:
                trail[dy, dx] = OPP_TRAIL
            elif occupant == OccupantView.SELF_AGENT:
                flat_cell = to_flat(cell.position)
                if flat_cell in my_trail_cells:
                    trail[dy, dx] = MY_TRAIL
                else:
                    hidden.append(flat_cell)  # could hide the opponent's trail (after a respawn)
            elif occupant == OccupantView.OPPONENT_AGENT:
                opp_seen_at = to_flat(cell.position)
                trail[dy, dx] = FREE if territory[dy, dx] == OPP_TERRITORY else OPP_TRAIL
                hidden.append(opp_seen_at)

    return Observation(
        turn=turn,
        my_pos=to_flat(game.get_agent_position()),
        respawn=to_flat(game.get_respawn_position()),
        my_trail=np.array(my_trail, dtype=np.int16),
        turns_left=game.get_remaining_turns(),
        my_count=game.get_owned_territory_cell_count(),
        opp_count=game.get_opponent_territory_cell_count(),
        window_territory=territory,
        window_trail=trail,
        window_origin=(origin.y, origin.x),
        opp_seen_at=opp_seen_at,
        last_result=last_result,
        trail_hidden=tuple(hidden),
    )


def send_move(game: GameApi, direction: Direction) -> MoveResult:
    """Make a move and report the result in our own type."""
    return result_from_framework(game.move(direction_to_framework(direction)))


def fallback_move(game: GameApi) -> MoveResult | None:
    """Last resort when our own pipeline failed: try the framework's four
    moves in order until one is accepted. Pure framework calls, so it works
    even if translation is what broke. It does not look for danger. Returns
    None if every move was refused (for example, because a move had already
    been made this turn)."""
    for direction in FrameworkDirection:
        result = game.move(direction)
        if result != FrameworkMoveResult.INVALID:
            return result_from_framework(result)
    return None
