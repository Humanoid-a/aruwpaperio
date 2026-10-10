"""Test-only bridge to the real engine, used to check our data structures
against ground truth. Production code must not import territorygame outside
the (future) adapter and self-play modules; tests may, because they are the
oracle."""

import dataclasses
import queue

import numpy as np

from candidate.agent import adapter
from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.cells import (
    FREE,
    MY_TERRITORY,
    MY_TRAIL,
    OPP_TERRITORY,
    OPP_TRAIL,
)
from candidate.agent.data.observation import Observation
from candidate.agent.data.types import Direction as OurDirection
from candidate.agent.data.types import MoveResult as OurMoveResult
from candidate.agent.data.world import WorldState
from territorygame.api.agent_controller import AgentController
from territorygame.api.direction import Direction
from territorygame.api.game_api import GameApi
from territorygame.api.grid_position import GridPosition
from territorygame.api.move_result import MoveResult
from territorygame.domain.game_config import GameConfig
from territorygame.engine.game_engine import GameEngine
from territorygame.engine.game_observer import GameObserver
from territorygame.helpers.movement_utils import is_valid_move


_OUR_DIRECTION = {
    Direction.NORTH: OurDirection.NORTH,
    Direction.EAST: OurDirection.EAST,
    Direction.SOUTH: OurDirection.SOUTH,
    Direction.WEST: OurDirection.WEST,
}
_OUR_RESULT = {
    MoveResult.MOVED: OurMoveResult.NORMAL,
    MoveResult.CAPTURED: OurMoveResult.CAPTURED,
    MoveResult.DIED: OurMoveResult.DIED,
    MoveResult.INVALID: OurMoveResult.INVALID,
}


class _Collector(GameObserver):
    def __init__(self) -> None:
        self.snapshots: "queue.Queue" = queue.Queue()

    def on_game_state_changed(self, snapshot) -> None:
        self.snapshots.put(snapshot)


def play_match(
    controllers: list[AgentController], turns: int, respawns: list[tuple[int, int]] | None = None
) -> list:
    """Run a full match on the default 50 x 50 config and return every
    published snapshot: snapshot 0 is the initial board, snapshot j the board
    after j turns (players alternate, player 0 first). `respawns` optionally
    moves the two respawn cells, as (y, x), e.g. closer together."""
    overrides = {}
    if respawns is not None:
        overrides["respawn_positions"] = [GridPosition(x, y) for y, x in respawns]
    config = dataclasses.replace(
        GameConfig.load_default(), turns_per_player=turns, autoplay_turn_delay_millis=0, **overrides
    )
    assert (config.board_width, config.board_height) == (BOARD_WIDTH, BOARD_HEIGHT)
    engine = GameEngine(config, controllers)
    collector = _Collector()
    engine.add_observer(collector)
    engine.reset(controllers)
    snapshots = [collector.snapshots.get(timeout=10)]
    engine.start()
    while not snapshots[-1].game_over:
        snapshots.append(collector.snapshots.get(timeout=60))
    return snapshots


def _flat(position) -> int:
    return position.y * BOARD_WIDTH + position.x


def world_from_snapshot(snapshot, me_index: int) -> WorldState:
    """The engine's true board as a WorldState seen by player `me_index`."""
    me, opp = snapshot.players[me_index], snapshot.players[1 - me_index]
    territory = np.full((BOARD_HEIGHT, BOARD_WIDTH), FREE, dtype=np.int8)
    trail = np.full((BOARD_HEIGHT, BOARD_WIDTH), FREE, dtype=np.int8)
    for y, row in enumerate(snapshot.cells):
        for x, cell in enumerate(row):
            if cell.territory_owner is not None:
                territory[y, x] = MY_TERRITORY if cell.territory_owner == me.id else OPP_TERRITORY
            if cell.trail_owner is not None:
                trail[y, x] = MY_TRAIL if cell.trail_owner == me.id else OPP_TRAIL
    return WorldState(
        territory,
        trail,
        _flat(me.position),
        _flat(opp.position),
        [_flat(p) for p in me.trail],
        [_flat(p) for p in opp.trail],
        me.territory_count,
        opp.territory_count,
    )


def observation_from_game(game, turns_played: int) -> Observation:
    """The production adapter's translation of a live GameApi (the last move's
    result is not tracked here, so it is left unset)."""
    return adapter.observation_from_game(game, turns_played, None)


class Recording(AgentController):
    """Wraps a controller and records the Observation it faces at the start of
    each turn (not each retry) into `sink` as (player_index, observation)."""

    def __init__(self, inner: AgentController, player_index: int, total_turns: int, sink: list) -> None:
        self._inner = inner
        self._player_index = player_index
        self._total_turns = total_turns
        self._sink = sink
        self._last_turns_left: int | None = None

    def take_turn(self, game) -> None:
        turns_left = game.get_remaining_turns()
        if turns_left != self._last_turns_left:
            self._last_turns_left = turns_left
            self._sink.append(
                (self._player_index, observation_from_game(game, self._total_turns - turns_left))
            )
        self._inner.take_turn(game)

    def get_debug_state(self):
        return self._inner.get_debug_state()


class _LoggingGame(GameApi):
    """Passes everything through to the real GameApi and logs each move."""

    def __init__(self, inner: GameApi, player_index: int, attempts: list) -> None:
        self._inner, self._player_index, self._attempts = inner, player_index, attempts

    def get_agent_position(self):
        return self._inner.get_agent_position()

    def get_respawn_position(self):
        return self._inner.get_respawn_position()

    def get_owned_territory_cell_count(self):
        return self._inner.get_owned_territory_cell_count()

    def get_opponent_territory_cell_count(self):
        return self._inner.get_opponent_territory_cell_count()

    def get_remaining_turns(self):
        return self._inner.get_remaining_turns()

    def get_active_trail(self):
        return self._inner.get_active_trail()

    def get_visible_grid(self):
        return self._inner.get_visible_grid()

    def get_board_width(self):
        return self._inner.get_board_width()

    def get_board_height(self):
        return self._inner.get_board_height()

    def move(self, direction):
        result = self._inner.move(direction)
        self._attempts.append((self._player_index, _OUR_DIRECTION[direction], _OUR_RESULT[result]))
        return result


class MoveLogger(AgentController):
    """Wraps a controller and logs every move it attempts, in order, as
    (player_index, Direction, MoveResult) in our own types. INVALID attempts
    are logged too; the engine asks again after one."""

    def __init__(self, inner: AgentController, player_index: int, attempts: list) -> None:
        self._inner, self._player_index, self._attempts = inner, player_index, attempts

    def take_turn(self, game) -> None:
        self._inner.take_turn(_LoggingGame(game, self._player_index, self._attempts))

    def get_debug_state(self):
        return self._inner.get_debug_state()


class Marcher(AgentController):
    """Walks east along its row, then north along the east edge, leaving a
    trail straight across the other player's territory."""

    def take_turn(self, game) -> None:
        for direction in (Direction.EAST, Direction.NORTH, Direction.SOUTH, Direction.WEST):
            if is_valid_move(game, direction):
                game.move(direction)
                return
        game.move(Direction.EAST)


class Pacer(AgentController):
    """Stays inside its own 3x3 start: one step north, then east and west forever."""

    def __init__(self) -> None:
        self._turn = 0

    def take_turn(self, game) -> None:
        if self._turn == 0:
            direction = Direction.NORTH
        else:
            direction = Direction.EAST if self._turn % 2 == 1 else Direction.WEST
        self._turn += 1
        game.move(direction)
