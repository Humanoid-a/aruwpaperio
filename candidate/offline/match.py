"""Running headless matches on the real engine: the one place in candidate/ that starts a GameEngine.

play() runs a match to the end and returns a MatchResult (final counts, deaths and kills per player, how many turns the
engine reported as failed). Everything that plays a match without the GUI goes through it: collection (collect.py),
replay (replay.py), the runtime check (runtime_check.py) and the engine tests (tests/engine_support.py). Two engine
details are handled here once, so no caller has to know them:

  * reset() runs on the engine's thread and clears its running flag, while start() sets that flag at once on the
    caller's thread. Calling them back to back makes the run loop exit before the first turn, so play() starts only
    after the reset has published the fresh board.
  * A 2000-turn match publishes 4001 board snapshots; keeping them all costs gigabytes. play() keeps only the latest
    unless asked (keep=True, for the tests and the runtime check, which compare every board).
  * Building those snapshots is itself a quarter of a headless game's time: after every turn the engine copies all
    2,500 cells into new Python objects and counts both territories cell by cell, for observers such as the GUI. When
    nobody looks at the boards (no keep, no watch of the caller's), play() runs _ResultOnlyEngine, which publishes
    them without cells or players until the last one. The turns themselves are the engine's, unchanged.

Taped wraps any controller and writes down the move the engine accepted on each of its turns ('N', 'E', 'S', 'W',
or '-' for a turn that ended without one). Both players' tapes and the config are enough to replay the match exactly
(Replayer): the engine is deterministic given the moves, so the replay passes through exactly the same boards.
"""

import dataclasses
import threading
from dataclasses import dataclass, field

from territorygame.api.agent_controller import AgentController
from territorygame.api.direction import Direction as FrameworkDirection
from territorygame.api.game_api import GameApi
from territorygame.api.grid_position import GridPosition
from territorygame.api.move_result import MoveResult as FrameworkResult
from territorygame.domain.game_config import GameConfig
from territorygame.engine.game_engine import GameEngine
from territorygame.engine.game_observer import GameObserver
from territorygame.engine.game_snapshot import GameSnapshot

LETTER = {FrameworkDirection.NORTH: "N", FrameworkDirection.EAST: "E", FrameworkDirection.SOUTH: "S", FrameworkDirection.WEST: "W"}
DIRECTION = {v: k for k, v in LETTER.items()}

FAR = None                                          # the default config's respawns, (25, 4) and (25, 45) as (y, x)
CLOSE = ((25, 18), (25, 31))                        # 13 cells apart, (y, x) per player: the players meet


@dataclass(frozen=True)
class MatchResult:
    counts: tuple[int, int]
    deaths: tuple[int, int]
    kills: tuple[int, int]
    failed_turns: int                               # turns the engine reported an error for (forfeits, exceptions)
    turns: int
    respawns: tuple | None
    snapshots: list = field(default_factory=list, repr=False)   # every board from the fresh one on, if kept


class Watch(GameObserver):
    """Follows a match: the latest board (what the next turn faces, since observers run before the next turn), the
    failed turns, and every board if `keep`."""

    def __init__(self, keep: bool = False) -> None:
        self.keep = keep
        self.snapshots: list = []
        self.latest = None
        self.failed = 0
        self.published = threading.Event()
        self.done = threading.Event()

    def on_game_state_changed(self, snapshot) -> None:
        self.latest = snapshot
        if self.keep:
            self.snapshots.append(snapshot)
        self.published.set()
        if snapshot.error_message:                         # set exactly when the turn just played failed
            self.failed += 1
        if snapshot.game_over:
            self.done.set()


def config_for(turns: int, respawns=FAR) -> GameConfig:
    """The default config with `turns` per player, no pause between turns, and the given respawns ((y, x) per player)."""
    overrides = {}
    if respawns is not None:
        overrides["respawn_positions"] = [GridPosition(x, y) for y, x in respawns]
    return dataclasses.replace(GameConfig.load_default(), turns_per_player=turns, autoplay_turn_delay_millis=0, **overrides)


class _ResultOnlyEngine(GameEngine):
    """The framework's engine, publishing a board without its cells and players until the match is over (see the module
    notes). The error message and game_over flag are published as usual, since Watch counts failed turns with them."""

    def _build_snapshot(self) -> GameSnapshot:
        if self._state.is_game_over() or self._halted:
            return super()._build_snapshot()
        board = self._state.get_board()
        return GameSnapshot(
            board.get_width(), board.get_height(), (), (), self._state.get_active_player_id(),
            self._state.get_last_move_result(), self._config.visibility_window_size, False,
            self._state.get_last_turn_error(),
        )


def play(controllers: list[AgentController], turns: int, respawns=FAR, timeout: float = 3600.0, keep: bool = False,
         watch: Watch | None = None, on_engine=None) -> MatchResult:
    """Play a match to the end. Pass your own `watch` to follow it while it runs (it then sees every board).
    `on_engine(engine)` is called with the engine before the match starts (offline.oracle reads the truth from it)."""
    lean = watch is None and not keep
    watch = watch if watch is not None else Watch(keep)
    engine = (_ResultOnlyEngine if lean else GameEngine)(config_for(turns, respawns), controllers)
    if on_engine is not None:
        on_engine(engine)
    engine.add_observer(watch)
    engine.reset(controllers)
    if not watch.published.wait(60):                   # see the module notes: start only after the reset is done
        raise TimeoutError("the engine did not publish the fresh match")
    engine.start()
    if not watch.done.wait(timeout):
        engine.pause()
        raise TimeoutError(f"the match did not finish within {timeout} s")
    players = watch.latest.players
    return MatchResult(
        counts=(players[0].territory_count, players[1].territory_count),
        deaths=(players[0].death_count, players[1].death_count),
        kills=(players[0].kill_count, players[1].kill_count),
        failed_turns=watch.failed, turns=turns, respawns=None if respawns is None else tuple(map(tuple, respawns)),
        snapshots=watch.snapshots,
    )


# ---- the tape --------------------------------------------------------------------------------------------------------


class _TapingGame(GameApi):
    def __init__(self, inner: GameApi, on_accept) -> None:
        self._inner, self._on_accept = inner, on_accept

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
        if result != FrameworkResult.INVALID:
            self._on_accept(direction)
        return result


class Taped(AgentController):
    """Wraps a controller; `moves` holds one letter per turn it was given."""

    def __init__(self, inner: AgentController) -> None:
        self.inner = inner
        self.moves: list[str] = []
        self._turns_left: int | None = None

    def take_turn(self, game: GameApi) -> None:
        left = game.get_remaining_turns()
        if left != self._turns_left:                       # a new turn (not the engine asking again after INVALID)
            self._turns_left = left
            self.moves.append("-")
        self.inner.take_turn(_TapingGame(game, self._accept))

    def _accept(self, direction) -> None:
        self.moves[-1] = LETTER[direction]

    def get_debug_state(self):
        return self.inner.get_debug_state()

    @property
    def tape(self) -> str:
        return "".join(self.moves)


class ReplayDiverged(RuntimeError):
    pass


class Replayer(AgentController):
    """Plays back a tape. A '-' is replayed as a failed turn (an exception, which the engine turns into a forfeit)."""

    def __init__(self, tape: str) -> None:
        self.tape = tape
        self.index = -1
        self._turns_left: int | None = None

    def take_turn(self, game: GameApi) -> None:
        left = game.get_remaining_turns()
        if left != self._turns_left:
            self._turns_left = left
            self.index += 1
        letter = self.tape[self.index]
        if letter == "-":
            raise ReplayDiverged("replaying a turn that had no accepted move")
        if game.move(DIRECTION[letter]) == FrameworkResult.INVALID:
            raise ReplayDiverged(f"turn {self.index}: the engine refused {letter}, which it accepted in the recording")

    def get_debug_state(self):
        return f"replay {self.index}"


class AgentAdapter(AgentController):
    """Our agent.controller.Controller as the framework's AgentController."""

    def __init__(self, agent) -> None:
        self.agent = agent

    def take_turn(self, game) -> None:
        self.agent.take_turn(game)

    def get_debug_state(self):
        return self.agent.debug_state()
