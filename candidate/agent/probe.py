"""A deliberately simple loop-maker that gives the run-time pipeline something
to chew on. It is NOT a strategy. It exists so that a live match produces
exits, captures, and (optionally) deaths and kills, which the runtime then
checks against the engine.

Each excursion: walk out of home a few steps, walk sideways a few steps, then
come back along the shortest route home that avoids the trail. Optional
extras that probe specific runtime paths:

  * hunt: step onto an adjacent opponent trail whenever one is there
    (exercises kill detection);
  * suicide_every=N: on every Nth excursion, step onto my own trail on the way
    back (exercises death handling).

It talks to the world only through a TurnView, and it follows the rule that
choose() must not advance the plan: the engine may ask again after an INVALID
move in the same turn, so the plan only advances in on_result().
"""

import random
from enum import Enum, auto

from candidate.agent.data.cells import MY_TRAIL, OPP_TRAIL
from candidate.agent.data.types import Direction, MoveResult
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.distance import shortest_path
from candidate.agent.geometry.fill import BOARD
from candidate.agent.geometry.grid import step
from candidate.agent.runtime import TurnView


class Phase(Enum):
    HOME = auto()
    OUT = auto()
    ACROSS = auto()
    BACK = auto()


_PERPENDICULAR = {
    Direction.NORTH: (Direction.EAST, Direction.WEST),
    Direction.SOUTH: (Direction.EAST, Direction.WEST),
    Direction.EAST: (Direction.NORTH, Direction.SOUTH),
    Direction.WEST: (Direction.NORTH, Direction.SOUTH),
}


class ProbePolicy:
    def __init__(self, seed: int = 0, max_out: int = 5, hunt: bool = True, suicide_every: int | None = None) -> None:
        self._seed = seed
        self.max_out = max_out
        self.hunt = hunt
        self.suicide_every = suicide_every
        self.reset()

    def reset(self) -> None:
        """Start a new match."""
        self._rng = random.Random(self._seed)
        self._phase = Phase.HOME
        self._direction: Direction | None = None
        self._left = 0
        self.excursions = 0
        self.captures = 0
        self.deaths = 0

    def debug_state(self) -> str:
        return f"{self._phase.name} {self._left}" if self._phase in (Phase.OUT, Phase.ACROSS) else self._phase.name

    # ---- deciding --------------------------------------------------------

    def choose(self, view: TurnView) -> Direction:
        if not view.safe_moves and not view.legal_moves:
            return Direction.NORTH  # nothing is allowed; the engine will refuse and the controller falls back
        if self.hunt:
            for direction in view.safe_moves:
                if view.obs.window_cell(step(view.obs.my_pos, direction))[1] == OPP_TRAIL:
                    return direction
        if not view.exposed:
            self._phase = Phase.HOME
            return self._leave_home(view)
        if self._phase == Phase.HOME:
            self._phase = Phase.BACK  # exposed without a plan (for example after a refused move): just go home
        if self._wants_suicide(view):
            for direction in view.legal_moves:
                if view.obs.window_cell(step(view.obs.my_pos, direction))[1] == MY_TRAIL:
                    return direction
        if self._phase == Phase.OUT:
            if self._left > 0 and self._can_step(view, self._direction):
                return self._direction
            self._start_across(view)
        if self._phase == Phase.ACROSS:
            if self._left > 0 and self._can_step(view, self._direction):
                return self._direction
            self._phase = Phase.BACK
        return self._go_home(view)

    def _can_step(self, view: TurnView, direction: Direction) -> bool:
        return direction in view.safe_moves

    def _wants_suicide(self, view: TurnView) -> bool:
        return bool(
            self.suicide_every
            and self.excursions > 0
            and self.excursions % self.suicide_every == 0
            and len(view.obs.my_trail) >= 4
        )

    def _leave_home(self, view: TurnView) -> Direction:
        """Walk through my own land to its edge, then step off it."""
        mine = view.my_territory_bb
        outside = BOARD & ~mine
        if not bitboard.has_cell(mine, view.obs.my_pos):
            return self._go_home(view)  # standing on someone else's land with no trail
        route = shortest_path(view.obs.my_pos, outside, mine)
        if route:
            direction = view.direction_of(route[0])
            if direction in view.safe_moves:
                if not bitboard.has_cell(mine, route[0]):
                    self._begin_out(direction)
                return direction
        return self._rng.choice(view.safe_moves or view.legal_moves)

    def _begin_out(self, direction: Direction) -> None:
        self._phase = Phase.OUT
        self._direction = direction
        longest = self.max_out * 2 if self.excursions % 5 == 4 else self.max_out
        self._left = self._rng.randint(2, longest)

    def _start_across(self, view: TurnView) -> None:
        options = [d for d in _PERPENDICULAR[self._direction] if self._can_step(view, d)]
        if options:
            self._phase = Phase.ACROSS
            self._direction = self._rng.choice(options)
            self._left = self._rng.randint(2, 4)
        else:
            self._phase = Phase.BACK

    def _go_home(self, view: TurnView) -> Direction:
        path = view.home_path
        if path:
            direction = view.direction_of(path[0])
            if direction is not None and direction in view.legal_moves:
                return direction
        return self._rng.choice(view.safe_moves or view.legal_moves)

    # ---- bookkeeping -----------------------------------------------------

    def on_result(self, view: TurnView, direction: Direction, result: MoveResult) -> None:
        """Advance the plan after the engine accepted a move."""
        if result == MoveResult.INVALID:
            return
        if result in (MoveResult.CAPTURED, MoveResult.DIED):
            self.captures += result == MoveResult.CAPTURED
            self.deaths += result == MoveResult.DIED
            self.excursions += 1
            self._phase, self._left = Phase.HOME, 0
            return
        if self._phase in (Phase.OUT, Phase.ACROSS) and direction == self._direction:
            self._left -= 1
