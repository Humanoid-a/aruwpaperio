"""A hand-driven two-player game on the framework's real rules, for testing
the run-time layer turn by turn: I play through an AgentRuntime, and the test
makes the opponent's moves itself, so any situation can be staged."""

from candidate.agent import adapter
from candidate.agent.config import WINDOW_SIZE
from candidate.agent.data.types import Direction, MoveResult, Side
from candidate.agent.data.world import WorldState
from candidate.agent.rules.match import MatchRules
from candidate.agent.runtime import AgentRuntime, TurnView
from candidate.tests.rules_oracle import real_state_from_world, world_from_real
from territorygame.domain.player_id import PlayerId
from territorygame.engine.game_api_impl import GameApiImpl
from territorygame.rules.move_resolver import MoveResolver
from territorygame.rules.respawn_service import RespawnService
from territorygame.rules.territory_resolver import TerritoryResolver
from territorygame.visibility.visibility_service import VisibilityService


class LiveGame:
    def __init__(self, world: WorldState, rules: MatchRules) -> None:
        self.rules = rules
        self.state = real_state_from_world(world, rules)
        resolver = MoveResolver(RespawnService(), TerritoryResolver())
        visibility = VisibilityService(WINDOW_SIZE)
        self.me = GameApiImpl(self.state, PlayerId(0), resolver, visibility)
        self.opp = GameApiImpl(self.state, PlayerId(1), resolver, visibility)

    def world(self) -> WorldState:
        return world_from_real(self.state)

    def my_turn(self, runtime: AgentRuntime, direction: Direction) -> tuple[TurnView, MoveResult]:
        """One whole turn of mine: begin, then move in `direction`."""
        view = self.begin(runtime)
        return view, runtime.move(self.me, direction)

    def begin(self, runtime: AgentRuntime) -> TurnView:
        self.me.reset_for_new_turn()
        return runtime.begin_turn(self.me)

    def opponent_moves(self, direction: Direction) -> MoveResult:
        self.opp.reset_for_new_turn()
        return adapter.result_from_framework(self.opp.move(adapter.direction_to_framework(direction)))
