"""The run-time layer driven turn by turn against the framework's real rules.

Each scenario stages a situation with LiveGame (I play through an
AgentRuntime, the test plays the opponent) and checks the event the runtime
infers, what it does to the memory, and that nothing was flagged as an error.
"""

from candidate.agent.data.cells import FREE, MY_TERRITORY, MY_TRAIL, OPP_TERRITORY, OPP_TRAIL, UNKNOWN
from candidate.agent.data.types import Direction, MoveResult, Side
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.geometry.grid import flat
from candidate.agent.rules.match import MatchRules
from candidate.agent.runtime import AgentRuntime, Event
from candidate.tests.builders import put_territory, put_trail
from candidate.tests.live import LiveGame
from territorygame.api.move_result import MoveResult as FrameworkMoveResult
from territorygame.domain.player_id import PlayerId

N, E, S, W = Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST
ME, OPP = Side.ME, Side.OPP
RULES = MatchRules.from_respawns(flat(25, 4), flat(25, 45))


def new_runtime():
    echoed: list[str] = []
    return AgentRuntime(Diagnostics(echo=echoed.append)), echoed


def pace(live: LiveGame) -> None:
    """The opponent makes a harmless move inside its own start."""
    live.opponent_moves(E if live.world().opp_pos == flat(25, 45) else W)


def play(live, runtime, moves):
    """My moves in order, the opponent pacing after each. Returns [(view, result)]."""
    out = []
    for direction in moves:
        out.append(live.my_turn(runtime, direction))
        pace(live)
    return out


def assert_no_errors(runtime):
    assert runtime.diag.errors() == [], [str(a) for a in runtime.diag.errors()]


# ---- matches, turns and retries -----------------------------------------


def test_the_first_turn_starts_a_match_and_remembers_the_window():
    runtime, _ = new_runtime()
    live = LiveGame(RULES.initial_world(), RULES)
    view = live.begin(runtime)
    assert view.events == (Event.MATCH_STARTED,)
    assert view.obs.turn == 0 and runtime.diag.match_number == 1
    assert runtime.memory.agrees_with(view.obs)
    assert runtime.rules.opp.respawn == flat(25, 45)  # the mirrored-opponent assumption
    assert view.territory_drift == 0 and not view.retry and not view.exposed


def test_turns_are_numbered_from_zero_and_the_last_result_reaches_the_next_observation():
    runtime, _ = new_runtime()
    live = LiveGame(RULES.initial_world(), RULES)
    turns = play(live, runtime, [E, E, E])
    assert [v.obs.turn for v, _ in turns] == [0, 1, 2]
    assert live.begin(runtime).obs.last_result == MoveResult.NORMAL
    assert_no_errors(runtime)


def test_an_invalid_move_makes_the_engine_ask_again_in_the_same_turn():
    runtime, _ = new_runtime()
    rules = MatchRules.from_respawns(flat(0, 10), flat(25, 45))
    live = LiveGame(rules.initial_world(), rules)
    view, result = live.my_turn(runtime, N)  # off the board
    assert result == MoveResult.INVALID
    again = live.begin(runtime)
    assert again is view and again.retry
    assert runtime.diag.counts["retries"] == 1 and runtime.diag.counts["pred_result_ok"] == 1
    assert N not in again.legal_moves
    assert runtime.move(live.me, E) == MoveResult.NORMAL
    live.opponent_moves(E)
    assert live.begin(runtime).obs.turn == 1
    assert_no_errors(runtime)


def test_a_refused_destination_is_remembered_for_the_rest_of_the_turn():
    class RefusesOnce:
        """Stands in for an engine that refuses a move the runtime thought was fine."""

        def __init__(self, inner):
            self._inner, self.refused = inner, False

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def move(self, direction):
            if not self.refused:
                self.refused = True
                return FrameworkMoveResult.INVALID
            return self._inner.move(direction)

    runtime, _ = new_runtime()
    live = LiveGame(RULES.initial_world(), RULES)
    game = RefusesOnce(live.me)
    view = runtime.begin_turn(game)
    assert E in view.legal_moves
    assert runtime.move(game, E) == MoveResult.INVALID
    again = runtime.begin_turn(game)
    assert again.retry and again.blocked == {flat(25, 5)}
    assert E not in again.legal_moves and E not in again.safe_moves
    assert [a.kind for a in runtime.diag.errors()] == ["RULE_MISMATCH"]  # the runtime did expect that move to work
    assert runtime.move(game, W) == MoveResult.NORMAL


def test_a_new_match_on_the_same_controller_is_detected_and_starts_fresh():
    runtime, echoed = new_runtime()
    live = LiveGame(RULES.initial_world(), RULES)
    play(live, runtime, [E, E, E, S])
    other = LiveGame(RULES.initial_world(), RULES)  # the GUI's Reset: a fresh board and turn count
    view = other.begin(runtime)
    assert view.events == (Event.MATCH_STARTED,) and view.obs.turn == 0
    assert runtime.diag.match_number == 2
    assert any("match 1" in text for text in echoed), "the first match's report should have been printed"
    assert (runtime.memory.territory == UNKNOWN).sum() > 2000  # only the new window is known
    assert_no_errors(runtime)


def test_a_forfeited_turn_is_noticed():
    runtime, _ = new_runtime()
    live = LiveGame(RULES.initial_world(), RULES)
    live.begin(runtime)                                  # my turn begins, but no move is made ...
    live.state.decrement_remaining_turns(PlayerId(0))    # ... and the engine moves on
    view = live.begin(runtime)
    assert Event.FORFEITED_TURN in view.events and view.obs.last_result == MoveResult.INVALID
    assert runtime.diag.counts["forfeits"] == 1
    assert_no_errors(runtime)


# ---- captures and memory ------------------------------------------------


def test_closing_a_loop_is_a_capture_and_the_predicted_size_is_exact():
    runtime, _ = new_runtime()
    live = LiveGame(RULES.initial_world(), RULES)
    # home, out three, down two, back three along the bottom, up onto home
    turns = play(live, runtime, [E, E, E, E, S, S, W, W, W, N])
    assert [r for _, r in turns] == [MoveResult.NORMAL] * 9 + [MoveResult.CAPTURED]
    view = live.begin(runtime)
    truth = live.world()
    assert Event.CAPTURED in view.events and view.obs.last_result == MoveResult.CAPTURED
    assert view.obs.my_count == truth.my_count == 9 + 8 + 2   # eight trail cells and the two they enclose
    assert runtime.diag.counts["gain_ok"] == 1 and runtime.diag.counts["gain_off"] == 0
    assert runtime.memory.my_territory_count() == truth.my_count and view.territory_drift == 0
    assert runtime.memory.trail[25, 6] == FREE and runtime.memory.territory[25, 6] == MY_TERRITORY
    assert_no_errors(runtime)


def test_a_capture_updates_memory_for_cells_that_are_out_of_view():
    # The far side of this loop is more than five columns from where it closes, so only the
    # runtime applying the capture itself can tell memory about those cells.
    runtime, _ = new_runtime()
    live = LiveGame(RULES.initial_world(), RULES)
    moves = [E] + [E] * 7 + [S] * 3 + [W] * 7 + [N] * 2
    turns = play(live, runtime, moves)
    assert turns[-1][1] == MoveResult.CAPTURED
    view = live.begin(runtime)
    truth = live.world()
    assert Event.CAPTURED in view.events
    assert not view.obs.covers(flat(26, 11)) and not view.obs.covers(flat(25, 12))
    assert truth.territory[26, 11] == MY_TERRITORY and runtime.memory.territory[26, 11] == MY_TERRITORY
    assert runtime.memory.territory[25, 12] == MY_TERRITORY and runtime.memory.trail[25, 12] == FREE
    assert runtime.memory.my_territory_count() == truth.my_count == view.obs.my_count
    assert_no_errors(runtime)


# ---- deaths and kills ---------------------------------------------------


def test_stepping_on_my_own_trail_is_inferred_as_my_death_and_memory_resets():
    runtime, _ = new_runtime()
    live = LiveGame(RULES.initial_world(), RULES)
    turns = play(live, runtime, [E, E, E, E, S, W, N])  # the last step lands on my trail at (25, 7)
    assert turns[-1][1] == MoveResult.DIED
    view = live.begin(runtime)
    assert Event.SELF_DIED in view.events and view.obs.my_pos == flat(25, 4) and not view.exposed
    assert view.obs.my_count == 9 and runtime.memory.my_territory_count() == 9
    assert (runtime.memory.trail == MY_TRAIL).sum() == 0
    assert runtime.diag.counts["state_ok"] == 7
    assert_no_errors(runtime)


def test_an_opponent_cutting_my_trail_is_inferred_from_the_reset():
    world = RULES.initial_world()
    put_trail(world, ME, [(25, 6), (25, 7)])
    put_trail(world, OPP, [(24, 6)])
    runtime, _ = new_runtime()
    live = LiveGame(world, RULES)
    live.my_turn(runtime, N)    # I step to (24, 7) ...
    live.opponent_moves(S)      # ... and the opponent steps onto my trail at (25, 6)
    view = live.begin(runtime)
    assert Event.KILLED_BY_OPPONENT in view.events
    assert view.obs.my_pos == flat(25, 4) and not view.exposed and view.obs.my_count == 9
    assert runtime.memory.my_territory_count() == 9
    assert runtime.diag.counts["state_explained_by_kill"] == 1
    assert_no_errors(runtime)


def test_stepping_on_the_opponents_trail_is_a_kill_and_resets_what_i_remember_of_them():
    world = RULES.initial_world()
    put_trail(world, ME, [(25, 6), (25, 7)])
    put_trail(world, OPP, [(25, 8), (24, 8)])   # their trail runs right beside my head
    runtime, _ = new_runtime()
    live = LiveGame(world, RULES)
    view = live.begin(runtime)
    assert runtime.memory.trail[25, 8] == OPP_TRAIL and view.predict(E).killed
    assert runtime.move(live.me, E) == MoveResult.NORMAL
    pace(live)
    view = live.begin(runtime)
    truth = live.world()
    assert Event.KILLED_OPPONENT in view.events and truth.opp_trail == []
    assert (runtime.memory.trail == OPP_TRAIL).sum() == 0
    assert runtime.memory.territory[25, 45] == OPP_TERRITORY  # their start land, back by the rules
    assert_no_errors(runtime)


def test_land_taken_by_the_opponent_is_noticed_as_territory_lost():
    world = RULES.initial_world()
    put_territory(world, OPP, [(y, 40) for y in range(20, 31)])
    put_territory(world, ME, [(25, 38), (25, 39)])  # my land inside the opponent's coming loop
    put_trail(world, OPP, [(20, 39), (20, 38), (20, 37)] + [(y, 37) for y in range(21, 30)] + [(30, 37), (30, 38), (30, 39)])
    runtime, _ = new_runtime()
    live = LiveGame(world, RULES)
    live.begin(runtime)
    runtime.move(live.me, E)
    assert live.opponent_moves(E) == MoveResult.CAPTURED  # the opponent closes its loop, taking my two cells
    view = live.begin(runtime)
    assert Event.TERRITORY_LOST in view.events and view.obs.my_count == 9
    assert any(a.kind == "COUNT_MISMATCH" for a in runtime.diag.anomalies)
    assert_no_errors(runtime)  # losing land to the opponent is information, not an error


# ---- the derived facts --------------------------------------------------


def test_home_path_leads_back_around_my_own_trail():
    world = RULES.initial_world()
    put_trail(world, ME, [(25, 6), (25, 7), (26, 7), (26, 6)])
    runtime, _ = new_runtime()
    view = LiveGame(world, RULES).begin(runtime)
    assert view.exposed and view.home_distance == 1 and view.home_path == [flat(26, 5)]
    assert view.direction_of(view.home_path[0]) == W


def test_at_home_the_path_home_is_empty_and_every_move_is_safe():
    runtime, _ = new_runtime()
    view = LiveGame(RULES.initial_world(), RULES).begin(runtime)
    assert view.home_path == [] and view.home_distance == 0
    assert set(view.legal_moves) == set(view.safe_moves) == {N, E, S, W}


def test_safe_moves_exclude_my_own_trail_and_legal_moves_exclude_the_opponent():
    world = RULES.initial_world()
    put_trail(world, ME, [(25, 6), (25, 7), (26, 7), (26, 6)])
    put_trail(world, OPP, [(27, 6)])
    runtime, _ = new_runtime()
    view = LiveGame(world, RULES).begin(runtime)  # head (26, 6): N (25, 6) and E (26, 7) are my trail, S is the opponent
    assert set(view.legal_moves) == {N, E, W}
    assert view.safe_moves == (W,)
    assert view.predict(N).result == MoveResult.DIED and view.predict(S).result == MoveResult.INVALID
    assert view.lead == view.obs.my_count - view.obs.opp_count


def test_the_opponents_last_sighting_is_remembered_after_it_leaves_the_window():
    world = RULES.initial_world()
    put_trail(world, OPP, [(25, 8)])
    runtime, _ = new_runtime()
    live = LiveGame(world, RULES)
    first = live.begin(runtime)
    assert first.obs.opp_seen_at == flat(25, 8) and first.opp_last_seen == (flat(25, 8), 0)
    runtime.move(live.me, W)      # I step west to (25, 3) ...
    live.opponent_moves(E)        # ... and they step east to (25, 9), now out of my view
    second = live.begin(runtime)
    assert second.obs.opp_seen_at is None
    assert second.opp_last_seen == (flat(25, 8), 0)
    assert_no_errors(runtime)
