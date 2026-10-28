"""The controller never raises, recovers with a legal move, and reports."""

import json


from candidate.agent import adapter
from candidate.agent.controller import Controller
from candidate.agent.data.types import Direction, MoveResult
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.geometry.grid import flat
from candidate.agent.probe import ProbePolicy
from candidate.agent.rules.match import MatchRules
from candidate.agent.q.strategy import QStrategy
from candidate.candidate_controller import CandidateController
from candidate.tests.live import LiveGame
from territorygame.controller.available_controllers import ALL
from territorygame.domain.player_id import PlayerId

RULES = MatchRules.from_respawns(flat(25, 4), flat(25, 45))
E, W = Direction.EAST, Direction.WEST


class Scripted:
    """A strategy that does whatever a test tells it to."""

    def __init__(self, choose=None):
        self._choose = choose or (lambda view: Direction.EAST)
        self.results = []

    def reset(self):
        pass

    def choose(self, view):
        return self._choose(view)

    def on_result(self, view, direction, result):
        self.results.append(result)

    def debug_state(self):
        return "SCRIPT"


def boom(view):
    raise RuntimeError("boom")


def one_turn(controller, live):
    live.me.reset_for_new_turn()
    controller.take_turn(live.me)  # must not raise
    assert live.me.was_move_made_this_turn()


def opponent_paces(live):
    live.opponent_moves(E if live.world().opp_pos == flat(25, 45) else W)


# ---- never raising ------------------------------------------------------


def test_a_strategy_that_raises_still_makes_a_move_and_the_failure_is_recorded():
    diag = Diagnostics()
    controller = Controller(Scripted(boom), diag)
    live = LiveGame(RULES.initial_world(), RULES)
    one_turn(controller, live)
    assert [a.kind for a in diag.errors()] == ["CONTROLLER_EXCEPTION"] and diag.counts["exceptions"] == 1
    assert "boom" in diag.errors()[0].detail


def test_the_fallback_prefers_a_safe_move():
    from candidate.agent.data.types import Side
    from candidate.tests.builders import put_trail

    world = RULES.initial_world()
    put_trail(world, Side.ME, [(25, 6), (25, 7), (26, 7), (26, 6)])   # head (26, 6): north and east are my trail
    live = LiveGame(world, RULES)
    one_turn(Controller(Scripted(boom), Diagnostics()), live)
    after = live.world()
    assert after.my_pos == flat(26, 5) and after.my_trail == []       # stepped home (closing the loop), did not die
    assert after.my_count > world.my_count


def test_a_strategy_that_returns_nonsense_is_handled_the_same_way():
    for nonsense in (None, 9, "north"):
        diag = Diagnostics()
        controller = Controller(Scripted(lambda view, n=nonsense: n), diag)
        live = LiveGame(RULES.initial_world(), RULES)
        one_turn(controller, live)
        assert diag.counts["exceptions"] == 1


def test_a_failure_while_translating_the_observation_falls_back_to_a_framework_move(monkeypatch):
    diag = Diagnostics()
    controller = Controller(Scripted(), diag)
    live = LiveGame(RULES.initial_world(), RULES)

    def broken(*args, **kwargs):
        raise adapter.AdapterError("the framework changed")

    monkeypatch.setattr(adapter, "observation_from_game", broken)
    one_turn(controller, live)
    assert [a.kind for a in diag.errors()] == ["CONTROLLER_EXCEPTION"]
    monkeypatch.undo()
    opponent_paces(live)
    one_turn(controller, live)             # and the next turn works again
    assert diag.counts["turns"] == 1


def test_a_normal_strategy_produces_no_errors_over_many_turns():
    diag = Diagnostics()
    controller = Controller(ProbePolicy(seed=2), diag)
    live = LiveGame(RULES.initial_world(), RULES)
    for _ in range(60):
        one_turn(controller, live)
        opponent_paces(live)
    assert diag.errors() == [] and diag.counts["turns"] == 60


# ---- reporting ----------------------------------------------------------


def nearly_finished_game(turns_left: int):
    live = LiveGame(RULES.initial_world(), RULES)
    for _ in range(10_000 - turns_left):
        live.state.decrement_remaining_turns(PlayerId(0))
    return live


def test_the_final_report_is_printed_once_at_the_last_turn_and_not_again_for_the_next_match():
    echoed: list[str] = []
    controller = Controller(Scripted(), Diagnostics(echo=echoed.append), report_every=2)
    live = nearly_finished_game(3)
    for _ in range(3):
        one_turn(controller, live)
        opponent_paces(live)
    reports = [t for t in echoed if "runtime report" in t]
    assert len(reports) == 1 and "match 1" in reports[0] and "turns 0-2 of 3" in reports[0]
    assert any(t.startswith("[candidate] turn 2") for t in echoed)

    one_turn(controller, LiveGame(RULES.initial_world(), RULES))    # a new match starts
    assert len([t for t in echoed if "runtime report" in t]) == 1    # nothing was reported twice
    assert controller.diag.match_number == 2


def test_a_per_turn_log_is_written_when_a_directory_is_given(tmp_path):
    diag = Diagnostics(log_dir=tmp_path / "logs")
    controller = Controller(ProbePolicy(seed=1), diag)
    live = LiveGame(RULES.initial_world(), RULES)
    for _ in range(5):
        one_turn(controller, live)
        opponent_paces(live)
    diag.close()
    assert (tmp_path / "logs" / ".gitignore").read_text().strip() == "*"
    lines = [json.loads(line) for line in diag.log_path.read_text().splitlines()]
    assert [line["turn"] for line in lines] == [0, 1, 2, 3, 4]
    assert {"pos", "trail", "mine", "theirs", "dir", "result", "events", "state", "ms"} <= set(lines[0])
    assert lines[0]["events"] == ["MATCH_STARTED"]


def test_the_debug_label_shows_the_state_and_the_error_count():
    diag = Diagnostics()
    controller = Controller(Scripted(), diag)
    assert controller.debug_state() == "SCRIPT"
    diag.anomaly("X", 0, "something")
    assert controller.debug_state() == "SCRIPT | err 1"


# ---- the framework-facing wrapper ---------------------------------------


def test_the_gui_registry_builds_a_working_candidate_controller_without_arguments():
    option = ALL[3]
    assert option.label == "Candidate Controller"
    controller = option.factory(1001)
    assert isinstance(controller, CandidateController)
    assert isinstance(controller._agent.strategy, QStrategy)
    assert controller._agent.diag.echo is not None and controller._agent.diag.log_dir.name == "runtime_logs"


def test_the_probe_is_still_available_as_an_explicit_strategy():
    controller = CandidateController(strategy=ProbePolicy(seed=1), diagnostics=Diagnostics())
    live = LiveGame(RULES.initial_world(), RULES)
    for _ in range(10):
        live.me.reset_for_new_turn()
        controller.take_turn(live.me)
        opponent_paces(live)
    assert isinstance(controller._agent.strategy, ProbePolicy) and controller.get_debug_state()


def test_the_wrapper_plays_turns_and_labels_itself_for_the_gui():
    controller = CandidateController(diagnostics=Diagnostics())
    live = LiveGame(RULES.initial_world(), RULES)
    for _ in range(10):
        live.me.reset_for_new_turn()
        controller.take_turn(live.me)
        opponent_paces(live)
    assert isinstance(controller.get_debug_state(), str) and controller.get_debug_state()


def test_the_wrapper_accepts_another_strategy():
    strategy = Scripted()
    controller = CandidateController(strategy=strategy, diagnostics=Diagnostics())
    live = LiveGame(RULES.initial_world(), RULES)
    live.me.reset_for_new_turn()
    controller.take_turn(live.me)
    assert strategy.results == [MoveResult.NORMAL]
