"""The per-turn console line showing where the opponent most probably is."""

import numpy as np
import pytest

from candidate.agent.controller import Controller, belief_legend, belief_line
from candidate.agent.data.types import Direction
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.geometry.grid import flat
from candidate.agent.probe import ProbePolicy
from candidate.agent.rules.match import MatchRules
from candidate.agent.runtime import AgentRuntime
from candidate.candidate_controller import CandidateController
from candidate.tests.live import LiveGame
from candidate.tests.test_hypotheses import make_view, store_of

E, W = Direction.EAST, Direction.WEST
FAR = MatchRules.from_respawns(flat(25, 4), flat(25, 45))
MID = MatchRules.from_respawns(flat(25, 20), flat(25, 29))


def pace(live):
    live.opponent_moves(E if live.world().opp_pos == flat(25, 45) else W)


def belief_lines(echoed):
    return [text for text in echoed if text.startswith("[belief] turn")]


class Scripted:
    def __init__(self, choose=None):
        self._choose = choose or (lambda view: E)

    def reset(self):
        pass

    def choose(self, view):
        return self._choose(view)

    def on_result(self, view, direction, result):
        pass

    def debug_state(self):
        return "SCRIPT"


# ---- BeliefView.top_positions -----------------------------------------------


def test_the_top_positions_are_listed_most_probable_first():
    store = store_of([(flat(1, 1), [], 0.1, 9), (flat(2, 2), [], 0.5, 9), (flat(3, 3), [], 0.3, 9), (flat(4, 4), [], 0.1, 9)])
    top = make_view(store).top_positions(3)
    assert [c for c, _ in top] == [flat(2, 2), flat(3, 3), flat(1, 1)]      # the tie at 0.1 goes to the lower cell
    assert [p for _, p in top] == pytest.approx([0.5, 0.3, 0.1])


def test_only_cells_with_probability_are_listed_so_there_can_be_fewer_than_asked():
    store = store_of([(flat(7, 7), [], 1.0, 9)])
    assert make_view(store).top_positions(5) == [(flat(7, 7), pytest.approx(1.0))]
    assert make_view(store).top_positions(0) == []


def test_the_listed_probabilities_are_those_of_the_position_marginal():
    rng = np.random.default_rng(2)
    store = store_of([(int(c), [], float(w), 9) for c, w in zip(rng.integers(0, 2500, 40), rng.random(40))])
    view = make_view(store)
    for cell, p in view.top_positions(5):
        assert p == pytest.approx(view.probability_at(cell))
    probabilities = [p for _, p in view.top_positions(5)]
    assert probabilities == sorted(probabilities, reverse=True)
    assert min(probabilities) >= np.sort(view.position.reshape(-1))[-5] - 1e-9


# ---- the line -------------------------------------------------------------------


def test_the_line_names_the_turn_whether_it_is_in_view_and_the_cells_as_x_y():
    runtime = AgentRuntime(Diagnostics())
    live = LiveGame(FAR.initial_world(), FAR)
    view = live.begin(runtime)
    line = belief_line(view, 5)
    assert line.startswith("[belief] turn    0 | out of view | ")
    assert "(45,25) 95.0%" in line                       # x=45, y=25: the opponent's respawn
    assert line.count("%") == 5 and "(45,24) 1.3%" in line


def test_a_visible_opponent_is_a_single_certain_cell():
    runtime = AgentRuntime(Diagnostics())
    live = LiveGame(MID.initial_world(), MID)
    for turn in range(10):
        view = live.begin(runtime)
        if view.obs.opp_seen_at is not None:
            line = belief_line(view, 5)
            x, y = view.obs.opp_seen_at % 50, view.obs.opp_seen_at // 50
            assert "| in view     |" in line and f"({x},{y}) 100.0%" in line and line.count("%") == 1
            return
        runtime.move(live.me, E if turn % 2 == 0 else W)
        live.opponent_moves(W if turn < 5 else E)
    pytest.fail("the opponent never came into view")


def test_very_small_probabilities_keep_their_precision():
    store = store_of([(flat(1, 1), [], 0.9995, 9), (flat(2, 2), [], 0.0005, 9)])
    runtime = AgentRuntime(Diagnostics())
    view = LiveGame(FAR.initial_world(), FAR).begin(runtime)
    view.belief = make_view(store)
    assert "0.050%" in belief_line(view, 2)


def test_no_belief_no_line():
    runtime = AgentRuntime(Diagnostics())
    view = LiveGame(FAR.initial_world(), FAR).begin(runtime)
    view.belief = None
    assert belief_line(view, 5) is None


# ---- when it is printed ---------------------------------------------------------


def run_turns(controller, live, turns):
    for _ in range(turns):
        live.me.reset_for_new_turn()
        controller.take_turn(live.me)
        pace(live)


def test_one_line_is_printed_on_every_turn_in_order_with_at_most_k_cells():
    echoed: list[str] = []
    controller = Controller(ProbePolicy(seed=1), Diagnostics(echo=echoed.append), show_belief=5)
    run_turns(controller, LiveGame(FAR.initial_world(), FAR), 8)
    lines = belief_lines(echoed)
    assert [int(line.split("turn")[1].split("|")[0]) for line in lines] == list(range(8))
    assert all(line.count("%") <= 5 for line in lines) and "(45,25) 95.0%" in lines[0]


def test_a_legend_is_printed_once_per_match_before_the_first_line():
    echoed: list[str] = []
    controller = Controller(ProbePolicy(seed=1), Diagnostics(echo=echoed.append), show_belief=5)
    run_turns(controller, LiveGame(FAR.initial_world(), FAR), 3)
    legend = belief_legend(5)
    assert echoed.count(legend) == 1 and echoed.index(legend) < echoed.index(belief_lines(echoed)[0])
    run_turns(controller, LiveGame(FAR.initial_world(), FAR), 2)       # a second match
    assert echoed.count(legend) == 2


def test_the_number_of_cells_follows_the_setting():
    for k in (1, 3):
        echoed: list[str] = []
        controller = Controller(ProbePolicy(seed=1), Diagnostics(echo=echoed.append), show_belief=k)
        run_turns(controller, LiveGame(FAR.initial_world(), FAR), 3)
        assert all(line.count("%") <= k for line in belief_lines(echoed)) and belief_lines(echoed)[0].count("%") == k


def test_a_refused_move_that_makes_the_engine_ask_again_does_not_print_twice():
    echoed: list[str] = []
    rules = MatchRules.from_respawns(flat(0, 10), flat(25, 45))
    live = LiveGame(rules.initial_world(), rules)
    asked_north = Scripted(lambda view: Direction.NORTH if not view.retry else E)   # north is off the board: refused
    controller = Controller(asked_north, Diagnostics(echo=echoed.append), show_belief=5)
    live.me.reset_for_new_turn()
    controller.take_turn(live.me)       # refused
    controller.take_turn(live.me)       # asked again in the same turn
    assert live.me.was_move_made_this_turn() and len(belief_lines(echoed)) == 1


def test_it_is_off_by_default_in_the_controller_and_on_in_the_gui_controller():
    echoed: list[str] = []
    quiet = Controller(Scripted(), Diagnostics(echo=echoed.append))
    run_turns(quiet, LiveGame(FAR.initial_world(), FAR), 3)
    assert belief_lines(echoed) == []

    shown: list[str] = []
    gui = CandidateController(diagnostics=Diagnostics(echo=shown.append))
    run_turns(gui, LiveGame(FAR.initial_world(), FAR), 3)
    assert len(belief_lines(shown)) == 3 and all(line.count("%") <= 5 for line in belief_lines(shown))

    off: list[str] = []
    silent = CandidateController(diagnostics=Diagnostics(echo=off.append), show_belief=0)
    run_turns(silent, LiveGame(FAR.initial_world(), FAR), 3)
    assert belief_lines(off) == []


def test_without_an_echo_nothing_is_printed_and_nothing_breaks():
    controller = Controller(ProbePolicy(seed=1), Diagnostics(), show_belief=5)
    run_turns(controller, LiveGame(FAR.initial_world(), FAR), 4)
    assert controller.diag.counts["turns"] == 4 and controller.diag.errors() == []


def test_the_line_comes_before_the_end_of_match_report():
    echoed: list[str] = []
    controller = Controller(ProbePolicy(seed=1), Diagnostics(echo=echoed.append), show_belief=5)
    live = LiveGame(FAR.initial_world(), FAR)
    from territorygame.domain.player_id import PlayerId

    for _ in range(10_000 - 3):
        live.state.decrement_remaining_turns(PlayerId(0))          # three turns left
    run_turns(controller, live, 3)
    last_line = max(i for i, text in enumerate(echoed) if text.startswith("[belief] turn"))
    report = next(i for i, text in enumerate(echoed) if "runtime report" in text)
    assert last_line < report
