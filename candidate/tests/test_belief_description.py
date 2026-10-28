"""The per-frame description of the whole belief state (the "[belief state]" lines), and the summary behind it."""

import pytest

from candidate.agent.belief import publish as publish_module
from candidate.agent.controller import Controller
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.probe import ProbePolicy
from candidate.candidate_controller import CandidateController
from candidate.runtime_check import run_check
from candidate.tests.live import LiveGame
from candidate.tests.test_belief_display import FAR, run_turns


def state_lines(echoed):
    return [text for text in echoed if text.startswith("[belief state]")]


def played(turns=6, **kwargs):
    echoed: list[str] = []
    controller = Controller(ProbePolicy(seed=1), Diagnostics(echo=echoed.append), **kwargs)
    live = LiveGame(FAR.initial_world(), FAR)
    run_turns(controller, live, turns)
    return controller, echoed


def test_four_lines_describe_the_belief_on_every_turn():
    _, echoed = played(6, describe_belief=True)
    lines = state_lines(echoed)
    assert len(lines) == 6 * 4
    for turn in range(6):
        block = lines[4 * turn: 4 * turn + 4]
        assert all(f"turn {turn:4d}" in line for line in block)
        assert "hypotheses hold" in block[0] and "opponent: territory count" in block[1]
        assert "me: territory count" in block[2] and "this frame:" in block[3]


def test_the_description_has_no_cell_list_and_is_off_unless_asked_for():
    _, echoed = played(3, show_belief=5)
    assert state_lines(echoed) == []
    _, echoed = played(3, describe_belief=True)
    assert not any(text.startswith("[belief] turn") for text in echoed)          # the top-k line is a separate option
    assert not any("(45,25)" in line for line in state_lines(echoed))


def test_both_lines_can_be_shown_together_and_the_gui_controller_shows_both():
    _, echoed = played(2, show_belief=5, describe_belief=True)
    assert len(state_lines(echoed)) == 8 and sum(text.startswith("[belief] turn") for text in echoed) == 2
    shown: list[str] = []
    gui = CandidateController(diagnostics=Diagnostics(echo=shown.append))
    run_turns(gui, LiveGame(FAR.initial_world(), FAR), 2)
    assert len(state_lines(shown)) == 8
    off: list[str] = []
    run_turns(CandidateController(diagnostics=Diagnostics(echo=off.append), describe_belief=False), LiveGame(FAR.initial_world(), FAR), 2)
    assert state_lines(off) == []


def test_the_description_agrees_with_the_engines_counts():
    _, echoed = played(5, describe_belief=True)
    for line in state_lines(echoed):
        if "opponent: territory count" in line:
            assert "territory count 9 on every hypothesis (engine says 9)" in line
        if "me: territory count" in line:
            assert "territory count 9;" in line and "100.0% of the mass agrees" in line


def test_the_summary_is_attached_to_every_published_view():
    controller, _ = played(4, describe_belief=True)
    summary = controller.runtime.belief.view.summary
    assert summary is not None
    assert summary.opp_counts == (9,) and summary.my_count == 9
    assert summary.my_views == 1 and summary.my_views_exact == pytest.approx(1.0)
    assert summary.exposed_share is not None and 0.0 <= summary.exposed_share <= 1.0


def test_the_frame_numbers_are_for_this_frame_not_running_totals():
    controller, _ = played(8, describe_belief=True)
    belief = controller.runtime.belief
    assert belief.view.summary.successors < belief.stats.children                     # not cumulative
    seen: list[int] = []
    original = publish_module.summarise

    def spy(*args):
        summary = original(*args)
        seen.append(summary.successors)
        return summary

    publish_module.summarise = spy
    try:
        fresh = Controller(ProbePolicy(seed=1), Diagnostics(), describe_belief=True)
        run_turns(fresh, LiveGame(FAR.initial_world(), FAR), 8)
    finally:
        publish_module.summarise = original
    assert sum(seen) == fresh.runtime.belief.stats.children                           # the frames add up to the total


def test_a_belief_with_no_hypotheses_is_described_without_failing(monkeypatch):
    controller, echoed = played(3, describe_belief=True)
    belief = controller.runtime.belief
    from candidate.agent.data.hypotheses import HypothesisStore

    belief.state.store = HypothesisStore.empty()
    summary = publish_module.summarise(belief.state, controller.runtime._view.obs, belief.stats, belief._stats_mark)
    assert summary.opp_counts == () and summary.exposed_share is None and summary.my_views == 0


def test_it_works_in_a_whole_engine_match():
    result = run_check("hunter", turns=60, close=True)
    assert result.ok, result.format()
