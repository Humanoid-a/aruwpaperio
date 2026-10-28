"""The belief wired into the run-time layer, on the framework's real rules."""

import pytest

from candidate.agent.belief.filter import BeliefFilter
from candidate.agent.belief.motion import ConstantMotionModel
from candidate.agent.data.belief_view import BeliefEvent
from candidate.agent.data.types import Direction, Side
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.geometry.grid import flat
from candidate.agent.rules.match import MatchRules
from candidate.agent.runtime import AgentRuntime, Event
from candidate.tests.builders import put_trail
from candidate.tests.live import LiveGame

N, E, S, W = Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST
ME, OPP = Side.ME, Side.OPP
FAR = MatchRules.from_respawns(flat(25, 4), flat(25, 45))
MID = MatchRules.from_respawns(flat(25, 20), flat(25, 29))      # the opponent's respawn is the mirror of mine


def new_runtime(belief=None):
    return AgentRuntime(Diagnostics(), belief)


def pace(live):
    live.opponent_moves(E if live.world().opp_pos == flat(25, 45) else W)


def sound(runtime):
    state = runtime.belief.state
    assert state.store.problems(state.opp_diffs, state.my_diffs) == []
    assert float(state.store.weight.sum()) + state.residual.total() == pytest.approx(1.0)


def test_every_turn_carries_a_normalised_belief_for_that_turn():
    runtime = new_runtime()
    live = LiveGame(FAR.initial_world(), FAR)
    for turn in range(6):
        view = live.begin(runtime)
        assert view.belief is not None and view.belief.turn == view.obs.turn == turn
        assert view.belief.position.sum() == pytest.approx(1.0, abs=1e-5)
        sound(runtime)
        runtime.move(live.me, E if turn % 2 == 0 else W)
        pace(live)


def walk_in_then_out(turn):
    """The opponent's move: west for five turns (to 24, three cells from where I pace), then back east."""
    return W if turn < 5 else E


def test_the_true_opponent_position_always_has_probability_while_it_walks_into_view():
    runtime = new_runtime()
    live = LiveGame(MID.initial_world(), MID)
    seen_at = None
    for turn in range(12):
        view = live.begin(runtime)
        truth = live.world().opp_pos
        assert view.belief.probability_at(truth) > 0, f"turn {turn}"
        if view.obs.opp_seen_at is not None and seen_at is None:
            seen_at = turn
            assert view.belief.sighted_at == truth and view.belief.probability_at(truth) == pytest.approx(1.0)
            assert BeliefEvent.SIGHTED in view.belief.events and BeliefEvent.RESEEDED not in view.belief.events
        runtime.move(live.me, E if turn % 2 == 0 else W)
        assert live.opponent_moves(walk_in_then_out(turn)).name != "INVALID"
    assert seen_at is not None, "the opponent should have walked into view"


def test_when_the_opponent_leaves_view_its_probability_spreads_instead_of_vanishing():
    runtime = new_runtime()
    live = LiveGame(MID.initial_world(), MID)
    entropy_when_seen, entropy_after = [], []
    for turn in range(18):
        view = live.begin(runtime)
        (entropy_when_seen if view.obs.opp_seen_at is not None else entropy_after).append(view.belief.entropy)
        assert view.belief.probability_at(live.world().opp_pos) > 0, f"turn {turn}"
        runtime.move(live.me, E if turn % 2 == 0 else W)
        assert live.opponent_moves(walk_in_then_out(turn)).name != "INVALID"
    assert entropy_when_seen and max(entropy_when_seen) < 1.0
    assert entropy_after[-1] > max(entropy_when_seen) + 1.0       # unseen again: a wide belief, not a stale point


def test_killing_the_opponent_collapses_the_belief_to_its_respawn():
    world = FAR.initial_world()
    put_trail(world, ME, [(25, 6), (25, 7)])
    put_trail(world, OPP, [(25, 8), (24, 8)])
    runtime = new_runtime()
    live = LiveGame(world, FAR)
    view = live.begin(runtime)
    assert view.predict(E).killed
    runtime.move(live.me, E)          # onto (25, 8): its trail
    pace(live)                        # it respawns, and steps east
    view = live.begin(runtime)
    assert Event.KILLED_OPPONENT in view.events
    assert BeliefEvent.OPPONENT_RESET in view.belief.events and runtime.belief.stats.resets == 1
    assert view.belief.n_hypotheses == 4                                  # its four possible first steps from home
    assert view.belief.probability_at(flat(25, 46)) == pytest.approx(0.25)
    assert (runtime.belief.state.store.opp_count == 9).all()
    sound(runtime)


def test_being_killed_by_the_opponent_leaves_it_on_my_trail_with_certainty():
    world = FAR.initial_world()
    put_trail(world, ME, [(25, 6), (25, 7)])
    put_trail(world, OPP, [(24, 6)])
    runtime = new_runtime()
    live = LiveGame(world, FAR)
    live.begin(runtime)
    runtime.move(live.me, N)          # I step to (24, 7) ...
    live.opponent_moves(S)            # ... and it steps onto my trail at (25, 6)
    view = live.begin(runtime)
    assert Event.KILLED_BY_OPPONENT in view.events
    assert view.belief.sighted_at == flat(25, 6) and view.belief.probability_at(flat(25, 6)) == pytest.approx(1.0)
    assert BeliefEvent.RESEEDED not in view.belief.events
    assert (runtime.belief.state.store.flags & 4).all()
    sound(runtime)


def test_a_new_match_starts_the_belief_over():
    runtime = new_runtime()
    live = LiveGame(FAR.initial_world(), FAR)
    for turn in range(5):
        live.begin(runtime)
        runtime.move(live.me, E if turn % 2 == 0 else W)
        pace(live)
    other = LiveGame(FAR.initial_world(), FAR)
    view = other.begin(runtime)
    assert runtime.belief.stats.updates == 1                  # statistics restart with the match
    assert view.belief.probability_at(FAR.opp.respawn) == pytest.approx(0.95)


def test_the_belief_is_reported_with_the_other_diagnostics():
    runtime = new_runtime()
    live = LiveGame(FAR.initial_world(), FAR)
    for turn in range(8):
        live.begin(runtime)
        runtime.move(live.me, E if turn % 2 == 0 else W)
        pace(live)
    diag = runtime.diag
    assert len(diag.samples["belief_hypotheses"]) == 8 and len(diag.timings["belief"]) == 8
    report = diag.report()
    assert "belief: hypotheses mean" in report and "belief " in report.split("latency")[1]


def test_a_custom_belief_filter_can_be_supplied_to_the_runtime():
    seen = []

    class Recorder(ConstantMotionModel):
        def score(self, batch):
            seen.append(batch.n)
            return super().score(batch)

    runtime = new_runtime(BeliefFilter(motion=Recorder()))
    live = LiveGame(FAR.initial_world(), FAR)
    for turn in range(4):
        live.begin(runtime)
        runtime.move(live.me, E if turn % 2 == 0 else W)
        pace(live)
    assert len(seen) == 4 and seen[0] == 1 and max(seen) > 1


def test_a_skipped_turn_makes_the_belief_start_again_from_what_it_can_see():
    runtime = new_runtime()
    live = LiveGame(FAR.initial_world(), FAR)
    for turn in range(3):
        live.begin(runtime)
        runtime.move(live.me, E)
        pace(live)
    live.state.decrement_remaining_turns(live.me._owner)       # two of my turns pass at once
    live.state.decrement_remaining_turns(live.me._owner)
    view = live.begin(runtime)
    assert any(a.kind == "SKIPPED_TURNS" for a in runtime.diag.anomalies)
    assert BeliefEvent.RESEEDED in view.belief.events and view.belief.position.sum() == pytest.approx(1.0, abs=1e-5)
