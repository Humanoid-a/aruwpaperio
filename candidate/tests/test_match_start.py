"""The start of a match: the world state (memory) and the belief both begin exactly right."""

import numpy as np
import pytest

from candidate.agent.data.belief_view import BeliefEvent
from candidate.agent.data.cells import FREE, MY_TERRITORY, OPP_TERRITORY, UNKNOWN
from candidate.agent.data.types import Direction
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.agent.rules.match import MatchRules
from candidate.agent.runtime import AgentRuntime, Event
from candidate.tests.live import LiveGame
from candidate.tests.test_reference_rules import RULES as MID_RULES, loop_world

E, W = Direction.EAST, Direction.WEST
FAR = MatchRules.from_respawns(flat(25, 4), flat(25, 45))
SECOND = MatchRules.from_respawns(flat(25, 45), flat(25, 4))


def first_frame(rules):
    runtime = AgentRuntime(Diagnostics())
    live = LiveGame(rules.initial_world(), rules)
    view = live.begin(runtime)
    return runtime, live, view


# ---- the world state (memory) ----------------------------------------------------------------------------


def test_memory_knows_both_start_lands_from_the_first_frame_even_the_one_out_of_view():
    runtime, live, view = first_frame(FAR)
    memory = runtime.memory
    assert not view.obs.covers(FAR.opp.respawn)                       # the opponent's land is nowhere near my window
    for cell in FAR.me.start_cells:
        assert memory.territory.flat[cell] == MY_TERRITORY and memory.last_seen.flat[cell] == 0
    for cell in FAR.opp.start_cells:
        assert memory.territory.flat[cell] == OPP_TERRITORY and memory.trail.flat[cell] == FREE
        assert memory.last_seen.flat[cell] == 0
    assert memory.problems() == []


def test_everything_else_is_still_unknown_to_memory_except_what_the_window_shows():
    runtime, live, view = first_frame(FAR)
    memory = runtime.memory
    known = memory.known_mask()
    rows, cols = view.obs.window_slices
    expected = np.zeros((50, 50), dtype=bool)
    expected[rows, cols] = True
    for cell in FAR.opp.start_cells:
        expected.flat[cell] = True
    assert np.array_equal(known, expected)
    assert (memory.territory[~known] == UNKNOWN).all()


def test_memory_matches_the_boards_initial_world_where_it_knows_anything():
    runtime, live, view = first_frame(FAR)
    world = FAR.initial_world()
    known = runtime.memory.known_mask()
    # world.trail has no trail anywhere at the start, as memory has none
    assert np.array_equal(runtime.memory.territory[known], world.territory[known])


def test_the_belief_and_memory_agree_about_the_opponents_start_land_from_the_first_frame():
    runtime, live, view = first_frame(FAR)
    state = runtime.belief.state
    assert np.array_equal(state.opp_base, bitboard.from_mask(runtime.memory.territory == OPP_TERRITORY))
    assert np.array_equal(state.my_base, bitboard.from_mask(runtime.memory.territory == MY_TERRITORY))
    assert (state.opp_land(state.store.opp_land_id) == FAR.opp.start_bb).all()


def test_a_second_match_on_the_same_runtime_starts_from_a_fresh_seeded_memory():
    runtime, live, _ = first_frame(FAR)
    for turn in range(4):
        runtime.move(live.me, E if turn % 2 == 0 else W)
        live.opponent_moves(W if live.world().opp_pos == flat(25, 45) else E)
        live.begin(runtime)
    other = LiveGame(FAR.initial_world(), FAR)
    view = other.begin(runtime)
    assert Event.MATCH_STARTED in view.events
    assert runtime.memory.known_mask().sum() == view.obs.window_territory.size + 9    # window plus its start land
    assert runtime.belief.state.store.n >= 1 and runtime.belief.stats.updates == 1


# ---- the belief ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rules", [FAR, SECOND])
def test_the_belief_starts_certain_whichever_seat_i_have(rules):
    runtime, live, view = first_frame(rules)
    belief = view.belief
    assert belief.sighted_at is None and BeliefEvent.RESEEDED not in belief.events
    assert belief.position.sum() == pytest.approx(1.0, abs=1e-5)
    assert belief.summary.opp_counts == (9,) and belief.summary.my_views == 1 and belief.summary.my_views_exact == pytest.approx(1.0)


def test_the_seat_decides_whether_the_opponent_has_moved_when_i_first_look():
    _, _, first_mover = first_frame(FAR)
    _, _, second_mover = first_frame(SECOND)
    assert first_mover.belief.probability_at(FAR.opp.respawn) == pytest.approx(0.95)       # it has not moved yet
    assert second_mover.belief.probability_at(SECOND.opp.respawn) == pytest.approx(0.05)   # it has moved once


# ---- switched on in the middle of a match -------------------------------------------------------------------------


def mid_match():
    runtime = AgentRuntime(Diagnostics())
    live = LiveGame(loop_world(), MID_RULES)
    view = live.begin(runtime)
    return runtime, live, view


def test_a_controller_started_mid_match_does_not_pretend_it_is_the_start():
    runtime, live, view = mid_match()
    assert view.obs.my_count != 9 or view.obs.opp_count != 9 or view.obs.my_trail.size > 0
    assert BeliefEvent.RESEEDED in view.belief.events                       # rebuilt from what it can see
    rows, cols = view.obs.window_slices
    outside = np.ones((50, 50), dtype=bool)
    outside[rows, cols] = False
    assert not runtime.memory.known_mask()[outside].any()                   # no start land was invented for cells never seen
    assert view.belief.position.sum() == pytest.approx(1.0, abs=1e-5)
