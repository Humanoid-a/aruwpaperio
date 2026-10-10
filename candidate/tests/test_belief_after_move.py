"""Step 5 of a frame: my move is applied to every hypothesis at the end of the frame, right after I made it."""

import numpy as np
import pytest

from candidate.agent import adapter
from candidate.agent.belief.after_move import MyMove
from candidate.agent.belief.filter import BeliefFilter
from candidate.agent.data.belief_view import BeliefEvent
from candidate.agent.data.types import Direction, MoveResult
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.agent.runtime import AgentRuntime
from candidate.tests.belief_rig import FAR, NEAR, Rig, plain_move, prediction, prime
from candidate.tests.builders import make_observation
from candidate.tests.live import LiveGame
from candidate.tests.test_belief_filter import opponent_land

E, W, N = Direction.EAST, Direction.WEST, Direction.NORTH


def cells(*yx):
    return [flat(y, x) for y, x in yx]


def capture_move():
    """I closed a loop: my trail cell (25, 14) and the three cells (24..26, 15) became mine (gain 4)."""
    region = bitboard.from_cells(cells((24, 15), (25, 15), (26, 15)))
    return MyMove(
        MoveResult.CAPTURED,
        prediction(MoveResult.CAPTURED, region=region, gain=4, dest=flat(25, 15), new_pos=flat(25, 15)),
        tuple(cells((25, 14))), 9,
    )


def rig_with_one_opponent_hypothesis(count=9):
    rig = Rig(NEAR)
    prime(rig, [(flat(25, 17), [], 1.0, count)])        # standing on its own land, which memory also believes is its land
    return rig


def its_land(rig):
    return rig.state.opp_land(rig.state.store.opp_land_id)[0]


def my_land(rig):
    return rig.state.my_land(rig.state.store.my_land_id)[0]


# ---- what it does to the stored states ------------------------------------------------


def test_my_capture_takes_the_land_it_covers_from_every_stored_state_and_from_its_count():
    rig = rig_with_one_opponent_hypothesis()
    rig.move(capture_move())
    assert int(rig.state.store.opp_count[0]) == 9 - 3
    for cell in cells((24, 15), (25, 15), (26, 15)):
        assert not bitboard.has_cell(its_land(rig), cell)
    assert bitboard.has_cell(its_land(rig), flat(25, 17))            # the rest of its land is untouched


def test_my_capture_gives_the_land_it_covers_to_my_land_under_every_hypothesis():
    rig = rig_with_one_opponent_hypothesis()
    rig.move(capture_move())
    assert bitboard.popcount(my_land(rig)) == 9 + 1 + 3
    for cell in cells((25, 14), (24, 15), (25, 15), (26, 15)):
        assert bitboard.has_cell(my_land(rig), cell)


def test_my_trail_cells_lying_on_its_land_are_taken_too():
    rig = rig_with_one_opponent_hypothesis()
    move = MyMove(
        MoveResult.CAPTURED,
        prediction(MoveResult.CAPTURED, region=bitboard.empty(), gain=2, dest=flat(25, 13), new_pos=flat(25, 13)),
        tuple(cells((25, 15), (25, 14))), 9,    # my trail crossed (25, 15), which is its land
    )
    rig.move(move)
    assert int(rig.state.store.opp_count[0]) == 8


def test_my_death_gives_my_start_land_back_even_if_the_opponent_held_it():
    rig = Rig(NEAR)
    prime(rig, [(flat(25, 17), [], 1.0, 10)])
    rig.state.opp_base = bitboard.from_cells(list(NEAR.opp.start_cells) + cells((25, 11)))   # it holds (25, 11), in my start
    died = MyMove(MoveResult.DIED, prediction(MoveResult.DIED, dest=flat(25, 12), new_pos=flat(25, 10)), tuple(cells((25, 12))), 9)
    rig.move(died)
    assert int(rig.state.store.opp_count[0]) == 9 and not bitboard.has_cell(its_land(rig), flat(25, 11))
    assert np.array_equal(my_land(rig), NEAR.me.start_bb)


def test_my_death_wipes_the_land_i_had_down_to_my_start_land():
    rig = rig_with_one_opponent_hypothesis()
    rig.move(capture_move())
    died = MyMove(MoveResult.DIED, prediction(MoveResult.DIED, dest=flat(25, 12), new_pos=flat(25, 10)), tuple(cells((25, 12))), 13)
    rig.move(died)
    assert np.array_equal(my_land(rig), NEAR.me.start_bb) and rig.state.me.count == 9


def test_cutting_its_trail_collapses_every_stored_state_to_the_one_certain_state():
    rig = Rig(NEAR)
    prime(rig, [(flat(25, 20), [(25, 20)], 0.5, 9), (flat(26, 20), [(26, 20)], 0.5, 9)])
    kill = MyMove(
        MoveResult.NORMAL, prediction(MoveResult.NORMAL, killed=True, dest=flat(25, 12), new_pos=flat(25, 12), new_trail=cells((25, 12))), (), 9
    )
    rig.move(kill)
    store = rig.state.store
    assert store.n == 1 and int(store.pos[0]) == NEAR.opp.respawn and store.weight[0] == 1.0
    assert int(store.opp_count[0]) == 9 and int(store.trail_len[0]) == 0 and rig.state.residual.total() == 0
    assert rig.belief.stats.resets == 1
    view = rig.step(make_observation((25, 10), turn=4, respawn=(25, 10), opp_count=9, territory=opponent_land(NEAR, (25, 10))))
    assert BeliefEvent.OPPONENT_RESET in view.events


def test_cutting_its_trail_gives_it_its_start_land_back_out_of_mine():
    rig = Rig(NEAR)
    prime(rig, [(flat(25, 20), [(25, 20)], 1.0, 9)])
    from candidate.agent.belief.land import rewrite_my_land

    rewrite_my_land(rig.state, rig.state.my_base, add=bitboard.from_cells(cells((25, 15), (25, 8))))   # (25, 15) lies in its start
    kill = MyMove(
        MoveResult.NORMAL, prediction(MoveResult.NORMAL, killed=True, dest=flat(25, 12), new_pos=flat(25, 12), new_trail=cells((25, 12))), (), 11
    )
    rig.move(kill)
    assert not bitboard.has_cell(my_land(rig), flat(25, 15)) and bitboard.has_cell(my_land(rig), flat(25, 8))
    assert rig.state.me.count == 10                       # the engine's 11, less the one cell its respawn took back
    assert rig.state.store.n == 1


# ---- it leaves me where my move put me ----------------------------------------------------------------


def test_afterwards_the_belief_has_me_where_my_move_put_me_with_the_engines_count():
    rig = rig_with_one_opponent_hypothesis()
    rig.move(plain_move([(25, 11), (25, 12)], count=9))
    me = rig.state.me
    assert me.pos == flat(25, 12) and me.count == 9 and me.result == MoveResult.NORMAL
    assert sorted(bitboard.to_cells(me.trail_bb)) == cells((25, 11), (25, 12))
    rig.move(capture_move())
    assert rig.state.me.pos == flat(25, 15) and rig.state.me.count == 9 + 4 and rig.state.me.result == MoveResult.CAPTURED
    assert rig.state.me.trail_bb.sum() == 0


def test_a_capture_whose_size_is_unknown_leaves_my_count_unknown():
    rig = rig_with_one_opponent_hypothesis()
    unknown = MyMove(MoveResult.CAPTURED, prediction(MoveResult.CAPTURED, region=bitboard.empty(), gain=None), tuple(cells((25, 14))), 9)
    rig.move(unknown)
    assert rig.state.me.count is None


def test_an_observation_puts_me_back_as_the_engine_shows_me():
    rig = rig_with_one_opponent_hypothesis()
    rig.move(plain_move([(25, 11), (25, 12)]))
    rig.step(make_observation((25, 10), turn=5, respawn=(25, 10), opp_count=9, territory=opponent_land(NEAR, (25, 10))))
    assert rig.state.me.pos == flat(25, 10) and rig.state.me.result is None and rig.state.me.trail_bb.sum() == 0


# ---- cheap when nothing happened ------------------------------------------------------------------------------


def test_an_ordinary_step_changes_no_hypothesis_and_copies_nothing():
    rig = rig_with_one_opponent_hypothesis()
    state = rig.state
    store, opp_diffs, my_diffs = state.store, state.opp_diffs, state.my_diffs
    arrays = (store.opp_count, store.opp_land_id, store.my_land_id, store.weight)
    rig.move(plain_move([(25, 11)]))
    assert state.store is store and state.opp_diffs is opp_diffs and state.my_diffs is my_diffs
    assert all(a is b for a, b in zip(arrays, (store.opp_count, store.opp_land_id, store.my_land_id, store.weight)))


def test_only_the_territory_differences_in_use_are_rewritten():
    rig = rig_with_one_opponent_hypothesis()
    state = rig.state
    for i in range(50):                                          # differences left over from pruned hypotheses
        state.opp_diffs.intern(bitboard.from_cells([flat(40, i), flat(41, i)]), bitboard.empty())
    assert len(state.opp_diffs) == 51
    rig.move(capture_move())
    assert len(state.opp_diffs) <= 2                             # the empty one and the one in use


def test_rebasing_after_the_window_update_also_drops_unused_differences():
    rig = Rig(FAR)
    for turn in range(10):
        rig.step(make_observation((25, 4), turn=turn, respawn=(25, 4)))
        used = len(np.unique(rig.state.store.opp_land_id))
        assert len(rig.state.opp_diffs) <= used + 1


# ---- the runtime applies it at the end of the frame -------------------------------------------------------------------


def run_with_call_log(monkeypatch):
    log: list[str] = []
    original_obs = adapter.observation_from_game
    original_observe, original_after = BeliefFilter.observe, BeliefFilter.after_my_move

    def observe_obs(*args, **kwargs):
        log.append("translate")
        return original_obs(*args, **kwargs)

    def observe(self, *args, **kwargs):
        log.append("observe")
        return original_observe(self, *args, **kwargs)

    def after(self, move):
        log.append("after_my_move")
        return original_after(self, move)

    monkeypatch.setattr(adapter, "observation_from_game", observe_obs)
    monkeypatch.setattr(BeliefFilter, "observe", observe)
    monkeypatch.setattr(BeliefFilter, "after_my_move", after)
    return log


def test_a_frame_observes_then_the_move_is_made_then_it_is_applied(monkeypatch):
    runtime = AgentRuntime(Diagnostics())
    live = LiveGame(FAR.initial_world(), FAR)
    log = run_with_call_log(monkeypatch)
    live.begin(runtime)
    assert log == ["translate", "observe"]                      # nothing is applied at the start of a frame
    runtime.move(live.me, E)
    assert log == ["translate", "observe", "after_my_move"]     # ... it is the last thing in it
    live.opponent_moves(E)
    live.begin(runtime)
    assert log[3:] == ["translate", "observe"]                  # the next frame again starts with the observation


def test_a_refused_move_is_not_applied_and_the_move_that_follows_is(monkeypatch):
    rules = MatchRules_from_corner()
    live = LiveGame(rules.initial_world(), rules)
    runtime = AgentRuntime(Diagnostics())
    log = run_with_call_log(monkeypatch)
    live.begin(runtime)
    assert runtime.move(live.me, N).name == "INVALID"           # north is off the board
    assert "after_my_move" not in log
    runtime.move(live.me, E)
    assert log.count("after_my_move") == 1


def MatchRules_from_corner():
    from candidate.agent.rules.match import MatchRules

    return MatchRules.from_respawns(flat(0, 10), flat(25, 45))


def test_the_belief_already_has_my_move_in_it_when_the_move_call_returns():
    runtime = AgentRuntime(Diagnostics())
    live = LiveGame(FAR.initial_world(), FAR)
    live.begin(runtime)
    runtime.move(live.me, E)
    me = runtime.belief.state.me
    assert me.pos == flat(25, 5) and me.result == MoveResult.NORMAL


def test_a_failure_while_applying_my_move_does_not_raise_and_makes_the_belief_rebuild(monkeypatch):
    runtime = AgentRuntime(Diagnostics())
    live = LiveGame(FAR.initial_world(), FAR)
    live.begin(runtime)

    def broken(self, move):
        raise RuntimeError("boom")

    monkeypatch.setattr(BeliefFilter, "after_my_move", broken)
    runtime.move(live.me, E)                                    # the move was made; it must not look like a failure
    assert any(a.kind == "BELIEF_ERROR" for a in runtime.diag.anomalies)
    live.opponent_moves(E)
    monkeypatch.undo()
    view = live.begin(runtime)
    assert BeliefEvent.RESEEDED in view.belief.events


def test_the_time_for_the_step_is_reported_separately():
    runtime = AgentRuntime(Diagnostics())
    live = LiveGame(FAR.initial_world(), FAR)
    for turn in range(5):
        live.begin(runtime)
        runtime.move(live.me, E if turn % 2 == 0 else W)
        live.opponent_moves(E if live.world().opp_pos == flat(25, 45) else W)
    timings = runtime.diag.timings["belief_move"]
    assert len(timings) == 5 and max(timings) < 5.0                       # milliseconds; ordinary steps are nearly free
    assert "belief_move" in runtime.diag.report()
