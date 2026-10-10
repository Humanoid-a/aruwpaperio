"""Step 6 of the Q build, in pieces: QStrategy as the Strategy the controller runs, with the stand-in for Q."""

import numpy as np
import pytest

from candidate.agent.controller import Controller
from candidate.agent.data.types import Direction, MoveResult
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat, step
from candidate.agent.q import movers
from candidate.agent.q.config import QConfig
from candidate.agent.q.heuristic import HeuristicQ
from candidate.agent.q.strategy import QStrategy
from candidate.agent.q.types import State
from candidate.agent.runtime import Event
from candidate.tests.live import LiveGame
from candidate.tests.q_support import CLOSE, FAR, belief_of, scene, world_with

N, E, S, W = Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST


class FixedQ:
    """A Q that always returns the same five numbers (order: EXPAND, RETURN, HUNT, TRACK, PATROL)."""

    def __init__(self, *values):
        self.fixed = np.array(values, dtype=float)

    def values(self, frame, inputs=None):
        return self.fixed


def far_enemy(at=(25, 45)):
    return belief_of([dict(pos=flat(*at), weight=1.0)])


# ---- the hard rules, seen through the strategy --------------------------------------------------------------


def test_after_a_death_the_state_is_patrol_whatever_q_says():
    for event in (Event.SELF_DIED, Event.KILLED_BY_OPPONENT):
        strategy = QStrategy(FixedQ(9, 9, 9, 9, 0))
        view = scene(world_with(CLOSE), CLOSE, far_enemy(), events=(event,))
        move = strategy.choose(view)
        assert strategy.state == State.PATROL and strategy.last.forced == "death"
        assert bitboard.has_cell(strategy.last.snapshot.my_land_bb, step(view.obs.my_pos, move))     # and it stays home


def test_a_thin_margin_forces_return_and_the_move_is_returns():
    strategy = QStrategy(FixedQ(9, 0, 0, 0, 0))
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21), (25, 22)])
    view = scene(world, CLOSE, belief_of([dict(pos=flat(25, 27), weight=1.0)]))       # 5 from my head: margin 5 - 4 = 1
    move = strategy.choose(view)
    assert strategy.state == State.RETURN and strategy.last.forced == "safety"
    snap = strategy.last.snapshot
    assert move in view.safe_moves and snap.home_dist[step(view.obs.my_pos, move)] == snap.d4 - 1


def test_the_look_ahead_refuses_a_step_out_that_would_leave_a_negative_margin(monkeypatch):
    monkeypatch.setattr(movers, "move_for", lambda state, frame: E if state == State.EXPAND else movers._pace(frame))
    strategy = QStrategy(FixedQ(9, 0, 0, 0, 0))
    # I stand on the east edge of my land; the enemy is two cells beyond: stepping out lets it reach my trail first
    view = scene(world_with(CLOSE, me=(25, 19)), CLOSE, belief_of([dict(pos=flat(25, 21), weight=1.0)]))
    move = strategy.choose(view)
    assert strategy.last.forced == "lookahead" and strategy.state == State.PATROL
    assert bitboard.has_cell(strategy.last.snapshot.my_land_bb, step(view.obs.my_pos, move))


def test_the_look_ahead_sends_an_exposed_agent_home_instead_of_further_out(monkeypatch):
    monkeypatch.setattr(movers, "move_for", lambda state, frame: E if state == State.EXPAND else movers.return_(frame))
    strategy = QStrategy(FixedQ(9, 0, 0, 0, 0))
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21)])
    view = scene(world, CLOSE, belief_of([dict(pos=flat(25, 26), weight=1.0)]))      # margin 3 now, but east leaves it at -1
    move = strategy.choose(view)
    assert strategy.last.forced == "lookahead" and strategy.state == State.RETURN
    assert move != E and move in view.safe_moves


def test_the_look_ahead_leaves_a_safe_step_alone(monkeypatch):
    monkeypatch.setattr(movers, "move_for", lambda state, frame: E)
    strategy = QStrategy(FixedQ(9, 0, 0, 0, 0))
    view = scene(world_with(CLOSE, my_trail=[(25, 20), (25, 21)]), CLOSE, far_enemy())
    assert strategy.choose(view) == E and strategy.last.forced is None and strategy.state == State.EXPAND


def test_the_look_ahead_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(movers, "move_for", lambda state, frame: E)
    strategy = QStrategy(FixedQ(9, 0, 0, 0, 0), QConfig(safety_lookahead=False))
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21)])
    assert strategy.choose(scene(world, CLOSE, belief_of([dict(pos=flat(25, 26), weight=1.0)]))) == E


# ---- the switching cost, seen through the strategy ----------------------------------------------------------------------


def test_the_strategy_keeps_its_state_while_the_values_are_within_h():
    strategy = QStrategy(FixedQ(1.0, 0.0, 0.0, 0.5, 2.0))
    view = scene(world_with(CLOSE), CLOSE, far_enemy())
    strategy.choose(view)
    assert strategy.state == State.PATROL                         # started in PATROL; nothing beats it by more than 2
    strategy.q_model = FixedQ(5.0, 0.0, 0.0, 0.5, 2.0)
    strategy.choose(scene(world_with(CLOSE), CLOSE, far_enemy(), turn=41))
    assert strategy.state == State.EXPAND                         # now it does


# ---- the move: retries, the guard, the hints -----------------------------------------------------------------------------


def test_after_a_refused_move_the_state_is_kept_and_only_the_move_changes():
    strategy = QStrategy(FixedQ(0, 0, 0, 0, 5))
    view = scene(world_with(CLOSE, me=(25, 18)), CLOSE, far_enemy())
    first = strategy.choose(view)
    state = strategy.state
    view.retry = True
    view.blocked = frozenset({step(view.obs.my_pos, first)})
    for name in ("legal_moves", "safe_moves"):
        view.__dict__.pop(name, None)
    second = strategy.choose(view)
    assert strategy.state == state and second != first and second in view.safe_moves
    assert strategy.stats["retries"] == 1 and strategy.stats["state_PATROL"] == 1       # not counted as a new frame


def test_a_state_with_no_move_falls_back_on_a_safe_one_and_says_so(monkeypatch):
    monkeypatch.setattr(movers, "move_for", lambda state, frame: None)
    strategy = QStrategy(FixedQ(0, 0, 0, 0, 5))
    view = scene(world_with(CLOSE, my_trail=[(25, 20), (25, 21)]), CLOSE, far_enemy())
    move = strategy.choose(view)
    assert move in view.safe_moves and strategy.stats["guarded"] >= 1


def test_a_move_onto_my_own_trail_is_never_returned(monkeypatch):
    monkeypatch.setattr(movers, "move_for", lambda state, frame: W)             # straight back onto my trail
    strategy = QStrategy(FixedQ(0, 0, 0, 0, 5))
    view = scene(world_with(CLOSE, my_trail=[(25, 20), (25, 21)]), CLOSE, far_enemy())
    assert strategy.choose(view) in view.safe_moves


def test_leaving_my_land_remembers_the_direction_and_a_capture_forgets_it():
    strategy = QStrategy(FixedQ(9, 0, 0, 0, 0))
    view = scene(world_with(CLOSE, me=(25, 19)), CLOSE, far_enemy())
    strategy.choose(view)
    strategy.on_result(view, E, MoveResult.NORMAL)
    assert strategy.hints.outward == E
    strategy.on_result(view, N, MoveResult.INVALID)
    assert strategy.hints.outward == E                                           # a refusal changes nothing
    strategy.on_result(view, W, MoveResult.CAPTURED)
    assert strategy.hints.outward is None
    strategy.hints.outward = S
    strategy.on_result(view, W, MoveResult.DIED)
    assert strategy.hints.outward is None


def test_the_debug_label_names_the_state_and_a_forced_rule():
    strategy = QStrategy(FixedQ(0, 0, 0, 0, 5))
    strategy.choose(scene(world_with(CLOSE), CLOSE, far_enemy()))
    assert strategy.debug_state() == "PATROL"
    strategy.choose(scene(world_with(CLOSE), CLOSE, far_enemy(), events=(Event.SELF_DIED,), turn=41))
    assert strategy.debug_state() == "PATROL !death"


def test_reset_starts_a_match_afresh():
    strategy = QStrategy(FixedQ(9, 0, 0, 0, 0))
    strategy.choose(scene(world_with(CLOSE), CLOSE, far_enemy()))
    strategy.hints.outward = E
    strategy.reset()
    assert strategy.state == State.PATROL and strategy.hints.outward is None and not strategy.stats and strategy.last is None


# ---- the inputs are built every frame, as logged ---------------------------------------------------------------------------


def test_every_frame_records_the_three_inputs_the_values_and_the_masks():
    strategy = QStrategy()
    strategy.choose(scene(world_with(CLOSE, my_trail=[(25, 20), (25, 21)]), CLOSE, far_enemy()))
    record = strategy.last
    assert record.inputs.grid.shape == (7, 21, 21) and record.inputs.scalars.shape == (22,)
    assert record.inputs.confidence.shape == (8, 10, 10)
    assert record.values.shape == (5,) and record.valid.shape == (5,) and record.valid.dtype == bool
    assert record.inputs.scalars[17 + int(record.previous)] == 1.0                    # the state it was in when it looked


def test_the_inputs_can_be_left_out_for_speed():
    strategy = QStrategy(build_inputs=False)
    strategy.choose(scene(world_with(CLOSE), CLOSE, far_enemy()))
    assert strategy.last.inputs is None and strategy.timing_ms["inputs"][0] < 1.0


# ---- through the real controller and the real runtime -----------------------------------------------------------------------


def pace(live):
    live.opponent_moves(E if live.world().opp_pos == flat(25, 45) else W)


def test_a_whole_run_through_the_controller_makes_only_legal_moves_and_never_raises():
    strategy = QStrategy()
    controller = Controller(strategy, Diagnostics())
    live = LiveGame(FAR.initial_world(), FAR)
    for _ in range(150):
        live.me.reset_for_new_turn()
        controller.take_turn(live.me)
        assert live.me.was_move_made_this_turn()
        pace(live)
    diag = controller.diag
    assert diag.counts["exceptions"] == 0 and diag.errors() == []
    assert diag.counts["move_INVALID"] == 0 and strategy.stats["guarded"] == 0
    assert sum(v for k, v in strategy.stats.items() if k.startswith("state_")) == 150


def test_a_masked_state_is_never_the_state_taken_unless_a_hard_rule_forced_it():
    seen = []

    class Watch(QStrategy):
        def choose(self, view):
            move = super().choose(view)
            if not view.retry:
                r = self.last
                seen.append((r.valid[r.state], r.forced))
            return move

    controller = Controller(Watch(), Diagnostics())
    live = LiveGame(FAR.initial_world(), FAR)
    for _ in range(120):
        live.me.reset_for_new_turn()
        controller.take_turn(live.me)
        pace(live)
    assert len(seen) == 120 and all(valid or forced for valid, forced in seen)


def test_the_stand_in_q_is_in_cells_of_score_and_orders_the_states_sensibly():
    q = HeuristicQ()
    strategy = QStrategy(q)
    out = strategy.choose(scene(world_with(CLOSE), CLOSE, far_enemy()))
    values = strategy.last.values
    assert values[State.EXPAND] > values[State.PATROL]                          # at home with nothing in the way, go out
    long_trail = [(25, 20 + k) for k in range(12)]
    strategy.choose(scene(world_with(CLOSE, my_trail=long_trail), CLOSE, far_enemy(), turn=41))
    assert strategy.last.values[State.RETURN] > strategy.last.values[State.EXPAND]     # a long trail is worth banking
