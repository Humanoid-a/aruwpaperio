"""Recording games for training, from inside the controller, and the brief's exploration."""

import dataclasses

import numpy as np
import pytest

from candidate.agent.controller import Controller
from candidate.agent.data.types import Direction, MoveResult
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.geometry.grid import flat
from candidate.agent.behavior.config import BEHAVIOR
from candidate.agent.behavior.states import State
from candidate.agent.q.config import Q
from candidate.agent.q.record import GRID_SHAPE, GameRecorder, load_record, pack_grid, unpack_grid
from candidate.agent.q.strategy import QStrategy
from candidate.agent.q.version import inputs_version
from candidate.agent.runtime import Event
from candidate.tests.live import LiveGame
from candidate.tests.q_support import CLOSE, FAR, belief_of, scene, world_with
from territorygame.domain.player_id import PlayerId

E, W = Direction.EAST, Direction.WEST


class FixedQ:
    def __init__(self, *values):
        self.fixed = np.array(values, dtype=float)
        self.name = "fixed"

    def values(self, frame, inputs=None):
        return self.fixed


def far_enemy():
    return belief_of([dict(pos=flat(25, 45), weight=1.0)])


# ---- the file format -------------------------------------------------------------------------------------------------


def test_the_grid_packs_into_386_bytes_and_back_exactly():
    rng = np.random.default_rng(0)
    grids = (rng.random((5,) + GRID_SHAPE) < 0.4).astype(np.uint8)
    packed = np.stack([pack_grid(g) for g in grids])
    assert packed.shape == (5, 386) and packed.dtype == np.uint8
    assert np.array_equal(unpack_grid(packed), grids) and np.array_equal(unpack_grid(packed[0]), grids[0])


def test_every_frame_is_recorded_with_its_inputs_values_states_and_accepted_move(tmp_path):
    recorder = GameRecorder(tmp_path, label="unit")
    strategy = QStrategy(FixedQ(9, 0, 0, 0, 0, 0), recorder=recorder)
    first = scene(world_with(CLOSE), CLOSE, far_enemy(), turn=10)
    move = strategy.choose(first)
    kept = strategy.last
    strategy.on_result(first, move, MoveResult.NORMAL)
    second = scene(world_with(CLOSE), CLOSE, far_enemy(), turn=11)
    strategy.choose(second)                                             # no result: the turn ended without a move
    strategy.close()
    meta, a = load_record(recorder.last_path)
    assert meta["frames"] == 2 and meta["complete"] is False and meta["label"] == "unit" and meta["q_model"] == "fixed"
    assert meta["inputs_version"] == inputs_version() and meta["my_respawn"] == CLOSE.me.respawn
    assert a["turn"].tolist() == [10, 11] and a["direction"].tolist() == [int(move), -1]
    assert a["previous"][0] == int(State.PATROL) and a["state"][0] == int(kept.state)
    assert np.array_equal(unpack_grid(a["grid_bits"][0]), kept.inputs.grid)
    assert np.array_equal(a["scalars"][0], kept.inputs.scalars)
    assert np.array_equal(a["probmap"][0], kept.inputs.probmap.astype(np.float16))
    assert np.allclose(a["values"][0], [9, 0, 0, 0, 0, 0]) and a["valid"].shape == (2, 6)
    assert a["d5"][0] == kept.snapshot.d5 and a["d7"][0] == kept.snapshot.d7
    assert a["my_count"][0] == first.obs.my_count and a["opp_count"][0] == first.obs.opp_count


def test_the_game_is_saved_complete_after_my_last_turn(tmp_path):
    recorder = GameRecorder(tmp_path)
    strategy = QStrategy(recorder=recorder)
    controller = Controller(strategy, Diagnostics())
    live = LiveGame(FAR.initial_world(), FAR)
    for _ in range(10_000 - 4):
        live.state.decrement_remaining_turns(PlayerId(0))              # four turns left
    for _ in range(4):
        live.me.reset_for_new_turn()
        controller.take_turn(live.me)
        live.opponent_moves(E if live.world().opp_pos == flat(25, 45) else W)
    assert recorder.last_path is not None and not recorder.active
    meta, a = load_record(recorder.last_path)
    assert meta["complete"] is True and meta["frames"] == 4 and (a["direction"] >= 0).all()


def test_a_reset_in_the_middle_of_a_game_saves_it_as_incomplete(tmp_path):
    recorder = GameRecorder(tmp_path)
    strategy = QStrategy(recorder=recorder)
    strategy.choose(scene(world_with(CLOSE), CLOSE, far_enemy()))
    strategy.reset()
    assert recorder.last_path is not None and load_record(recorder.last_path)[0]["complete"] is False


def test_a_failure_to_write_never_reaches_the_game(tmp_path):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x")
    recorder = GameRecorder(blocker)
    strategy = QStrategy(recorder=recorder)
    strategy.choose(scene(world_with(CLOSE), CLOSE, far_enemy()))
    strategy.close()
    assert recorder.last_path is None and recorder.last_error


def test_recording_needs_the_inputs(tmp_path):
    with pytest.raises(ValueError):
        QStrategy(recorder=GameRecorder(tmp_path), build_inputs=False)


# ---- exploration -------------------------------------------------------------------------------------------------------


def run_frames(strategy, frames, **kw):
    out = []
    for turn in range(frames):
        strategy.choose(scene(world_with(CLOSE), CLOSE, far_enemy(), turn=40 + turn, **kw))
        out.append((strategy.state, strategy.last_explored, strategy.last.forced))
    return out


def test_without_exploration_nothing_is_explored():
    assert not any(e for _, e, _ in run_frames(QStrategy(FixedQ(0, 0, 0, 0, 0, 9)), 30))


def test_exploration_picks_valid_states_and_holds_each_pick_for_at_most_the_hold(monkeypatch):
    strategy = QStrategy(FixedQ(0, 0, 0, 0, 0, 9), explore_rate=0.3, explore_hold=4, seed=1)
    picks = []                                           # frames held, for every new pick
    original = strategy._rng.randint

    def recording_randint(a, b):
        held = original(a, b)
        picks.append(held)
        return held

    monkeypatch.setattr(strategy._rng, "randint", recording_randint)
    explored = 0
    for turn in range(200):
        strategy.choose(scene(world_with(CLOSE), CLOSE, far_enemy(), turn=40 + turn))
        if strategy.last_explored:
            explored += 1
            assert strategy.last.valid[strategy.state]   # never an invalid state
    assert picks and all(1 <= h <= 4 for h in picks)     # each pick is held 1 to 4 frames
    # at home every explored state stays valid, so every pick is held in full except possibly the last one
    assert sum(picks[:-1]) < explored <= sum(picks)
    assert 0.2 < explored / 200 < 0.9                    # 30% of frames start a hold of 2.5 frames on average


def test_a_hard_rule_beats_exploration():
    strategy = QStrategy(FixedQ(0, 0, 0, 0, 0, 9), explore_rate=1.0, seed=2)
    run_frames(strategy, 3)
    strategy.choose(scene(world_with(CLOSE), CLOSE, far_enemy(), turn=60, events=(Event.SELF_DIED,)))
    assert strategy.state == State.PATROL and strategy.last.forced == "death" and not strategy.last_explored


def test_the_same_seed_explores_the_same_way():
    a = run_frames(QStrategy(FixedQ(0, 0, 0, 0, 0, 9), explore_rate=0.2, seed=7), 60)
    b = run_frames(QStrategy(FixedQ(0, 0, 0, 0, 0, 9), explore_rate=0.2, seed=7), 60)
    c = run_frames(QStrategy(FixedQ(0, 0, 0, 0, 0, 9), explore_rate=0.2, seed=8), 60)
    assert a == b and a != c


def test_a_reset_restarts_the_exploration_sequence():
    strategy = QStrategy(FixedQ(0, 0, 0, 0, 0, 9), explore_rate=0.2, seed=3)
    first = run_frames(strategy, 40)
    strategy.reset()
    assert run_frames(strategy, 40) == first


# ---- the inputs version ----------------------------------------------------------------------------------------------


def test_the_inputs_version_follows_what_decides_the_inputs_and_nothing_else():
    assert inputs_version() == inputs_version(Q, BEHAVIOR) and len(inputs_version()) == 16
    assert inputs_version(behavior=dataclasses.replace(BEHAVIOR, distance_cap=40)) != inputs_version()      # the snapshot's
    assert inputs_version(behavior=dataclasses.replace(BEHAVIOR, possible_min_probability=1e-3)) != inputs_version()
    assert inputs_version(dataclasses.replace(Q, scale_distance=40.0)) != inputs_version()                 # an input's
    assert inputs_version(dataclasses.replace(Q, h_expand_base=7.0)) == inputs_version()                   # play only
    assert inputs_version(behavior=dataclasses.replace(BEHAVIOR, switch_cost=3.0)) == inputs_version()


# ---- the GUI's controller records its games -----------------------------------------------------------------------------


def test_the_gui_controller_records_by_default_and_code_with_its_own_diagnostics_does_not(monkeypatch, tmp_path):
    import candidate.candidate_controller as cc

    monkeypatch.setattr(cc, "RECORD_DIR", tmp_path / "gui")
    gui = cc.CandidateController()
    assert gui._agent.strategy.recorder is not None and gui._agent.strategy.recorder.directory == tmp_path / "gui"
    quiet = cc.CandidateController(diagnostics=Diagnostics())
    assert quiet._agent.strategy.recorder is None
    asked = cc.CandidateController(diagnostics=Diagnostics(), record=True)
    assert asked._agent.strategy.recorder is not None
    assert cc.CandidateController(record=False)._agent.strategy.recorder is None


def test_a_weights_file_that_cannot_be_loaded_falls_back_to_the_stand_in(tmp_path):
    import candidate.candidate_controller as cc
    from candidate.agent.q.heuristic import HeuristicQ

    said = []
    q = cc.load_q(tmp_path / "missing.npz", said.append)
    assert isinstance(q, HeuristicQ) and said and "could not load" in said[0]
    assert isinstance(cc.load_q(None), HeuristicQ)
