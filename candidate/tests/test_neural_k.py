"""The belief's neural motion model K (agent/belief/neural.py) and its training (offline/k_train.py)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from candidate.agent.belief import neural
from candidate.agent.belief.filter import BeliefFilter
from candidate.agent.belief.generate import Situation, classify
from candidate.agent.belief.motion import FeatureMotionModel
from candidate.agent.belief.neural import FEATURES, N_FEATURES, NeuralMotionModel, load_motion
from candidate.agent.controller import Controller
from candidate.agent.data.hypotheses import HypothesisStore
from candidate.agent.data.territory import TerritoryDiffs
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.agent.q.strategy import QStrategy
from candidate.offline import collect, match
from candidate.runtime_check import OPPONENTS, _Spy
from candidate.tests.q_support import CLOSE, cells

F = {name: i for i, name in enumerate(FEATURES)}


def batch_of(hypotheses, my_pos=flat(25, 22), my_trail=((25, 20), (25, 21), (25, 22)), my_land=CLOSE.me.start_bb):
    """A SuccessorBatch for hand-made hypotheses: dicts with pos, trail (cells), land (bitboard), weight."""
    opp_base = bitboard.empty()
    opp_diffs, my_diffs = TerritoryDiffs(), TerritoryDiffs()
    store = HypothesisStore.build(
        pos=[h["pos"] for h in hypotheses], last_dir=[h.get("last_dir", -1) for h in hypotheses],
        trail_len=[len(h.get("trail", ())) for h in hypotheses],
        trail_bb=np.stack([bitboard.from_cells(h.get("trail", ())) for h in hypotheses]),
        opp_land_id=[int(opp_diffs.intern_territory(h["land"], opp_base)[0]) for h in hypotheses],
        my_land_id=[0] * len(hypotheses), opp_count=[9] * len(hypotheses), flags=[0] * len(hypotheses),
        weight=[h.get("weight", 1.0) for h in hypotheses], hash=[0] * len(hypotheses))
    sit = Situation(CLOSE, None, opp_base, opp_diffs, my_pos, bitboard.from_cells(cells(*my_trail)), my_land, my_diffs, 9)
    return classify(SimpleNamespace(store=store), sit).batch


def test_the_features_are_each_hypothesis_own():
    west, east = bitboard.from_cells(cells((10, 10))), bitboard.from_cells(cells((10, 40)))
    b = batch_of([dict(pos=flat(10, 25), land=west), dict(pos=flat(10, 25), land=east)])
    f = neural.features(b)
    assert f.shape == (2, 4, N_FEATURES) and f.dtype == np.float32
    # same cell, different lands: its distance home after each move differs, and so does "nearer home"
    assert f[0, 3, F["enemy_home"]] == pytest.approx(14 / 50) and f[1, 1, F["enemy_home"]] == pytest.approx(14 / 50)
    assert f[0, 3, F["home_change"]] == 1 and f[1, 3, F["home_change"]] == -1
    assert f[0, 1, F["home_change"]] == -1 and f[1, 1, F["home_change"]] == 1


def test_the_kill_and_safety_factors_follow_their_definitions():
    land = bitboard.from_cells(cells((25, 30), (25, 31), (25, 32)))
    near = dict(pos=flat(25, 25), land=land, trail=cells((25, 29), (25, 28), (25, 27), (25, 26), (25, 25)))
    b = batch_of([near])
    f = neural.features(b)[0]
    west = 3                                                            # W from (25, 25) is (25, 24)
    assert f[west, F["sees_me"]] == 1                                    # I am at (25, 22), 2 columns away
    assert f[west, F["to_my_trail"]] == pytest.approx(2 / 50)            # (25, 24) to my trail cell (25, 22)
    assert f[west, F["to_its_trail"]] == pytest.approx(2 / 50)           # me to its new cell (25, 24)
    # it is 6 from home after the move, I am 2 from its trail: I could cut it first
    assert f[west, F["enemy_home"]] == pytest.approx(6 / 50) and f[west, F["i_kill_enemy"]] == pytest.approx(4 / 25)
    assert f[west, F["enemy_safety"]] == pytest.approx(-4 / 25)          # it sees me: from its side, unsafe
    assert f[west, F["my_home"]] == pytest.approx(4 / 50)                # (25, 22) home round my trail: (24, 22) .. (24, 19)
    assert f[west, F["enemy_kill_me"]] == pytest.approx((4 - 2) / 25)    # it reaches my trail before I am home
    assert f[0, F["return_gain"]] > 0                                     # going home pays: its 5 trail cells at least


def test_the_distance_to_each_trail_is_the_nearest_cells_taxicab_distance_capped():
    rng = np.random.default_rng(3)
    masks = rng.random((300, 50, 50)) < rng.choice([0.0, 0.0005, 0.003, 0.05], size=(300, 1, 1))
    masks[0, 0, 49] = True                                              # a cell in the last column
    sets = bitboard.from_mask(masks)
    ys, xs = np.divmod(np.arange(2500), 50)
    for origin in (0, flat(25, 49), flat(49, 0), flat(17, 31)):
        oy, ox = divmod(origin, 50)
        ring = (np.abs(ys - oy) + np.abs(xs - ox)).reshape(50, 50)
        expected = [min(int(ring[m].min()), neural.CAP) if m.any() else neural.CAP for m in masks]
        assert neural._distance_to_sets(origin, sets).tolist() == expected


def test_save_load_and_the_untrained_k_is_nearly_uniform(tmp_path):
    k = NeuralMotionModel.initial(3)
    path = k.save(tmp_path / "k.npz")
    again = NeuralMotionModel.load(path)
    b = batch_of([dict(pos=flat(10, 25), land=bitboard.from_cells(cells((10, 10))))])
    assert np.allclose(k.score(b), again.score(b)) and np.abs(k.score(b)).max() < 0.5
    assert isinstance(load_motion(None), FeatureMotionModel) and isinstance(load_motion(tmp_path / "none.npz"), FeatureMotionModel)


def test_with_a_neural_k_the_belief_never_rules_out_the_truth():
    collector = match.Watch(keep=True)
    spy = _Spy(QStrategy(), collector, player=0)
    controller = Controller(spy, Diagnostics(), belief=BeliefFilter(motion=NeuralMotionModel.initial(1)))
    match.play([match.AgentAdapter(controller), OPPONENTS["hunter"]()], 150, match.CLOSE, watch=collector)
    assert len(spy.belief_rows) == 150 and all(r["in_support"] for r in spy.belief_rows)
    assert not controller.diag.errors()


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    out = tmp_path_factory.mktemp("k_games")
    for i, opponent in enumerate(("noisy", "self", "enemy_copy")):
        collect.play_one(collect.GameSpec(index=i, opponent=opponent, seat=i % 2, close=True, turns=120, seed=5 + i,
                                          explore=0.0, model="heuristic", out=str(out), opponent_randomness=0.5))
    return out


def test_k_trains_on_the_opponents_real_moves_and_beats_a_uniform_guess(recorded, tmp_path):
    from candidate.offline import k_train

    data = k_train.build([recorded], workers=1, echo=None)
    assert data["features"].shape[1:] == (4, N_FEATURES) and len(data["label"]) == 3 * 120
    assert data["legal"][np.arange(len(data["label"])), data["label"]].all()
    rep = k_train.train_step([recorded], tmp_path / "k.npz", workers=1, epochs=20, echo=None)
    every = rep["validation"]["all"]
    assert every["trained"]["log_likelihood"] > every["uniform"]["log_likelihood"]
    assert NeuralMotionModel.load(tmp_path / "k.npz").meta["games"] == 3


def test_the_guis_controller_takes_a_k_from_the_variable(monkeypatch, tmp_path):
    import candidate.candidate_controller as cc

    path = NeuralMotionModel.initial(2).save(tmp_path / "k.npz")
    monkeypatch.setenv(cc.K_WEIGHTS_VARIABLE, str(path))
    c = cc.CandidateController(record=False, diagnostics=Diagnostics(echo=None))
    assert isinstance(c._agent.runtime.belief.motion, NeuralMotionModel)
    monkeypatch.delenv(cc.K_WEIGHTS_VARIABLE)
    plain = cc.CandidateController(record=False, diagnostics=Diagnostics(echo=None))
    assert isinstance(plain._agent.runtime.belief.motion, FeatureMotionModel)


def test_the_online_loop_trains_k_every_epoch_and_plays_the_next_with_it(tmp_path):
    from candidate.offline import online

    out = online.run("k", epochs=2, games=3, workers=1, turns=40, opponents="noisy=1,self=1,enemy_copy=1", gate_games=0,
                     play_k="trained",
                     enemy_randomness=1.0, self_randomness=1.0, explore=0.2, belief_share=1.0, close_share=0.5,
                     train_epochs=1, patience=1, root=tmp_path, quiet=True)
    run = tmp_path / "k"
    assert (online.epoch_dir(run, 0) / "k.npz").exists() and (online.epoch_dir(run, 1) / "k.npz").exists()
    assert out[0]["k"]["val_nll"] is not None and (run / "latest_k.npz").exists()
    epoch1 = json.loads((online.epoch_dir(run, 1) / "games.json").read_text())
    assert all(g["spec"]["k_model"] == str(online.epoch_dir(run, 0) / "k.npz") for g in epoch1)
