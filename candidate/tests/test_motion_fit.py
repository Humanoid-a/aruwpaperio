"""Fitting the belief's motion model K (offline/motion_fit.py): the replay of a recorded game, the features, the fit,
and the belief replayed with a given K."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from candidate.agent.belief.generate import Situation, classify
from candidate.agent.belief.motion import FeatureMotionModel, MotionWeights
from candidate.agent.data.hypotheses import HypothesisStore
from candidate.agent.data.territory import TerritoryDiffs
from candidate.agent.geometry import bitboard
from candidate.offline import collect
from candidate.offline import motion_fit as mf


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """One short recorded game against the hunter at the close respawns (they meet, so kills and chases happen)."""
    out = tmp_path_factory.mktemp("k_games")
    spec = collect.GameSpec(index=0, opponent="hunter", seat=1, close=True, turns=120, seed=3, explore=0.0,
                            model="heuristic", out=str(out))
    summary = collect.play_one(spec)
    path = mf.find_games([out])
    assert len(path) == 1 and summary["record"]
    return path[0]


def test_the_replay_follows_the_engine_and_rejects_a_wrong_tape(recorded):
    sidecar = json.loads(recorded.read_text())
    assert sum(1 for _ in mf.replay_boards(sidecar)) == 120                      # one row per opponent move, ends checked
    broken = dict(sidecar, final=dict(counts=[0, 0]))
    with pytest.raises(ValueError):
        list(mf.replay_boards(broken))


def test_the_stored_features_are_the_beliefs_own(recorded):
    rows = mf.opponent_rows(str(recorded))["rows"]
    weights = MotionWeights()
    F = mf.features_from(rows, weights.home_pressure_half, weights.view_radius)
    model = FeatureMotionModel(weights)
    for i, (turn, w, rules, opp_last) in enumerate(mf.replay_boards(json.loads(recorded.read_text()))):
        n = int(bitboard.popcount(w.opp_trail[0]))
        store = HypothesisStore.build(pos=[int(w.opp_pos[0])], last_dir=[opp_last if n else -1], trail_len=[n],
                                      trail_bb=w.opp_trail[0][None], opp_land_id=[0], my_land_id=[0],
                                      opp_count=[int(bitboard.popcount(w.opp_territory[0]))], flags=[0], weight=[1.0],
                                      hash=[0])
        sit = Situation(rules, None, w.opp_territory[0], TerritoryDiffs(), int(w.my_pos[0]), w.my_trail[0],
                        w.my_territory[0], TerritoryDiffs(), None)
        f = model.features(classify(SimpleNamespace(store=store), sit).batch)
        want = np.stack([f[name][0] for name in mf.FEATURES], axis=-1)
        legal = rows["legal"][i]
        assert np.allclose(F[i][legal], want[legal]), turn
    assert rows["legal"][np.arange(len(rows["label"])), rows["label"]].all()     # every move made was legal


def test_the_fit_recovers_weights_that_generated_the_moves():
    rng = np.random.default_rng(0)
    n = 6000
    F = rng.normal(size=(n, 4, 7))
    legal = rng.random((n, 4)) > 0.1
    legal[:, 0] = True
    true_w = np.array([-3.0, 2.0, 0.5, 1.0, 0.8, 1.2, 0.4])
    p = mf.probabilities(F, legal, true_w, 1.0, 0.02)
    label = np.array([rng.choice(4, p=row / row.sum()) for row in p])
    w = mf.fit_weights(F, legal, label, np.ones(n), np.zeros(7), 1.0, 0.02, l2=0.0)
    assert np.abs(w - true_w).max() < 0.25
    assert mf.metrics(F, legal, label, w, 1.0, 0.02)["log_likelihood"] > mf.metrics(F, legal, label, np.zeros(7), 1.0, 0.02)["log_likelihood"]


def test_the_belief_replayed_with_a_given_k_never_rules_out_the_truth(recorded):
    current = dict(zip(mf.FEATURES, mf.current_weights()[0].tolist()), home_pressure_half=6.0, view_radius=5)
    result = mf.belief_game((str(recorded), current, 120))
    assert result["turns"] == 120 and result["errors"] == 0 and result["failed_turns"] == 0
    assert result["possible"] == 1.0 and 0.0 < result["p_true"] <= 1.0
