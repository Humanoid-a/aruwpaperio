"""The feature-based motion model K: one test group per labelled feature (F1 .. F7)."""

from dataclasses import replace

import numpy as np
import pytest

from candidate.agent.belief.motion import FEATURE_NAMES, FeatureMotionModel, MotionWeights, move_probabilities
from candidate.agent.geometry import bitboard
from candidate.tests.test_batch_describe import real_batch

N, E, S, W = range(4)


def cell(y, x):
    return y * 50 + x


def scene(pos=cell(10, 20), trail_len=0, last_dir=-1, land=(cell(10, 10),), my_trail=(), **over):
    """One hypothesis standing at `pos`, all four moves legal and plain, unless overridden."""
    y, x = divmod(pos, 50)
    dest = np.array([[cell(y - 1, x), cell(y, x + 1), cell(y + 1, x), cell(y, x - 1)]], dtype=np.int64)
    false = np.zeros((1, 4), dtype=bool)
    fields = dict(
        n=1, pos=np.array([pos], dtype=np.int32), trail_len=np.array([trail_len], dtype=np.int32),
        last_dir=np.array([last_dir], dtype=np.int8), dest=dest, legal=~false, self_trail=false.copy(),
        kills_me=false.copy(), enters_own=false.copy(), captures=false.copy(),
        capture_gain=np.zeros((1, 4), dtype=np.int64),
        opp_land_bb=bitboard.from_cells(land), my_trail_bb=bitboard.from_cells(my_trail),
    )
    fields.update(over)
    return replace(real_batch(), **fields)


def features(batch, **weights):
    return FeatureMotionModel(MotionWeights(**weights)).features(batch)


def flag(column):
    out = np.zeros((1, 4), dtype=bool)
    out[0, column] = True
    return out


# ---- the structure -----------------------------------------------------------


def test_there_is_one_feature_per_name_each_an_n_by_4_array():
    batch = scene()
    parts = features(batch)
    assert tuple(parts) == FEATURE_NAMES
    assert all(a.shape == (1, 4) and np.isfinite(a).all() for a in parts.values())


def test_the_score_is_the_weighted_sum_of_the_features():
    batch = scene(trail_len=5, last_dir=E, self_trail=flag(W), kills_me=flag(N))
    w = MotionWeights()
    parts = features(batch)
    expected = sum(getattr(w, name) * parts[name] for name in FEATURE_NAMES)
    assert np.allclose(FeatureMotionModel().score(batch), expected)


def test_a_zero_weight_switches_a_feature_off():
    batch = scene(trail_len=5, last_dir=E, self_trail=flag(W), kills_me=flag(N))
    off = MotionWeights(**{name: 0.0 for name in FEATURE_NAMES})
    assert (FeatureMotionModel(off).score(batch) == 0).all()


def test_scoring_is_deterministic_and_leaves_the_batch_alone():
    batch = scene(trail_len=5, last_dir=E)
    before = [np.copy(getattr(batch, name)) for name in ("pos", "dest", "opp_land_bb", "my_trail_bb")]
    first, second = FeatureMotionModel().score(batch), FeatureMotionModel().score(batch)
    assert np.array_equal(first, second)
    after = [getattr(batch, name) for name in ("pos", "dest", "opp_land_bb", "my_trail_bb")]
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_it_does_not_print_unlike_the_placeholder(capsys):
    FeatureMotionModel().score(scene())
    assert capsys.readouterr().out == ""


# ---- F1: steps on its own trail -------------------------------------------------


def test_f1_the_move_onto_its_own_trail_scores_far_below_the_others():
    scores = FeatureMotionModel().score(scene(trail_len=3, self_trail=flag(S)))[0]
    assert scores[S] < scores[[N, E, W]].min() - 5.0


def test_f1_is_large_but_never_makes_the_move_impossible():
    batch = scene(trail_len=3, self_trail=flag(S))
    p = move_probabilities(FeatureMotionModel().score(batch), batch.legal, beta=1.0, epsilon=0.02)
    assert 0 < p[0, S] < 0.01


# ---- F2: steps on my trail --------------------------------------------------------


def test_f2_a_move_that_kills_me_is_preferred():
    scores = FeatureMotionModel().score(scene(kills_me=flag(E)))[0]
    assert scores.argmax() == E


# ---- F3: captures ---------------------------------------------------------------------


def test_f3_a_capture_pulls_and_a_bigger_one_pulls_harder_but_sublinearly():
    def gain(cells):
        g = np.zeros((1, 4), dtype=np.int64)
        g[0, N] = cells
        return features(scene(capture_gain=g, captures=flag(N)))["capture"][0, N]

    small, big = gain(4), gain(400)
    assert 0 < small < big < 100 * small
    assert features(scene())["capture"].sum() == 0


# ---- F4: going home -----------------------------------------------------------------------


def test_f4_a_long_trail_pulls_towards_its_own_land():
    # land at (10, 10), the opponent at (10, 20): west is the only move that gets closer
    f = features(scene(trail_len=8))["home_pull"][0]
    assert f[W] > 0 and (f[[N, E, S]] < 0).all()


def test_f4_the_pull_grows_with_the_trail_and_is_absent_without_one():
    short, long_ = features(scene(trail_len=1))["home_pull"][0, W], features(scene(trail_len=30))["home_pull"][0, W]
    assert 0 < short < long_ < 1
    assert (features(scene(trail_len=0))["home_pull"] == 0).all()


def test_f4_without_any_known_land_there_is_no_pull():
    assert (features(scene(trail_len=8, land=()))["home_pull"] == 0).all()


# ---- F5: chasing my trail -----------------------------------------------------------------------


def test_f5_it_closes_in_on_my_trail_when_it_can_see_it():
    f = features(scene(my_trail=(cell(10, 17),)))["chase"][0]       # three cells west
    assert f[W] > 0 and (f[[N, E, S]] < 0).all()


def test_f5_a_trail_outside_its_view_does_not_attract_it():
    assert (features(scene(my_trail=(cell(10, 40),)))["chase"] == 0).all()


def test_f5_the_view_radius_is_a_weight_of_the_model():
    batch = scene(my_trail=(cell(10, 17),))
    assert (features(batch, view_radius=2)["chase"] == 0).all()


# ---- F6: momentum ------------------------------------------------------------------------------------


def test_f6_it_prefers_to_keep_going_the_way_it_was():
    scores = FeatureMotionModel(MotionWeights(explore=0.0)).score(scene(trail_len=2, last_dir=S, land=()))[0]
    assert scores.argmax() == S


def test_f6_an_unexposed_opponent_has_no_momentum():
    assert (features(scene(last_dir=-1))["momentum"] == 0).all()


# ---- F7: exploring ---------------------------------------------------------------------------------------


def test_f7_leaving_its_land_from_inside_is_encouraged_but_stepping_on_its_land_is_not():
    f = features(scene(trail_len=0, enters_own=flag(W)))["explore"][0]
    assert f[W] == 0 and (f[[N, E, S]] == 1).all()


def test_f7_applies_only_when_it_has_no_trail():
    assert (features(scene(trail_len=3))["explore"] == 0).all()


# ---- the weights ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{"home_pressure_half": 0}, {"view_radius": 0}])
def test_nonsensical_weights_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MotionWeights(**kwargs)


# ---- with the filter ---------------------------------------------------------------------------------------


def test_the_filter_uses_the_feature_model_by_default():
    from candidate.agent.belief.filter import BeliefFilter

    assert isinstance(BeliefFilter().motion, FeatureMotionModel)
