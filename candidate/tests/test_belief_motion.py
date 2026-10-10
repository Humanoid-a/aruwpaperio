import numpy as np
import pytest

from candidate.agent.belief.config import BeliefConfig, BudgetPolicy, ProportionalBudget
from candidate.agent.belief.motion import ConstantMotionModel, MotionModel, SuccessorBatch, move_probabilities


def legal_all(n=3):
    return np.ones((n, 4), dtype=bool)


# ---- move_probabilities --------------------------------------------------


def test_equal_scores_give_equal_probabilities_over_the_legal_moves():
    legal = np.array([[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 1]], dtype=bool)
    p = move_probabilities(np.zeros((3, 4)), legal, beta=1.0, epsilon=0.02)
    assert np.allclose(p[0], 0.25) and np.allclose(p[1], [0.5, 0, 0.5, 0]) and np.allclose(p[2], [0, 0, 0, 1])


def test_illegal_moves_get_zero_and_each_row_sums_to_one():
    rng = np.random.default_rng(1)
    legal = rng.random((50, 4)) < 0.7
    legal[:, 0] = True
    p = move_probabilities(rng.normal(size=(50, 4)), legal, beta=2.0, epsilon=0.05)
    assert (p[~legal] == 0).all()
    assert np.allclose(p.sum(axis=1), 1.0)


def test_a_row_with_no_legal_move_is_all_zero_and_does_not_poison_the_others():
    legal = np.array([[0, 0, 0, 0], [1, 1, 0, 0]], dtype=bool)
    p = move_probabilities(np.array([[5.0, 1, 1, 1], [0.0, 0.0, 9, 9]]), legal, 1.0, 0.1)
    assert (p[0] == 0).all() and np.allclose(p[1], [0.5, 0.5, 0, 0])


def test_the_floor_keeps_every_legal_move_above_zero_however_sharp_the_model_is():
    scores = np.array([[100.0, 0.0, 0.0, 0.0]])
    p = move_probabilities(scores, legal_all(1), beta=50.0, epsilon=0.04)
    assert (p > 0).all() and p[0, 1] == pytest.approx(0.04 / 4)
    assert p[0, 0] == pytest.approx(0.96 + 0.01)


def test_higher_scores_get_higher_probability_and_beta_sharpens_the_preference():
    scores = np.array([[0.0, 1.0, 2.0, 3.0]])
    soft = move_probabilities(scores, legal_all(1), beta=0.5, epsilon=0.0)[0]
    sharp = move_probabilities(scores, legal_all(1), beta=3.0, epsilon=0.0)[0]
    assert (np.diff(soft) > 0).all() and (np.diff(sharp) > 0).all()
    assert sharp[3] > soft[3] and sharp[0] < soft[0]


def test_epsilon_one_is_uniform_over_the_legal_moves():
    p = move_probabilities(np.array([[9.0, 0, 0, 0]]), np.array([[1, 1, 1, 0]], dtype=bool), 5.0, 1.0)
    assert np.allclose(p, [[1 / 3, 1 / 3, 1 / 3, 0]])


def test_very_large_scores_do_not_overflow():
    p = move_probabilities(np.array([[1e6, -1e6, 0.0, 1e6]]), legal_all(1), beta=1.0, epsilon=0.01)
    assert np.isfinite(p).all() and p.sum() == pytest.approx(1.0)


# ---- the model interface ---------------------------------------------------


def test_the_abstract_model_cannot_be_used_directly():
    with pytest.raises(TypeError):
        MotionModel()


def test_the_constant_model_scores_every_move_the_same(capsys):
    from candidate.tests.test_batch_describe import real_batch

    batch = real_batch()
    scores = ConstantMotionModel().score(batch)
    assert scores.shape == (batch.n, 4) and (scores == 0).all()
    assert (ConstantMotionModel(2.5).score(batch) == 2.5).all()
    capsys.readouterr()      # the model prints a description of each batch it is given


def test_a_subclass_only_has_to_provide_score():
    class AlwaysNorth(MotionModel):
        def score(self, batch):
            out = np.zeros((batch.n, 4))
            out[:, 0] = 10.0
            return out

    batch = SuccessorBatch.__new__(SuccessorBatch)
    object.__setattr__(batch, "n", 2)
    p = move_probabilities(AlwaysNorth().score(batch), legal_all(2), 1.0, 0.02)
    assert (p.argmax(axis=1) == 0).all()


# ---- the budget -------------------------------------------------------------


def test_the_budget_is_proportional_to_a_positions_probability_within_its_limits():
    budget = ProportionalBudget(total=1000, min_keep=2, max_keep=300)
    mass = np.zeros((50, 50))
    mass[10, 10], mass[10, 11], mass[20, 20], mass[30, 30] = 0.5, 0.0004, 0.2, 0.0996
    keep = budget.per_cell(mass)
    assert keep[10, 10] == 300            # 500 wanted, capped at max_keep
    assert keep[20, 20] == 200            # exactly proportional
    assert keep[30, 30] == 100
    assert keep[10, 11] == 2              # a rare position still keeps the minimum
    assert keep[0, 0] == 0                # no probability, no hypotheses
    assert keep.dtype.kind == "i"


def test_a_position_that_certainly_holds_the_opponent_gets_the_largest_allowance():
    mass = np.zeros((50, 50))
    mass[7, 7] = 1.0
    assert ProportionalBudget(total=2000, max_keep=1500).per_cell(mass)[7, 7] == 1500
    assert ProportionalBudget(total=800, max_keep=1500).per_cell(mass)[7, 7] == 800


def test_the_budget_policy_is_pluggable():
    class Flat(BudgetPolicy):
        def per_cell(self, mass):
            return np.where(mass > 0, 3, 0)

    assert BeliefConfig(budget=Flat()).budget.per_cell(np.full((50, 50), 0.1)).max() == 3
    with pytest.raises(TypeError):
        BudgetPolicy()


@pytest.mark.parametrize(
    "kwargs",
    [{"min_keep": 0}, {"min_keep": 5, "max_keep": 2}, {"total": 0}],
)
def test_a_nonsensical_budget_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ProportionalBudget(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"epsilon": 1.0}, {"epsilon": -0.1}, {"capacity": 0}, {"first_update_stay": 1.0}, {"seat_confidence": 0.4}, {"seat_confidence": 1.0}, {"rehydrate_below": 2.0}],
)
def test_a_nonsensical_belief_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        BeliefConfig(**kwargs)


def test_the_default_config_keeps_the_stored_hypotheses_well_inside_the_capacity():
    config = BeliefConfig()
    assert isinstance(config.budget, ProportionalBudget)
    assert config.budget.total < config.capacity and config.epsilon > 0
