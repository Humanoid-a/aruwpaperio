"""The chooser (masks, switching cost, hard rules; step 5 of the Q build)."""

import itertools

import numpy as np
import pytest

from candidate.agent.behavior.chooser import Choice, choose, switching_cost
from candidate.agent.behavior.config import BehaviorConfig
from candidate.agent.behavior.states import N_STATES, State

E, E2, R, H, T, P = State.EXPAND, State.EXPAND2, State.RETURN, State.HUNT, State.TRACK, State.PATROL
ALL = np.ones(N_STATES, dtype=bool)


def pick(values, current, valid=ALL, died=False, trail=0, margin=50.0, cfg=None):
    kw = dict(died=died, trail_len=trail, margin=margin)
    return choose(np.array(values, dtype=float), valid, current, **kw, **({} if cfg is None else {"cfg": cfg}))


def test_a_death_sends_the_controller_to_patrol_whatever_q_says():
    choice = pick([9, -9, 9, 9, 9, 0], E, died=True, trail=0)
    assert choice == Choice(P, "death")


def test_death_wins_over_the_safety_rule():
    assert pick([0, -9, 9, 0, 0, 0], E, died=True, trail=3, margin=0).state == P


def test_a_margin_at_or_below_the_threshold_forces_return():
    assert pick([9, -9, 0, 9, 9, 9], E, trail=4, margin=1) == Choice(R, "safety")
    assert pick([9, -9, 0, 9, 9, 9], E, trail=4, margin=-3).forced == "safety"
    assert pick([9, -9, 0, 0, 0, 0], E, trail=4, margin=2).forced is None             # one more turn of margin: Q decides


def test_the_safety_rule_has_nothing_to_protect_without_a_trail():
    choice = pick([9, -9, 0, 0, 0, 0], E, trail=0, margin=-5)
    assert choice.state == E and choice.forced is None


def test_the_safety_rule_overrides_the_mask_and_the_switching_cost():
    valid = np.array([1, 0, 0, 1, 1, 1], dtype=bool)                                  # RETURN masked, margin low: still forced
    assert pick([9, -9, 0, 9, 9, 9], E, valid=valid, trail=4, margin=0).state == R


def test_a_masked_state_is_never_chosen():
    for mask in itertools.product([0, 1], repeat=N_STATES):
        valid = np.array(mask, dtype=bool)
        if not valid.any():
            continue
        for current in State:
            choice = pick([5, -9, 9, 3, 7, 1], current, valid=valid)
            if not valid[current] or choice.state != current:
                assert valid[choice.state]


def test_when_the_values_are_within_h_of_each_other_the_state_never_changes():
    rng = np.random.default_rng(0)
    for _ in range(200):
        base = float(rng.normal() * 10)
        values = base + rng.uniform(0, 2.0, N_STATES)                                     # spread < h = 2
        for current in State:
            assert pick(values, current).state == current


def test_a_state_must_beat_the_current_one_by_more_than_h():
    assert pick([0, -9, 0, 0, 0, 2.0], E).state == E                                   # exactly h: stays
    assert pick([0, -9, 0, 0, 0, 2.01], E).state == P


def test_switching_from_return_to_expand_or_the_turn_costs_more():
    assert switching_cost(R, E) == 4.0 and switching_cost(R, E2) == 4.0
    assert switching_cost(E, R) == 2.0 and switching_cost(P, E) == 2.0 and switching_cost(E, E2) == 2.0
    assert pick([3.5, -9, 0, 0, 0, 0], R, valid=np.array([1, 0, 1, 0, 0, 0], dtype=bool)).state == R      # 3.5 < 0 + 4
    assert pick([4.5, -9, 0, 0, 0, 0], R, valid=np.array([1, 0, 1, 0, 0, 0], dtype=bool)).state == E


def test_an_invalid_current_state_is_left_at_once_for_the_best_valid_one():
    valid = np.array([1, 0, 0, 0, 1, 1], dtype=bool)                                  # RETURN closed its loop
    assert pick([1.0, -9, 9.0, 0, 1.5, 1.2], R, valid=valid).state == T                # no switching cost to leave it


def test_ties_go_to_the_current_state_or_the_lower_index():
    assert pick([1, -9, 1, 1, 1, 1], T).state == T
    assert pick([1, -9, 1, 1, 1, 1], R, valid=np.array([0, 0, 0, 1, 1, 1], dtype=bool)).state == H


def test_with_nothing_valid_the_current_state_is_kept_for_the_caller_to_handle():
    assert pick([1, -9, 2, 3, 4, 5], T, valid=np.zeros(N_STATES, dtype=bool)).state == T


def test_the_thresholds_come_from_the_config():
    cfg = BehaviorConfig(safety_threshold=3, switch_cost=0.5)
    assert pick([0, -9, 0, 0, 0, 0], E, trail=2, margin=3, cfg=cfg).forced == "safety"
    assert pick([0, -9, 0, 0, 0, 0.6], E, cfg=cfg).state == P


def test_the_shapes_are_checked():
    with pytest.raises(ValueError):
        choose(np.zeros(N_STATES - 1), ALL, E, died=False, trail_len=0, margin=0)
