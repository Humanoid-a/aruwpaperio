"""The belief against the real engine over whole matches: the one guarantee that
matters is that the opponent's true position is never given zero probability."""

import pytest

from candidate.runtime_check import run_check


def assert_sound(result):
    assert result.ok, result.format()
    m = result.metrics
    assert m["belief_in_support"] == m["belief_turns"]                 # the truth always had some probability
    if m["seen_turns"]:
        assert m["belief_p_true_seen_min"] > 0.999                     # and when it is in view, it is certain
    assert m["belief_hypotheses_max"] <= 8000                          # inside the configured capacity
    return m


def test_a_hunting_opponent_is_tracked_exactly_and_never_lost():
    m = assert_sound(run_check("hunter", turns=250, close=True))
    assert m["belief_reseeds"] == 0 and m["seen_turns"] > 50
    assert m["belief_exact_mass_mean"] > 0.2                           # real mass on the exact hidden state, not just the cell


def test_the_belief_survives_being_killed_and_killing_as_the_other_player():
    m = assert_sound(run_check("hunter", turns=250, close=True, suicide_every=4, player=1))
    assert m["deaths_true"] > 10 and m["kills_true"] > 5               # a rough match: the belief is reset often
    assert m["belief_reseeds"] == 0


def test_an_opponent_that_makes_loops_and_captures_does_not_confuse_it():
    m = assert_sound(run_check("basic", turns=250, close=True))        # this scenario once zeroed the truth
    assert m["belief_reseeds"] == 0


def test_a_wandering_opponent_is_covered_even_when_it_is_rarely_in_view():
    m = assert_sound(run_check("random", turns=200, close=True))
    assert m["belief_reseeds"] == 0


def test_a_distant_careful_opponent_gives_a_wide_but_valid_belief():
    result = run_check("enemy", turns=150)
    m = assert_sound(result)
    assert m["belief_p_true_mean"] < 0.5                               # it cannot be known where it is, and the belief says so


def test_the_belief_update_stays_within_a_sensible_time_budget():
    result = run_check("random", turns=120, close=True)
    mean, p95, _ = result.diagnostics.timing_stats("belief")
    assert mean < 120 and p95 < 400                                    # milliseconds; generous, it is typically a fraction of this
