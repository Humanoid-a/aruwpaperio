"""Whole matches in the real engine, scored against its ground truth."""

import pytest

from candidate.agent.diagnostics import Diagnostics
from candidate.candidate_controller import CandidateController
from candidate.runtime_check import run_check
from candidate.tests.engine_support import play_match
from territorygame.controller.enemy_state_machine import EnemyStateMachine


def assert_clean(result):
    assert result.ok, result.format()
    m = result.metrics
    assert m["adapter_mismatches"] == 0 and m["window_wrong"] == 0
    assert m["memory_hidden_wrong"] <= m["guess_wrong"]    # memory is never worse than the bare guess
    assert m["deaths_inferred"] == m["deaths_true"] and m["kills_inferred"] == m["kills_true"]


def test_against_the_enemy_state_machine_the_pipeline_is_exact_and_captures_land():
    result = run_check("enemy", turns=150)
    assert_clean(result)
    m = result.metrics
    assert m["captures"] >= 10 and m["memory_exact_turns"] == m["observations"]
    d = result.diagnostics
    assert d.counts["pred_result_bad"] == 0 and d.counts["state_bad"] == 0 and d.counts["gain_off"] == 0


@pytest.mark.parametrize("player", [0, 1])
def test_against_a_hunter_deaths_and_kills_are_inferred_exactly(player):
    result = run_check("hunter", turns=250, player=player, close=True, suicide_every=4)
    assert_clean(result)
    m = result.metrics
    assert m["deaths_true"] >= 10 and m["kills_true"] >= 5, "the scenario should produce deaths and kills"
    d = result.diagnostics
    assert d.counts["event_KILLED_BY_OPPONENT"] > 0 and d.counts["event_SELF_DIED"] > 0
    assert m["memory_exact_turns"] == m["observations"]


@pytest.mark.parametrize("opponent", ["random", "basic"])
def test_against_other_opponents_nothing_is_flagged(opponent):
    assert_clean(run_check(opponent, turns=150, close=True))


def test_the_candidate_controller_the_gui_builds_plays_a_whole_match_without_trouble():
    diag = Diagnostics()
    candidate = CandidateController(diagnostics=diag)
    snapshots = play_match([candidate, EnemyStateMachine()], 120, respawns=[(25, 18), (25, 31)])
    assert all(s.error_message is None for s in snapshots), "a raised exception would show up here as a forfeited turn"
    assert snapshots[-1].game_over and diag.errors() == [] and diag.counts["turns"] == 120
    assert diag.counts["pred_result_bad"] == 0


def test_the_runtime_stays_far_inside_any_plausible_per_move_budget():
    result = run_check("enemy", turns=100)
    mean, _, worst = result.diagnostics.timing_stats("runtime")
    assert mean < 5 and worst < 100          # milliseconds, with a wide margin
    mean, _, _ = result.diagnostics.timing_stats("adapter")
    assert mean < 10
