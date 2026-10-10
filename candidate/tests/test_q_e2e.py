"""Step 6 of the Q build: the stand-in Q playing whole engine matches, scored against the engine's true state."""

import numpy as np
import pytest

import candidate.agent.q.strategy as strategy_module
from candidate.agent.controller import Controller
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.q import confidence
from candidate.agent.q.config import QConfig
from candidate.agent.q.strategy import QStrategy
from candidate.runtime_check import CLOSE_RESPAWNS, OPPONENTS, _Candidate, run_check
from candidate.tests.engine_support import play_match, world_from_snapshot
from candidate.tests.raider import Raider


def play(cfg, **kwargs):
    """A match with QStrategy built from `cfg`; the substitution lasts only for this call."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(strategy_module, "QStrategy", lambda: QStrategy(cfg=cfg))
        return run_check(strategy="q", **kwargs)


@pytest.mark.parametrize("opponent, player, close", [
    ("enemy", 0, False), ("enemy", 1, True), ("basic", 0, True), ("random", 1, False), ("hunter", 0, True), ("hunter", 1, True),
])
def test_it_plays_a_clean_match_without_dying(opponent, player, close):
    result = run_check(opponent=opponent, turns=200, player=player, close=close, strategy="q")
    stats = result.strategy.stats
    assert result.ok, result.format()
    assert result.diagnostics.counts["anomaly_error"] == 0 and result.diagnostics.counts["forfeits"] == 0
    assert result.metrics["deaths_true"] == 0, result.format()                     # no avoidable deaths
    assert stats["guarded"] <= 2 and stats["retries"] <= 2
    assert sum(v for k, v in stats.items() if k.startswith("state_")) == 200


def test_against_a_raider_that_takes_my_land_the_stand_in_still_plays_clean(monkeypatch):
    monkeypatch.setitem(OPPONENTS, "raider", Raider)
    result = run_check(opponent="raider", turns=200, close=True, strategy="q")
    assert result.ok, result.format()


def test_without_the_look_ahead_the_hunter_kills_it_and_with_it_nothing_does():
    off = play(QConfig(safety_lookahead=False), opponent="hunter", turns=200, close=True)
    on = run_check(opponent="hunter", turns=200, close=True, strategy="q")
    assert off.metrics["deaths_true"] >= 5 and on.metrics["deaths_true"] == 0


def test_it_actually_expands_captures_and_hunts():
    result = run_check(opponent="hunter", turns=300, close=True, strategy="q")
    stats = result.strategy.stats
    assert result.metrics["captures"] >= 8 and result.metrics["kills_true"] >= 3
    assert stats["state_EXPAND"] > 0 and stats["state_RETURN"] > 0 and stats["state_HUNT"] > 0


def test_every_input_is_built_every_frame_and_the_per_frame_cost_is_small_next_to_the_belief():
    result = run_check(opponent="enemy", turns=120, strategy="q")
    timing = result.strategy.timing_ms
    assert all(len(v) == 120 for v in timing.values())
    assert np.mean(timing["snapshot"]) < 40 and np.mean(timing["inputs"]) < 10
    assert np.mean(timing["q"]) < 5 and np.mean(timing["move"]) < 25


# ---- the confidence map against the engine's true board ------------------------------------------------------------------


@pytest.mark.parametrize("opponent", ["hunter", "enemy"])
def test_the_confidence_map_tells_the_truth_about_what_it_is_sure_of(opponent):
    boards = {}

    class Recording(QStrategy):
        def choose(self, view):
            move = super().choose(view)
            if not view.retry:
                boards[view.obs.turn] = (confidence.build_confidence_board(view, self.last.snapshot), view.obs)
            return move

    candidate = _Candidate(Controller(Recording(), Diagnostics()))
    snaps = play_match([candidate, OPPONENTS[opponent]()], 200, respawns=list(CLOSE_RESPAWNS))
    sure = {k: [0, 0] for k in ("enemy land", "enemy trail", "my land")}       # cells out of view with confidence >= 0.99
    for turn, (board, obs) in boards.items():
        world = world_from_snapshot(snaps[2 * turn], 0)
        truth = {"enemy land": world.territory == 3, "enemy trail": world.trail == 4, "my land": world.territory == 1}
        view = np.zeros((50, 50), dtype=bool)
        view[obs.window_slices] = True
        for i, name in enumerate(sure):
            value, conf = board[i] > 0, board[3 + i]
            # what is in view is exactly what the observation says
            assert (value[view] == truth[name][view])[conf[view] == 1.0].all() or name == "enemy trail"
            surely = (conf >= 0.99) & ~view
            sure[name][0] += int(surely.sum())
            sure[name][1] += int((surely & (value != truth[name])).sum())
        assert board[6].flat[world.opp_pos] > 0                                  # the enemy's true cell always has probability
        assert board[6].sum() == pytest.approx(1.0, abs=1e-3)
    for name, (n, wrong) in sure.items():
        assert n > 10_000 and wrong / n <= 0.01, (name, n, wrong)
