"""Both territories belong to each hypothesis, and both territory counts are held against the engine's.

  * a hypothesis carries the opponent's land (opp_land_id) and my land (my_land_id);
  * the opponent's count must match on every hypothesis, every frame;
  * my count is checked too, using each hypothesis' own idea of what its move takes from me.
"""

import numpy as np
import pytest

from candidate.agent.belief.config import BeliefConfig
from candidate.agent.belief.filter import BeliefFilter
from candidate.agent.belief.land import rewrite_my_land
from candidate.agent.belief.settle import rehydrate
from candidate.agent.data.cells import MY_TERRITORY, OPP_TERRITORY
from candidate.agent.data.hypotheses import HypothesisStore
from candidate.agent.data.territory import TerritoryDiffs
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.runtime_check import OPPONENTS, run_check
from candidate.tests.belief_rig import FAR, NEAR, Rig, plain_move, prime
from candidate.tests.builders import make_observation, square
from candidate.tests.raider import Raider


def cells(*yx):
    return [flat(y, x) for y, x in yx]


def bit(*yx):
    return bitboard.from_cells(cells(*yx))


def view_of_mine(belief):
    """My land as each stored hypothesis sees it: (n, 50) bitboards."""
    return belief.state.my_land(belief.state.store.my_land_id)


# ---- the store ------------------------------------------------------------------------


def test_a_hypothesis_built_without_a_view_of_my_land_uses_memorys():
    rig = Rig(NEAR)
    assert rig.state.store.my_land_id.tolist() == [0]


def test_an_unknown_id_for_my_land_is_reported():
    rig = Rig(NEAR)
    store = rig.state.store.take([0])
    store.my_land_id[0] = 7
    assert any("my land" in p for p in store.problems(my_diffs=TerritoryDiffs()))
    store.my_land_id[0] = 0
    assert store.problems(my_diffs=rig.state.my_diffs) == []


def test_hypotheses_that_differ_only_in_what_i_own_are_different_hypotheses():
    rig = Rig(NEAR)
    two = HypothesisStore.concatenate([rig.state.store, rig.state.store])
    two.my_land_id[1] = rig.state.my_diffs.intern(bit((25, 14)), bitboard.empty())
    key = two.merge_key(rig.state.opp_diffs, rig.state.my_diffs)
    assert key[0] != key[1]


# ---- the start of a match -----------------------------------------------------------------


def test_every_hypothesis_starts_out_owning_exactly_my_start_land():
    rig = Rig(NEAR)
    assert (bitboard.popcount(view_of_mine(rig.belief)) == 9).all()
    land = {c: MY_TERRITORY for c in square(25, 10)}
    rig.step(make_observation((25, 10), turn=0, respawn=(25, 10), territory=land))
    assert (bitboard.popcount(view_of_mine(rig.belief)) == 9).all()
    assert (bitboard.popcount(rig.state.opp_land(rig.state.store.opp_land_id)) >= 9).all()


# ---- the gate on my count: a scene with two views of my land -----------------------------------


def two_views_capture(observed_my_count, view_a, view_b, count_after=9):
    """Two hypotheses about to close the same loop east into its land. The loop encloses (25, 14), so a hypothesis
    that thinks (25, 14) is mine believes the capture takes one cell from me. They differ only in their view of my land."""
    rig = Rig(NEAR)
    trail = [(24, 14), (24, 13), (25, 13), (26, 13), (26, 14)]
    prime(rig, [(flat(26, 14), trail, 0.5, 9), (flat(26, 14), trail, 0.5, 9)])
    state = rig.state
    state.my_diffs = TerritoryDiffs()
    state.store.my_land_id[:] = state.my_diffs.intern_territory(np.stack([view_a, view_b]), state.my_base)
    land = {c: MY_TERRITORY for c in square(25, 10)}
    theirs = [(24, 15), (25, 15), (26, 15), (24, 14), (24, 13), (25, 13), (26, 13), (26, 14), (25, 14)]
    land.update({c: OPP_TERRITORY for c in theirs})
    obs = make_observation((25, 10), turn=5, respawn=(25, 10), opp_at=(26, 15), opp_count=15,
                           my_count=observed_my_count, territory=land)
    rig.move(plain_move([], pos_after=flat(25, 10), count=count_after))
    rig.step(obs)
    return rig


MINE = NEAR.me.start_bb
LOSES_A_CELL = (MINE & ~bit((24, 9))) | bit((25, 14))      # nine cells, one of them inside the loop
KEEPS_ALL = MINE                                           # nine cells, none inside the loop


@pytest.mark.parametrize("observed, survivor_cells", [(8, 8), (9, 9)])
def test_the_hypothesis_whose_view_of_my_land_explains_my_count_survives(observed, survivor_cells):
    rig = two_views_capture(observed, LOSES_A_CELL, KEEPS_ALL)
    assert rig.state.store.n == 1
    assert bitboard.popcount(view_of_mine(rig.belief)[0]) == survivor_cells
    assert rig.belief.stats.cut_by_my_count >= 1


def test_a_hypothesis_whose_view_of_my_land_is_off_is_not_judged_by_it():
    # The first thinks I own ten cells, but the engine says nine: its idea of what it takes from me cannot be trusted.
    rig = two_views_capture(9, MINE | bit((25, 14)), KEEPS_ALL)
    # Neither is cut. Both end up owning the same nine cells (the first loses the one inside the loop), so they merge.
    assert rig.belief.stats.cut_by_my_count == 0
    assert rig.state.store.n == 1 and rig.state.store.weight.sum() == pytest.approx(1.0)


# ---- guessed histories ---------------------------------------------------------------------------------


def test_guessed_histories_inherit_the_most_likely_view_of_my_land():
    rig = Rig(FAR)
    prime(rig, [(flat(30, 30), [], 0.2, 9)])
    state = rig.state
    rewrite_my_land(state, state.my_base, add=bit((10, 10)))
    state.store.weight = np.array([0.2])
    state.residual.mass[12, 12] = 0.8
    rehydrate(state, BeliefConfig(rehydrate_below=0.9, rehydrate_fraction=0.5), rig.belief.stats, 9)
    assert state.store.n > 1
    assert all(bitboard.has_cell(v, flat(10, 10)) for v in view_of_mine(rig.belief))


# ---- against the real engine -----------------------------------------------------------------------------


@pytest.fixture
def audit(monkeypatch):
    """Record, after each frame, how many stored hypotheses misstate either count, and when my remembered
    land first stopped matching the engine's count (after which memory itself is out of date)."""
    log = {"frames": 0, "opp_wrong": 0, "my_wrong": 0, "my_wrong_before_drift": 0, "drift_turn": None}
    original = BeliefFilter.observe

    def audited(self, obs, memory, *args, **kwargs):
        view = original(self, obs, memory, *args, **kwargs)
        state = self.state
        store = state.store
        drifted = bitboard.popcount(state.my_base) != obs.my_count
        if drifted and log["drift_turn"] is None:
            log["drift_turn"] = obs.turn
        if store.n:
            wrong = bitboard.popcount(view_of_mine(self)) != obs.my_count
            log["frames"] += 1
            log["opp_wrong"] += int((store.opp_count != obs.opp_count).sum())
            log["my_wrong"] += int(wrong.sum())
            if log["drift_turn"] is None:
                log["my_wrong_before_drift"] += int(wrong.sum())
        return view

    monkeypatch.setattr(BeliefFilter, "observe", audited)
    return log


@pytest.mark.parametrize("kwargs", [
    dict(opponent="hunter", turns=200, close=True),
    dict(opponent="hunter", turns=200, close=True, suicide_every=4, player=1),
    dict(opponent="basic", turns=200, close=True),
])
def test_every_hypothesis_matches_both_engine_counts_in_every_frame(audit, kwargs):
    result = run_check(**kwargs)
    assert result.ok, result.format()
    assert audit["frames"] > 100
    assert audit["opp_wrong"] == 0
    assert audit["my_wrong"] == 0, audit


def test_an_opponent_that_captures_my_land_does_not_break_the_belief(audit, monkeypatch):
    monkeypatch.setitem(OPPONENTS, "raider", Raider)
    result = run_check("raider", turns=300, close=True)
    m = result.metrics
    assert result.ok, result.format()
    assert m["belief_in_support"] == m["belief_turns"] and m["belief_reseeds"] == 0
    assert audit["opp_wrong"] == 0
    assert audit["drift_turn"] is not None                         # it did take land from me, out of view
    assert audit["my_wrong_before_drift"] == 0                     # and until it did, every view of my land was exact
