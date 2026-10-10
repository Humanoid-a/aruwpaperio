"""Step 1 of the Q build: the snapshot (D1..D8) on small hand-built boards."""

import numpy as np
import pytest

from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.agent.q import votes
from candidate.agent.q.config import QConfig
from candidate.agent.q.snapshot import build_snapshot
from candidate.tests.q_support import CLOSE, belief_of, bb, cells, scene, world_with

OPP_LAND = [(y, x) for y in range(24, 27) for x in range(30, 33)]


def out_east(n=3):
    return [(25, 19 + k) for k in range(1, n + 1)]


def test_a_straight_trail_with_the_enemy_far_away():
    view = scene(world_with(CLOSE, my_trail=out_east(3)), CLOSE)
    snap = build_snapshot(view)
    assert snap.trail_len == 3 and not snap.on_land
    assert snap.d1 == 9                  # the enemy is certainly at (25, 31); my nearest trail cell is (25, 22)
    # my own trail blocks the straight way back, so home is up and round: 4 steps
    assert snap.d4 == 4 and snap.home_path == cells((24, 22), (24, 21), (24, 20), (24, 19))
    assert snap.d7 == 5
    assert snap.d2 == 50 and snap.d3 == 0.0 and snap.d8 == 50.0      # no enemy trail; it stands on its land
    assert snap.d5 == 6                  # three trail cells and the three new ones on the way home, nothing enclosed
    assert snap.d6 == 0                  # the enemy has no trail


def test_without_a_trail_the_gains_and_the_trail_distance_are_none():
    snap = build_snapshot(scene(world_with(CLOSE), CLOSE))
    assert snap.trail_len == 0 and snap.on_land and snap.d1 == 50 and snap.d5 == 0 and snap.d4 == 0
    assert snap.home_path == []


def test_d7_is_d1_minus_d4_and_d8_is_d2_minus_d3_on_many_scenes():
    rng = np.random.default_rng(5)
    for _ in range(25):
        n = int(rng.integers(0, 6))
        enemy_at = (int(rng.integers(20, 30)), int(rng.integers(20, 40)))
        trail = out_east(n) if n else ()
        et = [(enemy_at[0], enemy_at[1] + k) for k in range(3)] if rng.random() < 0.5 else ()
        view = scene(world_with(CLOSE, my_trail=trail, opp_trail=et, opp=enemy_at if not et else None), CLOSE)
        snap = build_snapshot(view)
        assert snap.d7 == snap.d1 - snap.d4 and snap.d8 == snap.d2 - snap.d3


def test_the_margin_offset_shifts_d7_by_a_fixed_number_of_turns():
    view = scene(world_with(CLOSE, my_trail=out_east(3)), CLOSE)
    assert build_snapshot(view, QConfig(margin_offset=-1)).d7 == build_snapshot(view).d7 - 1


def test_a_visible_enemy_trail_is_known_and_reachable_but_not_under_the_enemy():
    world = world_with(CLOSE, opp_trail=[(25, 25), (25, 24), (25, 23)], opp_land=OPP_LAND)
    snap = build_snapshot(scene(world, CLOSE))
    assert bitboard.has_cell(snap.known_trail_bb, flat(25, 23)) and bitboard.has_cell(snap.known_trail_bb, flat(25, 24))
    assert snap.enemy_seen and bitboard.has_cell(snap.enemy_cell_bb, flat(25, 23))
    assert not bitboard.has_cell(snap.hunt_targets_bb, flat(25, 23))       # the enemy agent stands there
    # (25,24) is the nearest target, but the enemy agent at (25,23) blocks the straight row: round it is 8
    assert snap.d2 == 8 and snap.enemy_trail_known
    assert snap.d3 == 7.0                                                   # (25,23) to its land at x=30 is 7 away


def test_an_unseen_enemys_distance_home_is_averaged_over_the_position_marginal():
    belief = belief_of([dict(pos=flat(25, 40), weight=1.0), dict(pos=flat(25, 36), weight=1.0)])
    snap = build_snapshot(scene(world_with(CLOSE), CLOSE, belief))
    assert snap.d3 == pytest.approx((8 + 4) / 2)      # x=40 is 8 from x=32, x=36 is 4 from x=32


def test_the_possibility_mask_is_the_beliefs_support_and_sets_the_enemys_worst_case_arrival():
    world = world_with(CLOSE, my_trail=out_east(2))                   # trail cells (25,20), (25,21)
    belief = belief_of([dict(pos=flat(25, 28), weight=1.0)])
    snap = build_snapshot(scene(world, CLOSE, belief))
    assert snap.possible.sum() == 1 and snap.possible[25, 28]
    assert snap.d1 == 7                                  # (25, 28) to my nearest trail cell (25, 21)
    wider = belief_of([dict(pos=flat(25, 28), weight=1.0), dict(pos=flat(25, 24), weight=1.0)])
    assert build_snapshot(scene(world, CLOSE, wider)).d1 == 3          # (25, 24) is 3 from (25, 21)


def test_a_probability_floor_can_shrink_the_possibility_mask():
    world = world_with(CLOSE, my_trail=out_east(2))
    belief = belief_of([dict(pos=flat(25, 28), weight=1.0), dict(pos=flat(25, 24), weight=0.001)])
    assert build_snapshot(scene(world, CLOSE, belief)).d1 == 3
    assert build_snapshot(scene(world, CLOSE, belief), QConfig(possible_min_probability=0.01)).d1 == 7


def test_an_enemy_trail_the_belief_is_sure_of_counts_as_known_even_out_of_view():
    world = world_with(CLOSE)
    sure = belief_of([dict(pos=flat(25, 40), weight=1.0, trail=cells((25, 38), (25, 39), (25, 40)))])
    unsure = belief_of([dict(pos=flat(25, 40), weight=1.0, trail=cells((25, 39), (25, 40))),
                        dict(pos=flat(20, 40), weight=1.2)])
    assert build_snapshot(scene(world, CLOSE, sure)).enemy_trail_known
    assert not build_snapshot(scene(world, CLOSE, unsure)).enemy_trail_known      # 0.45 < 0.5


def test_the_enemys_gain_if_it_went_home_counts_my_cells_double():
    # Its trail leaves its land and bends round; going home closes a loop whose cells are partly mine.
    ring = [(24, 29), (24, 28), (25, 28), (26, 28), (26, 29)]
    mine_inside = [(25, 29)]                                   # my cell, enclosed by its loop
    world = world_with(CLOSE, opp_trail=ring, opp_land=OPP_LAND, my_land=mine_inside)
    with_mine = build_snapshot(scene(world, CLOSE)).d6
    without = build_snapshot(scene(world_with(CLOSE, opp_trail=ring, opp_land=OPP_LAND), CLOSE)).d6
    assert with_mine == without + 1                            # the enclosed cell of mine counts twice instead of once


def test_my_gain_counts_the_enemys_land_double():
    world = world_with(CLOSE, my_trail=[(25, 20), (24, 20)], opp_land=[(25, 20)])
    snap = build_snapshot(scene(world, CLOSE))
    # trail cells (25,20) [enemy land: counts 2] and (24,20) [free: 1]; home is one step away from (24,20)
    assert snap.d5 == 3


def test_the_hypothesis_votes_are_weights_summed_per_cell():
    a, b = bb((5, 5), (5, 6)), bb((5, 6), (5, 7))
    belief = belief_of([dict(pos=0, weight=1.0, opp_land=a, trail=cells((5, 5))),
                        dict(pos=1, weight=3.0, opp_land=b, trail=cells((5, 5), (5, 6)))])
    d = belief.detail
    land, trail = votes.enemy_land_votes(d), votes.enemy_trail_votes(d)
    assert land[5, 5] == pytest.approx(0.25) and land[5, 6] == pytest.approx(1.0) and land[5, 7] == pytest.approx(0.75)
    assert trail[5, 5] == pytest.approx(1.0) and trail[5, 6] == pytest.approx(0.75) and trail[5, 7] == 0


def test_the_trail_vote_is_the_same_whatever_the_chunk_size(monkeypatch):
    rng = np.random.default_rng(1)
    trails = bitboard.from_mask(rng.random((700, 50, 50)) < 0.05)
    weight = rng.random(700)
    whole = votes.trail_votes(trails, weight)
    monkeypatch.setattr(votes, "CHUNK", 64)
    assert np.allclose(votes.trail_votes(trails, weight), whole)


def test_the_snapshot_needs_a_belief_with_detail():
    view = scene(world_with(CLOSE), CLOSE)
    view.belief = None
    with pytest.raises(ValueError):
        build_snapshot(view)
