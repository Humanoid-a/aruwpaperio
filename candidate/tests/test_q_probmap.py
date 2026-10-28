"""Q's probability map (docs/Q_Probability_Map_Input.md): layers that sum to 1, the residual and the reach region,
confidence, visible cells, positions, and pooling."""

import math

import numpy as np
import pytest

from candidate.agent.behavior.snapshot import build_snapshot
from candidate.agent.geometry.grid import flat
from candidate.agent.q import grid, probmap
from candidate.agent.q.probmap import (ENEMY, ENEMY_POS, ENEMY_TRAIL, FREE, MINE, MY_POS, MY_TRAIL_CH, NONE, OWN_CONF,
                                       TRAIL_CONF)
from candidate.tests.q_support import CLOSE, belief_of, bb, rotate_world, scene, world_with

OPP_LAND = [(y, x) for y in range(24, 27) for x in range(30, 33)]
HALF_CONF = 1.0 - math.log(2) / math.log(3)              # confidence of a 1/2, 1/2, 0 split


def board_of(view):
    return probmap.build_probability_board(view, build_snapshot(view))


def window_of(view):
    mask = np.zeros((50, 50), dtype=bool)
    mask[view.obs.window_slices] = True
    return mask


def all_residual(world, turn=30, seen=(flat(25, 40), 20)):
    """A belief with no history at all, last seen at `seen` (cell, turn)."""
    return scene(world, CLOSE, belief_of(residual=np.ones((50, 50))), turn=turn, opp_last_seen=seen)


def test_shapes_types_and_both_groups_sum_to_one_on_every_cell():
    view = scene(world_with(CLOSE, my_trail=[(25, 20)]), CLOSE, belief_of(
        [dict(pos=flat(25, 40), weight=1.0, opp_land=bb((10, 10)))], residual=np.ones((50, 50)) * 0.001), turn=30,
        opp_last_seen=(flat(25, 40), 20))
    board = board_of(view)
    assert board.shape == (10, 50, 50) and board.dtype == np.float32
    assert np.allclose(board[FREE:ENEMY + 1].sum(axis=0), 1.0, atol=1e-5)
    assert np.allclose(board[NONE:ENEMY_TRAIL + 1].sum(axis=0), 1.0, atol=1e-5)
    assert ((board[[OWN_CONF, TRAIL_CONF]] >= -1e-6) & (board[[OWN_CONF, TRAIL_CONF]] <= 1 + 1e-6)).all()
    pooled = probmap.build_probmap(view, build_snapshot(view))
    assert pooled.shape == (10, 10, 10) and pooled.dtype == np.float32


def test_one_hypothesis_and_no_residual_is_the_true_board_with_confidence_one():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21)], opp_trail=[(25, 36), (25, 35)], opp_land=OPP_LAND)
    board = board_of(scene(world, CLOSE))
    assert np.array_equal(board[ENEMY] == 1, world.territory == 3)
    assert np.array_equal(board[MINE] == 1, world.territory == 1)
    assert np.array_equal(board[FREE] == 1, world.territory == 0)
    assert np.array_equal(board[ENEMY_TRAIL] == 1, world.trail == 4)
    assert np.array_equal(board[MY_TRAIL_CH] == 1, world.trail == 2)
    assert np.allclose(board[[OWN_CONF, TRAIL_CONF]], 1.0, atol=1e-6)


def test_visible_cells_are_what_is_seen_even_when_the_belief_knows_nothing():
    view = all_residual(world_with(CLOSE, my_trail=[(25, 20)]))
    board, inside = board_of(view), window_of(view)
    assert np.allclose(board[[OWN_CONF, TRAIL_CONF]][:, inside], 1.0, atol=1e-6)
    assert board[MINE, 25, 18] == 1.0 and board[MY_TRAIL_CH, 25, 20] == 1.0 and board[FREE, 25, 22] == 1.0


def test_residual_ownership_inside_the_reach_region_follows_memory_by_halves():
    # last seen 10 turns ago at (25, 40); memory (the whole true board): (30, 40) is mine, its land is enemy, the rest free
    world = world_with(CLOSE, my_land=[(30, 40)], opp_land=OPP_LAND)
    view = all_residual(world)
    board = board_of(view)
    assert probmap.reach_region(flat(25, 40), 10, view.obs)[30, 40]
    assert board[MINE, 30, 40] == pytest.approx(0.5) and board[ENEMY, 30, 40] == pytest.approx(0.5)     # mine -> half
    assert board[FREE, 25, 38] == pytest.approx(0.5) and board[ENEMY, 25, 38] == pytest.approx(0.5)     # free -> half
    assert board[ENEMY, 25, 31] == pytest.approx(1.0)                                                    # enemy stays
    assert board[OWN_CONF, 25, 38] == pytest.approx(HALF_CONF, abs=1e-6)


def test_residual_outside_the_reach_region_is_memorys_category_and_no_trail():
    world = world_with(CLOSE, opp_land=OPP_LAND + [(10, 10)], my_land=[(40, 40)])
    view = all_residual(world, turn=30, seen=(flat(25, 40), 28))                  # seen 2 turns ago: a tiny region
    board = board_of(view)
    assert board[ENEMY, 10, 10] == pytest.approx(1.0) and board[MINE, 40, 40] == pytest.approx(1.0)
    assert board[FREE, 5, 5] == pytest.approx(1.0) and board[OWN_CONF, 5, 5] == pytest.approx(1.0)
    assert board[NONE, 5, 5] == pytest.approx(1.0) and board[TRAIL_CONF, 5, 5] == pytest.approx(1.0)


def test_residual_trail_is_a_coin_flip_in_the_reach_region_but_never_on_my_trail():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21), (25, 22), (25, 23), (25, 24), (25, 25), (25, 26)])
    view = all_residual(world, turn=40, seen=(flat(25, 40), 20))                  # radius 20 reaches my trail
    board, reach = board_of(view), probmap.reach_region(flat(25, 40), 20, view.obs)
    assert board[ENEMY_TRAIL, 25, 38] == pytest.approx(0.5) and board[NONE, 25, 38] == pytest.approx(0.5)
    assert board[TRAIL_CONF, 25, 38] == pytest.approx(HALF_CONF, abs=1e-6)
    mine = np.zeros((50, 50), dtype=bool)
    mine.flat[list(view.obs.my_trail)] = True
    assert (mine & reach).any()                                                  # some of my trail is out of view and in reach
    assert (board[MY_TRAIL_CH][mine] == 1.0).all() and (board[ENEMY_TRAIL][mine] == 0.0).all()


def test_hypotheses_vote_and_a_split_between_mine_and_enemy_reads_as_a_coin_flip_not_as_free():
    cell = (30, 11)
    mine, enemy = bb(cell), bb(cell)
    belief = belief_of([dict(pos=flat(25, 40), weight=45, my_land=mine), dict(pos=flat(25, 40), weight=45, opp_land=enemy),
                        dict(pos=flat(25, 40), weight=10)])
    board = board_of(scene(world_with(CLOSE), CLOSE, belief))
    y, x = cell
    assert board[MINE, y, x] == pytest.approx(0.45) and board[ENEMY, y, x] == pytest.approx(0.45)
    assert board[FREE, y, x] == pytest.approx(0.10)


def test_a_cell_a_hypothesis_calls_its_land_is_not_also_mine_under_it():
    both = bb((30, 11))
    belief = belief_of([dict(pos=flat(25, 40), weight=1.0, opp_land=both, my_land=both)])
    board = board_of(scene(world_with(CLOSE), CLOSE, belief))
    assert board[ENEMY, 30, 11] == pytest.approx(1.0) and board[MINE, 30, 11] == pytest.approx(0.0)


def test_enemy_trail_is_the_weight_of_the_hypotheses_whose_trail_has_the_cell():
    belief = belief_of([dict(pos=flat(25, 40), weight=1.0, trail=[flat(25, 40), flat(25, 41)]),
                        dict(pos=flat(25, 40), weight=3.0, trail=[flat(25, 40)])])
    board = board_of(scene(world_with(CLOSE), CLOSE, belief))
    assert board[ENEMY_TRAIL, 25, 41] == pytest.approx(0.25) and board[ENEMY_TRAIL, 25, 40] == pytest.approx(1.0)


def test_the_enemy_position_is_the_probability_and_a_visible_enemy_is_certain():
    belief = belief_of([dict(pos=flat(25, 40), weight=1.0), dict(pos=flat(20, 40), weight=3.0)])
    board = board_of(scene(world_with(CLOSE), CLOSE, belief))
    assert board[ENEMY_POS, 25, 40] == pytest.approx(0.25) and board[ENEMY_POS, 20, 40] == pytest.approx(0.75)
    seen = board_of(scene(world_with(CLOSE, opp=(25, 22)), CLOSE))
    assert seen[ENEMY_POS, 25, 22] == 1.0 and seen[ENEMY_POS].sum() == 1.0


def test_pooling_averages_the_layers_and_confidences_and_sums_the_positions():
    view = scene(world_with(CLOSE), CLOSE, belief_of([dict(pos=flat(25, 40), weight=1.0)]))
    pooled = probmap.build_probmap(view, build_snapshot(view))
    assert pooled[ENEMY_POS].sum() == pytest.approx(1.0) and pooled[MY_POS].sum() == pytest.approx(1.0)
    assert np.allclose(pooled[FREE:ENEMY + 1].sum(axis=0), 1.0, atol=1e-5)
    board = np.zeros((10, 50, 50), dtype=np.float32)
    board[MINE, :5, :3] = 1.0
    board[FREE, :5, 3:5] = 1.0
    board[OWN_CONF, :5, :5] = 1.0                                     # block (0, 0): 60% mine, 40% free, all certain
    out = probmap.pool(board)
    assert out[MINE, 0, 0] == pytest.approx(0.6) and out[FREE, 0, 0] == pytest.approx(0.4) and out[OWN_CONF, 0, 0] == 1.0


def test_a_certain_mixed_block_differs_from_a_block_of_coin_flips():
    view = scene(world_with(CLOSE), CLOSE, belief_of([dict(pos=flat(25, 40), weight=1.0, my_land=bb((0, 0))),
                                                       dict(pos=flat(25, 40), weight=1.0)]))
    board = board_of(view)
    assert board[MINE, 0, 0] == pytest.approx(0.5) and board[OWN_CONF, 0, 0] == pytest.approx(HALF_CONF, abs=1e-6)
    assert board[OWN_CONF, 0, 1] == pytest.approx(1.0)                # everyone agrees it is free


def test_the_map_turns_with_the_board():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21)], opp_trail=[(25, 36), (25, 35)], opp_land=OPP_LAND)
    base = probmap.pool(board_of(scene(world, CLOSE)))
    for k, flip in ((1, False), (2, True)):
        turned_world, turned_rules = rotate_world(world, CLOSE, k, flip)
        turned = probmap.pool(board_of(scene(turned_world, turned_rules)))
        assert np.allclose(turned, grid.rotate(base, k, flip), atol=1e-6)
