"""Step 3 of the Q build: the confidence map, against the brief's acceptance checks."""

import numpy as np
import pytest

from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.agent.q import confidence, grid
from candidate.agent.q.snapshot import build_snapshot
from candidate.tests.q_support import CLOSE, belief_of, bb, cells, rotate_world, scene, world_with

OPP_LAND = [(y, x) for y in range(24, 27) for x in range(30, 33)]
VALUE_LAND, VALUE_TRAIL, VALUE_MINE, CONF_LAND, CONF_TRAIL, CONF_MINE, POS, MYPOS = range(8)


def board_of(view):
    return confidence.build_confidence_board(view, build_snapshot(view))


def window_of(view):
    mask = np.zeros((50, 50), dtype=bool)
    mask[view.obs.window_slices] = True
    return mask


def test_shapes_and_types():
    view = scene(world_with(CLOSE), CLOSE)
    snap = build_snapshot(view)
    assert confidence.build_confidence_board(view, snap).shape == (8, 50, 50)
    pooled = confidence.build_confidence(view, snap)
    assert pooled.shape == (8, 10, 10) and pooled.dtype == np.float32


def test_one_hypothesis_and_no_residual_gives_confidence_one_everywhere():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21)], opp_trail=[(25, 36), (25, 35)], opp_land=OPP_LAND)
    board = board_of(scene(world, CLOSE))
    assert np.allclose(board[CONF_LAND:CONF_MINE + 1], 1.0)
    # and the values are the true board
    assert np.array_equal(board[VALUE_LAND] > 0, world.territory == 3)
    assert np.array_equal(board[VALUE_MINE] > 0, world.territory == 1)
    assert np.array_equal(board[VALUE_TRAIL] > 0, world.trail == 4)


def test_visible_cells_have_confidence_one_for_every_item_even_when_the_belief_is_a_coin_flip():
    view = scene(world_with(CLOSE), CLOSE, belief_of(residual=np.ones((50, 50))))
    board = board_of(view)
    inside = window_of(view)
    for c in (CONF_LAND, CONF_TRAIL, CONF_MINE):
        assert (board[c][inside] == 1.0).all()


def test_all_residual_gives_enemy_trail_confidence_zero_throughout_the_reach_region():
    view = scene(world_with(CLOSE), CLOSE, belief_of(residual=np.ones((50, 50))), turn=30,
                 opp_last_seen=(flat(25, 40), 20))                     # last seen 10 turns ago at (25, 40)
    board = board_of(view)
    reach = confidence.reach_region(flat(25, 40), 10, view.obs)
    assert reach.sum() > 100
    assert np.allclose(board[CONF_TRAIL][reach], 0.0, atol=1e-6)
    assert np.allclose(board[CONF_LAND][reach], 0.0, atol=1e-6)        # a coin flip about its land there too
    outside = ~reach & ~window_of(view)
    assert np.allclose(board[CONF_TRAIL][outside], 1.0, atol=1e-6)     # out of reach it cannot have laid a trail
    assert np.allclose(board[VALUE_TRAIL][outside], 0.0)


def test_outside_the_reach_region_residual_mass_falls_back_on_memory():
    world = world_with(CLOSE, opp_land=OPP_LAND + [(10, 10)], my_land=[(40, 40)])
    view = scene(world, CLOSE, belief_of(residual=np.ones((50, 50))), turn=30, opp_last_seen=(flat(25, 40), 28))
    board = board_of(view)
    assert board[VALUE_LAND, 10, 10] == 1.0 and board[CONF_LAND, 10, 10] == pytest.approx(1.0)
    assert board[VALUE_MINE, 40, 40] == 1.0 and board[CONF_MINE, 40, 40] == pytest.approx(1.0)


def test_my_land_is_uncertain_where_hypotheses_disagree_about_an_enemy_capture():
    mine = bb((30, 10), (30, 11), (30, 12))
    took = bb((30, 10), (30, 12))                       # in this hypothesis the enemy captured (30, 11) from me
    belief = belief_of([dict(pos=flat(25, 40), weight=3.0, my_land=mine),
                        dict(pos=flat(25, 40), weight=1.0, my_land=took)])
    world = world_with(CLOSE, my_land=[(30, 10), (30, 11), (30, 12)])
    board = board_of(scene(world, CLOSE, belief))
    assert board[VALUE_MINE, 30, 11] == 1.0                            # 0.75 of the mass says it is still mine
    assert board[CONF_MINE, 30, 11] == pytest.approx(0.5)              # |2 x 0.75 - 1|
    assert board[CONF_MINE, 30, 10] == pytest.approx(1.0)              # they agree about the others


def test_enemy_land_is_a_weighted_vote_of_the_hypotheses():
    a, b = bb((10, 10), (10, 11)), bb((10, 11), (10, 12))
    belief = belief_of([dict(pos=flat(25, 40), weight=1.0, opp_land=a), dict(pos=flat(25, 40), weight=3.0, opp_land=b)])
    board = board_of(scene(world_with(CLOSE), CLOSE, belief))
    assert board[VALUE_LAND, 10, 10] == 0.0 and board[CONF_LAND, 10, 10] == pytest.approx(0.5)     # 0.25
    assert board[VALUE_LAND, 10, 11] == 1.0 and board[CONF_LAND, 10, 11] == pytest.approx(1.0)
    assert board[VALUE_LAND, 10, 12] == 1.0 and board[CONF_LAND, 10, 12] == pytest.approx(0.5)     # 0.75


def test_the_enemy_position_channel_is_the_probability_itself():
    belief = belief_of([dict(pos=flat(25, 40), weight=1.0), dict(pos=flat(20, 40), weight=3.0)])
    board = board_of(scene(world_with(CLOSE), CLOSE, belief))
    assert board[POS, 25, 40] == pytest.approx(0.25) and board[POS, 20, 40] == pytest.approx(0.75)
    assert board[POS].sum() == pytest.approx(1.0)


def test_a_visible_enemy_is_one_cell_with_probability_one():
    world = world_with(CLOSE, opp=(25, 22))
    board = board_of(scene(world, CLOSE))
    assert board[POS, 25, 22] == 1.0 and board[POS].sum() == 1.0


def test_after_pooling_the_two_position_channels_each_sum_to_one():
    residual = np.random.default_rng(0).random((50, 50))
    residual[20:31, 13:24] = 0.0                       # a real belief has nothing in the view when the enemy is not seen
    belief = belief_of([dict(pos=flat(25, 40), weight=1.0)], residual=residual)
    view = scene(world_with(CLOSE), CLOSE, belief)
    assert not window_of(view)[residual > 0].any()
    pooled = confidence.build_confidence(view, build_snapshot(view))
    assert pooled[POS].sum() == pytest.approx(1.0, abs=1e-5) and pooled[MYPOS].sum() == pytest.approx(1.0)
    my_y, my_x = divmod(view.obs.my_pos, 50)
    assert pooled[MYPOS, my_y // 5, my_x // 5] == 1.0


def test_pooling_averages_values_and_confidences_and_sums_positions():
    board = np.zeros((8, 50, 50), dtype=np.float32)
    board[VALUE_LAND, :5, :5] = 1.0
    board[VALUE_LAND, :2, :5] = 0.0                                    # 15 of the 25 cells believed enemy land
    board[CONF_LAND, :5, :5] = 1.0
    board[POS, 0, 0] = 0.25
    board[POS, 4, 4] = 0.5
    pooled = confidence.pool(board)
    assert pooled[VALUE_LAND, 0, 0] == pytest.approx(15 / 25) and pooled[CONF_LAND, 0, 0] == 1.0
    assert pooled[POS, 0, 0] == pytest.approx(0.75)


def test_a_certain_half_and_half_block_differs_from_a_block_we_know_nothing_about():
    board = np.zeros((8, 50, 50), dtype=np.float32)
    board[VALUE_LAND, :5, :3] = 1.0
    board[CONF_LAND, :5, :5] = 1.0                                     # block (0, 0): half enemy, all certain
    pooled = confidence.pool(board)
    assert pooled[VALUE_LAND, 0, 0] == pytest.approx(0.6) and pooled[CONF_LAND, 0, 0] == 1.0
    assert pooled[VALUE_LAND, 0, 1] == 0.0 and pooled[CONF_LAND, 0, 1] == 0.0       # unknown block


def test_pooling_commutes_with_rotating_and_flipping_the_board():
    rng = np.random.default_rng(3)
    board = rng.random((8, 50, 50)).astype(np.float32)
    for k, flip in ((1, False), (2, False), (3, False), (0, True), (1, True)):
        assert np.allclose(confidence.pool(grid.rotate(board, k, flip)), grid.rotate(confidence.pool(board), k, flip), atol=1e-6)


def test_rotating_the_whole_scene_rotates_the_confidence_map():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21), (24, 21)], opp_trail=[(26, 36), (26, 35)], opp_land=OPP_LAND)
    base = board_of(scene(world, CLOSE))
    for k, flip in ((1, False), (2, False), (3, True)):
        turned, rules = rotate_world(world, CLOSE, k, flip)
        assert np.allclose(board_of(scene(turned, rules)), grid.rotate(base, k, flip), atol=1e-6), (k, flip)


def test_the_reach_region_is_a_diamond_without_the_view():
    view = scene(world_with(CLOSE), CLOSE)
    reach = confidence.reach_region(flat(25, 30), 4, view.obs)
    assert reach[25, 34] and reach[21, 30] and not reach[25, 35] and not reach[22, 33]    # |dy| + |dx| <= 4
    assert not reach[view.obs.window_slices].any()


def test_before_any_sighting_the_reach_is_centred_on_the_enemys_respawn():
    view = scene(world_with(CLOSE), CLOSE, belief_of(residual=np.ones((50, 50))), turn=8)
    board = board_of(view)
    near = confidence.reach_region(CLOSE.opp.respawn, 8, view.obs)
    assert near[25, 31] and np.allclose(board[CONF_TRAIL][near], 0.0, atol=1e-6)
    assert not near[5, 5] and board[CONF_TRAIL, 5, 5] == pytest.approx(1.0)
