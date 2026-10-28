"""Step 2 of the Q build: the 23 scalars and the 7 x 21 x 21 grid."""

import numpy as np
import pytest

from candidate.agent.geometry.grid import flat
from candidate.agent.q import features, grid
from candidate.agent.q.config import QConfig
from candidate.agent.behavior.snapshot import build_snapshot
from candidate.agent.behavior.states import State
from candidate.agent.rules.match import MatchRules
from candidate.tests.q_support import CLOSE, belief_of, rotate_world, scene, world_with

OPP_LAND = [(y, x) for y in range(24, 27) for x in range(30, 33)]


def scalars(view, state=State.PATROL, cfg=None):
    snap = build_snapshot(view)
    return features.build_scalars(view, snap, state) if cfg is None else features.build_scalars(view, snap, state, cfg)


# ---- scalars ---------------------------------------------------------------------------------------------


def test_there_are_23_scalars_in_the_briefs_order():
    assert features.N_SCALARS == 23 and len(features.NAMES) == 23
    out = scalars(scene(world_with(CLOSE), CLOSE))
    assert out.shape == (23,) and out.dtype == np.float32


def test_scaling_follows_the_brief():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21), (25, 22)], my_land=[(y, x) for y in range(10, 15) for x in range(10, 20)])
    view = scene(world, CLOSE)
    snap = build_snapshot(view)
    f = dict(zip(features.NAMES, scalars(view, State.EXPAND)))
    obs = view.obs
    assert f["trail_len"] == pytest.approx(3 / 100)
    assert f["my_count"] == pytest.approx(obs.my_count / 500) and f["enemy_count"] == pytest.approx(obs.opp_count / 500)
    assert f["score_diff"] == pytest.approx((obs.my_count - obs.opp_count) / 500)
    assert f["d1"] == pytest.approx(snap.d1 / 50) and f["d4"] == pytest.approx(snap.d4 / 50)
    assert f["d5"] == pytest.approx(snap.d5 / 100) and f["d6"] == 0.0
    assert f["d7"] == pytest.approx(snap.d7 / 25)
    assert f["turns_left"] == pytest.approx(obs.turns_left / 2000)


def test_distances_and_margins_are_clipped():
    view = scene(world_with(CLOSE), CLOSE)                       # no trail: D1 = 50, no enemy trail: D2 = 50
    f = dict(zip(features.NAMES, scalars(view)))
    assert f["d1"] == 1.0 and f["d2"] == 1.0
    assert f["d7"] == 1.0 and f["d8"] == 1.0                     # 50 - 0 and 50 - 0 clip to +25 -> 1
    snap = build_snapshot(view)
    wide = QConfig(margin_clip=5)
    assert dict(zip(features.NAMES, scalars(view, cfg=wide)))["d7"] == 1.0
    assert snap.d7 == 50.0                                       # the raw value is kept for the label


def test_the_state_is_one_hot_in_the_last_five_entries():
    view = scene(world_with(CLOSE), CLOSE)
    for state in State:
        out = scalars(view, state)
        assert out[17:].sum() == 1.0 and out[17 + int(state)] == 1.0 and np.isfinite(out).all()


def test_flags_and_the_belief_spread():
    seen = scene(world_with(CLOSE, opp_trail=[(25, 25), (25, 24), (25, 23)], opp_land=OPP_LAND), CLOSE)
    f = dict(zip(features.NAMES, scalars(seen)))
    assert f["enemy_visible"] == 1.0 and f["enemy_trail_known"] == 1.0 and f["belief_spread"] == pytest.approx(
        np.log1p(1) / np.log(2500))
    wide = np.ones((50, 50))
    unseen = scene(world_with(CLOSE), CLOSE, belief_of(residual=wide))
    g = dict(zip(features.NAMES, scalars(unseen)))
    # every cell is equally likely, except the 121 I can see (the map knows it is not there)
    assert g["enemy_visible"] == 0.0 and g["belief_spread"] == pytest.approx(np.log1p(2500 - 121) / np.log(2500))


def test_turns_since_sighting_is_capped():
    view = scene(world_with(CLOSE), CLOSE, opp_last_seen=(flat(25, 40), 10), turn=30)
    assert dict(zip(features.NAMES, scalars(view)))["turns_since_sighting"] == pytest.approx(20 / 100)
    far = scene(world_with(CLOSE), CLOSE, opp_last_seen=(flat(25, 40), 10), turn=900)
    assert dict(zip(features.NAMES, scalars(far)))["turns_since_sighting"] == 1.0


# ---- the board can be rotated: the scalars do not change -------------------------------------------------------


def test_rotating_the_board_leaves_every_scalar_unchanged():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21), (24, 21), (23, 21)], opp_trail=[(25, 26), (25, 25), (24, 25)],
                       opp_land=OPP_LAND)
    base = scalars(scene(world, CLOSE))
    for k, flip in ((1, False), (2, False), (3, False), (0, True), (1, True)):
        turned, rules = rotate_world(world, CLOSE, k, flip)
        # the belief is the enemy exactly where it is, so scene() rotates it with the world
        assert np.allclose(scalars(scene(turned, rules)), base, atol=1e-6), (k, flip)


# ---- the grid ----------------------------------------------------------------------------------------------------


def channel(g, name):
    return g[("board", "visible", "my_land", "my_trail", "opp_land", "opp_trail", "opp_pos").index(name)]


def test_the_grid_has_seven_channels_of_21_by_21_centred_on_me():
    view = scene(world_with(CLOSE), CLOSE)
    g = grid.build_grid(view.obs)
    assert g.shape == (7, 21, 21) and g.dtype == np.uint8
    assert channel(g, "board").all()                              # far from every edge
    assert channel(g, "visible").sum() == 121 and channel(g, "visible")[5:16, 5:16].all()
    assert channel(g, "my_land")[10, 10] == 1                     # I stand on my land at the centre
    assert channel(g, "my_land").sum() == 9 and channel(g, "opp_land").sum() == 0     # its land is 13 cells away


def test_the_whole_trail_is_in_the_crop_even_outside_the_view():
    trail = [(25, 20 + k) for k in range(9)]                      # ends at (25, 28): 10 cells east of the start of the crop
    view = scene(world_with(CLOSE, my_trail=trail), CLOSE)
    g = grid.build_grid(view.obs)
    assert channel(g, "my_trail").sum() == 9
    assert channel(g, "visible")[10, 10 + 6] == 0                 # 6 cells east is outside the 11 x 11 view
    assert channel(g, "my_trail")[10, 10 - 0] == 1                # my head, at the centre


def test_enemy_cells_show_only_where_visible():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21)], opp=(25, 24), opp_land=OPP_LAND + [(25, 23)],
                       opp_trail=[(25, 24)])
    view = scene(world, CLOSE)
    g = grid.build_grid(view.obs)
    my_y, my_x = divmod(view.obs.my_pos, 50)
    assert channel(g, "opp_pos")[10 + 25 - my_y, 10 + 24 - my_x] == 1 and channel(g, "opp_pos").sum() == 1
    assert channel(g, "opp_trail")[10 + 25 - my_y, 10 + 24 - my_x] == 1
    assert channel(g, "opp_land")[10 + 25 - my_y, 10 + 23 - my_x] == 1
    assert channel(g, "opp_land").sum() == 1                      # the rest of its land is beyond the view


def test_cells_off_the_board_are_zero_in_every_channel():
    world = world_with(MatchRules.from_respawns(flat(1, 1), flat(25, 45)), me=(1, 1))
    rules = MatchRules.from_respawns(flat(1, 1), flat(25, 45))
    view = scene(world, rules)
    g = grid.build_grid(view.obs)
    board = channel(g, "board")
    assert board[:9].sum() == 0 and board[:, :9].sum() == 0       # rows/cols above and left of the board
    assert (g * (1 - board)).sum() == 0                           # nothing anywhere off the board
    assert board.sum() == 12 * 12


def test_rotating_the_board_rotates_the_grid_the_same_way():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21), (24, 21)], opp_trail=[(26, 24), (26, 23)], opp_land=OPP_LAND + [(26, 22)])
    base = grid.build_grid(scene(world, CLOSE).obs)
    for k, flip in ((1, False), (2, False), (3, False), (0, True), (3, True)):
        turned, rules = rotate_world(world, CLOSE, k, flip)
        assert np.array_equal(grid.build_grid(scene(turned, rules).obs), grid.rotate(base, k, flip)), (k, flip)
