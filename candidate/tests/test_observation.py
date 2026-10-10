import dataclasses

import numpy as np
import pytest

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, WINDOW_SIZE
from candidate.agent.data.cells import FREE, MY_TERRITORY, MY_TRAIL, OPP_TERRITORY, OPP_TRAIL
from candidate.agent.data.observation import Observation
from candidate.agent.data.types import MoveResult
from candidate.agent.geometry.grid import flat
from candidate.tests.builders import make_observation, square, window_bounds


def test_interior_window_is_full_size_and_centred_on_the_agent():
    obs = make_observation(pos=(25, 20))
    assert obs.window_territory.shape == (WINDOW_SIZE, WINDOW_SIZE)
    assert obs.window_origin == (20, 15)
    assert obs.covers(obs.my_pos)


@pytest.mark.parametrize(
    "pos,shape",
    [((0, 0), (6, 6)), ((0, 25), (6, 11)), ((49, 49), (6, 6)), ((25, 0), (11, 6)), ((3, 47), (9, 8))],
)
def test_windows_are_clipped_at_the_board_edge(pos, shape):
    obs = make_observation(pos=pos, respawn=pos)
    assert obs.window_territory.shape == shape
    assert obs.window_trail.shape == shape


def test_window_view_stacks_the_layers_with_the_trail_on_top():
    territory = {c: MY_TERRITORY for c in square(25, 4)}
    obs = make_observation(
        pos=(25, 4),
        territory=territory,
        trail={(24, 4): OPP_TRAIL},
        opp_at=(23, 4),
    )
    assert obs.window_cell(flat(24, 4)) == (MY_TERRITORY, OPP_TRAIL)  # both layers survive
    window = obs.window
    y0, x0 = obs.window_origin
    assert window[24 - y0, 4 - x0] == OPP_TRAIL
    assert window[26 - y0, 5 - x0] == MY_TERRITORY


def test_window_cell_and_covers_use_board_coordinates():
    obs = make_observation(pos=(25, 20))
    assert obs.covers(flat(20, 15)) and obs.covers(flat(30, 25))
    assert not obs.covers(flat(19, 20)) and not obs.covers(flat(25, 26))
    with pytest.raises(ValueError):
        obs.window_cell(flat(0, 0))


def test_window_slices_select_the_window_out_of_a_board_sized_array():
    obs = make_observation(pos=(2, 2), respawn=(2, 2))
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8)
    rows, cols = obs.window_slices
    assert board[rows, cols].shape == obs.window_territory.shape


def test_my_trail_must_end_at_the_agent():
    ok = make_observation(pos=(25, 7), my_trail=[(25, 6), (25, 7)])
    assert list(ok.my_trail) == [flat(25, 6), flat(25, 7)]
    with pytest.raises(ValueError, match="last cell"):
        make_observation(pos=(25, 7), my_trail=[(25, 7), (25, 6)])


def test_trail_cells_outside_the_window_are_allowed_and_not_drawn():
    # A long trail whose oldest cells are more than 5 columns behind the head.
    trail = [(25, x) for x in range(6, 20)]
    obs = make_observation(pos=(25, 19), my_trail=trail)
    assert len(obs.my_trail) == 14
    assert not obs.covers(flat(25, 6))
    assert obs.window_cell(flat(25, 19))[1] == MY_TRAIL


def test_window_trail_layer_must_agree_with_my_trail():
    good = make_observation(pos=(25, 7), my_trail=[(25, 6), (25, 7)])
    forged = good.window_trail.copy()
    y0, x0 = good.window_origin
    forged[25 - y0, 8 - x0] = MY_TRAIL  # a visible own-trail cell that is not in my_trail
    with pytest.raises(ValueError, match="do not match my_trail"):
        dataclasses.replace(good, window_trail=forged)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"turn": -1}, "turn"),
        ({"turns_left": -1}, "turns_left"),
        ({"my_count": -1}, "counts"),
        ({"my_pos": BOARD_HEIGHT * BOARD_WIDTH}, "off the board"),
        ({"opp_seen_at": 0}, "opp_seen_at is outside the window"),
        ({"window_origin": (45, 0)}, "inside the board"),
    ],
)
def test_construction_rejects_inconsistent_fields(change, message):
    base = make_observation(pos=(25, 20))
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(base, **change)


def test_agent_must_lie_inside_its_own_window():
    base = make_observation(pos=(25, 20))
    with pytest.raises(ValueError, match="my_pos is outside"):
        dataclasses.replace(base, my_pos=flat(0, 0))


def test_opponent_cannot_be_seen_on_the_agents_own_cell():
    base = make_observation(pos=(25, 20))
    with pytest.raises(ValueError, match="equals my_pos"):
        dataclasses.replace(base, opp_seen_at=base.my_pos)


def test_layers_must_use_their_own_codes():
    base = make_observation(pos=(25, 20))
    bad = base.window_territory.copy()
    bad[0, 0] = MY_TRAIL
    with pytest.raises(ValueError, match="invalid codes"):
        dataclasses.replace(base, window_territory=bad)


def test_oversized_window_is_rejected():
    base = make_observation(pos=(25, 20))
    big = np.zeros((WINDOW_SIZE + 2, WINDOW_SIZE), dtype=np.int8)
    with pytest.raises(ValueError, match="expected at most"):
        dataclasses.replace(base, window_territory=big, window_trail=big.copy(), window_origin=(10, 10))


def test_arrays_are_read_only_after_construction():
    obs = make_observation(pos=(25, 7), my_trail=[(25, 6), (25, 7)])
    for array in (obs.window_territory, obs.window_trail, obs.my_trail):
        with pytest.raises(ValueError):
            array[...] = 0


def test_observation_is_frozen_and_carries_the_last_move_result():
    obs = make_observation(last_result=MoveResult.CAPTURED)
    assert obs.last_result is MoveResult.CAPTURED
    assert make_observation().last_result is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.turn = 5


def test_opponent_territory_and_free_cells_round_trip_through_the_builder():
    obs = make_observation(
        pos=(25, 20),
        territory={(24, 21): OPP_TERRITORY, (26, 19): MY_TERRITORY},
    )
    assert obs.window_cell(flat(24, 21)) == (OPP_TERRITORY, FREE)
    assert obs.window_cell(flat(26, 19)) == (MY_TERRITORY, FREE)
    assert obs.window_cell(flat(25, 22)) == (FREE, FREE)
    y0, x0, h, w = window_bounds(25, 20)
    assert obs.window_origin == (y0, x0) and obs.window_territory.shape == (h, w)
