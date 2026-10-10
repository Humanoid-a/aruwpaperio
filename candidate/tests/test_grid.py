import pytest

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.data.types import Direction
from candidate.agent.geometry.grid import FLAT_OFFSETS, flat, manhattan, neighbors, step, to_yx


def test_flat_index_is_row_major_with_width_stride():
    assert flat(0, 0) == 0
    assert flat(2, 3) == 2 * BOARD_WIDTH + 3
    assert flat(BOARD_HEIGHT - 1, BOARD_WIDTH - 1) == CELL_COUNT - 1


def test_flat_and_to_yx_round_trip_on_every_cell():
    for c in range(CELL_COUNT):
        assert flat(*to_yx(c)) == c


@pytest.mark.parametrize("y,x", [(-1, 0), (0, -1), (BOARD_HEIGHT, 0), (0, BOARD_WIDTH)])
def test_flat_rejects_off_board_cells(y, x):
    with pytest.raises(ValueError):
        flat(y, x)


@pytest.mark.parametrize("c", [-1, CELL_COUNT])
def test_to_yx_rejects_out_of_range_indices(c):
    with pytest.raises(ValueError):
        to_yx(c)


def test_offsets_are_indexed_by_direction_in_n_e_s_w_order():
    assert FLAT_OFFSETS[Direction.NORTH] == -BOARD_WIDTH
    assert FLAT_OFFSETS[Direction.EAST] == +1
    assert FLAT_OFFSETS[Direction.SOUTH] == +BOARD_WIDTH
    assert FLAT_OFFSETS[Direction.WEST] == -1


def test_step_moves_one_cell_in_the_interior():
    c = flat(10, 10)
    assert step(c, Direction.NORTH) == flat(9, 10)
    assert step(c, Direction.EAST) == flat(10, 11)
    assert step(c, Direction.SOUTH) == flat(11, 10)
    assert step(c, Direction.WEST) == flat(10, 9)


def test_step_does_not_wrap_across_row_ends():
    last_column = flat(3, BOARD_WIDTH - 1)
    first_column = flat(4, 0)
    # The naive flat-offset arithmetic would land on the neighbouring row.
    assert last_column + FLAT_OFFSETS[Direction.EAST] == first_column
    assert step(last_column, Direction.EAST) is None
    assert step(first_column, Direction.WEST) is None


def test_step_off_the_top_and_bottom_is_none():
    assert step(flat(0, 7), Direction.NORTH) is None
    assert step(flat(BOARD_HEIGHT - 1, 7), Direction.SOUTH) is None


def test_neighbors_in_direction_order_and_clipped_at_corners():
    assert neighbors(flat(10, 10)) == (flat(9, 10), flat(10, 11), flat(11, 10), flat(10, 9))
    assert neighbors(flat(0, 0)) == (flat(0, 1), flat(1, 0))
    assert len(neighbors(flat(BOARD_HEIGHT - 1, BOARD_WIDTH - 1))) == 2


def test_manhattan_distance():
    assert manhattan(flat(0, 0), flat(0, 0)) == 0
    assert manhattan(flat(2, 3), flat(5, 1)) == 5


def test_direction_opposite():
    assert Direction.NORTH.opposite is Direction.SOUTH
    assert Direction.EAST.opposite is Direction.WEST
    assert all(d.opposite.opposite is d for d in Direction)
