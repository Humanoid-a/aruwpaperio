import numpy as np
import pytest

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat

RNG = np.random.default_rng(12345)


def random_mask(*batch: int, density: float = 0.3) -> np.ndarray:
    return RNG.random(batch + (BOARD_HEIGHT, BOARD_WIDTH)) < density


# ---- layout -------------------------------------------------------------


def test_bit_x_of_word_y_is_cell_y_x():
    bb = bitboard.from_cells([flat(2, 3)])
    assert bb.dtype == np.int64 and bb.shape == (BOARD_HEIGHT,)
    assert bb[2] == 1 << 3
    assert np.count_nonzero(bb) == 1


def test_last_column_uses_bit_49_and_stays_non_negative():
    bb = bitboard.from_cells([flat(0, BOARD_WIDTH - 1)])
    assert bb[0] == 1 << (BOARD_WIDTH - 1)
    assert bb[0] > 0


def test_flat_index_and_bit_position_agree_for_every_cell():
    for c in range(0, CELL_COUNT, 37):
        y, x = divmod(c, BOARD_WIDTH)
        bb = bitboard.from_cells([c])
        assert int(bb[y]) == 1 << x
        assert bitboard.has_cell(bb, c)


def test_empty_and_full():
    assert bitboard.popcount(bitboard.empty()) == 0
    assert bitboard.popcount(bitboard.full()) == CELL_COUNT
    assert bitboard.empty(4).shape == (4, BOARD_HEIGHT)
    assert bitboard.full(4, 2).shape == (4, 2, BOARD_HEIGHT)
    assert bitboard.is_valid(bitboard.full())


# ---- conversion ---------------------------------------------------------


def test_mask_round_trip_single_and_batch():
    single = random_mask()
    assert np.array_equal(bitboard.to_mask(bitboard.from_mask(single)), single)
    batch = random_mask(5)
    packed = bitboard.from_mask(batch)
    assert packed.shape == (5, BOARD_HEIGHT)
    assert np.array_equal(bitboard.to_mask(packed), batch)


def test_cells_round_trip_is_sorted_and_deduplicated():
    cells = [flat(30, 2), flat(0, 0), flat(30, 2), flat(49, 49), flat(10, 49)]
    assert bitboard.to_cells(bitboard.from_cells(cells)) == sorted(set(cells))


def test_from_mask_rejects_wrong_dtype_and_shape():
    with pytest.raises(TypeError):
        bitboard.from_mask(np.ones((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8))
    with pytest.raises(ValueError):
        bitboard.from_mask(np.ones((BOARD_HEIGHT, BOARD_WIDTH - 1), dtype=bool))


def test_to_mask_rejects_non_int64_boards():
    with pytest.raises(TypeError):
        bitboard.to_mask(np.zeros(BOARD_HEIGHT, dtype=np.uint64))


def test_to_cells_rejects_a_batch():
    with pytest.raises(ValueError):
        bitboard.to_cells(bitboard.empty(2))


# ---- single-cell operations --------------------------------------------


def test_set_clear_has_work_in_place():
    bb = bitboard.empty()
    bitboard.set_cell(bb, flat(7, 7))
    bitboard.set_cell(bb, flat(7, 8))
    assert bitboard.has_cell(bb, flat(7, 7)) and bitboard.has_cell(bb, flat(7, 8))
    assert not bitboard.has_cell(bb, flat(7, 6))
    bitboard.clear_cell(bb, flat(7, 7))
    assert not bitboard.has_cell(bb, flat(7, 7)) and bitboard.has_cell(bb, flat(7, 8))
    bitboard.clear_cell(bb, flat(7, 7))  # clearing an absent cell is a no-op
    assert bitboard.to_cells(bb) == [flat(7, 8)]


def test_set_cell_is_idempotent_and_does_not_touch_other_rows():
    bb = bitboard.empty()
    bitboard.set_cell(bb, flat(5, 5))
    bitboard.set_cell(bb, flat(5, 5))
    assert bitboard.popcount(bb) == 1


@pytest.mark.parametrize("c", [-1, CELL_COUNT])
def test_cell_operations_reject_out_of_range_indices(c):
    bb = bitboard.empty()
    for operation in (bitboard.has_cell, bitboard.set_cell, bitboard.clear_cell):
        with pytest.raises(ValueError):
            operation(bb, c)


def test_cell_operations_reject_batches():
    with pytest.raises(ValueError):
        bitboard.set_cell(bitboard.empty(3), 0)


# ---- complement / validity ---------------------------------------------


def test_complement_stays_on_the_board_unlike_bare_invert():
    bb = bitboard.from_mask(random_mask())
    assert bitboard.is_valid(bitboard.complement(bb))
    assert not bitboard.is_valid(~bb)  # sets bits 50..63 and goes negative
    assert np.array_equal(bitboard.complement(bitboard.complement(bb)), bb)
    assert np.array_equal(bitboard.complement(bitboard.empty()), bitboard.full())
    assert bitboard.popcount(bb) + bitboard.popcount(bitboard.complement(bb)) == CELL_COUNT


def test_is_valid_rejects_stray_bits_wrong_dtype_and_wrong_shape():
    stray = bitboard.empty()
    stray[3] = 1 << BOARD_WIDTH  # first bit past the board
    assert not bitboard.is_valid(stray)
    assert not bitboard.is_valid(np.zeros(BOARD_HEIGHT, dtype=np.uint64))
    assert not bitboard.is_valid(np.zeros(BOARD_HEIGHT - 1, dtype=np.int64))
    assert not bitboard.is_valid([0] * BOARD_HEIGHT)
    assert bitboard.is_valid(bitboard.empty(3))


# ---- popcount -----------------------------------------------------------


def test_popcount_matches_mask_sum_single_and_batch():
    mask = random_mask()
    assert bitboard.popcount(bitboard.from_mask(mask)) == int(mask.sum())
    batch = random_mask(6)
    counts = bitboard.popcount(bitboard.from_mask(batch))
    assert counts.shape == (6,)
    assert counts.tolist() == batch.reshape(6, -1).sum(axis=1).tolist()


def test_popcount_of_a_full_row_and_a_single_edge_cell():
    bb = bitboard.empty()
    bb[4] = bitboard.ROW_MASK
    assert bitboard.popcount(bb) == BOARD_WIDTH
    assert bitboard.popcount(bitboard.from_cells([flat(49, 49)])) == 1
