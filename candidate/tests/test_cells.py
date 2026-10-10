import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.cells import (
    FREE,
    MY_TERRITORY,
    MY_TRAIL,
    OPP_TERRITORY,
    OPP_TRAIL,
    UNKNOWN,
    combine,
    layer_bitboards,
    layer_problems,
)
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat

SHAPE = (BOARD_HEIGHT, BOARD_WIDTH)


def layers():
    return np.zeros(SHAPE, dtype=np.int8), np.zeros(SHAPE, dtype=np.int8)


def test_codes_match_the_spec_numbering():
    assert (UNKNOWN, FREE, MY_TERRITORY, MY_TRAIL, OPP_TERRITORY, OPP_TRAIL) == (-1, 0, 1, 2, 3, 4)


def test_combine_puts_the_trail_over_the_territory():
    territory, trail = layers()
    territory[5, 5] = MY_TERRITORY
    territory[6, 6] = OPP_TERRITORY
    trail[5, 5] = OPP_TRAIL  # opponent trail lying on my land
    trail[7, 7] = MY_TRAIL
    cells = combine(territory, trail)
    assert cells.dtype == np.int8
    assert cells[5, 5] == OPP_TRAIL
    assert cells[6, 6] == OPP_TERRITORY
    assert cells[7, 7] == MY_TRAIL
    assert cells[0, 0] == FREE


def test_combine_is_lossy_but_the_layers_are_not():
    territory, trail = layers()
    territory[5, 5] = MY_TERRITORY
    trail[5, 5] = OPP_TRAIL
    assert combine(territory, trail)[5, 5] == OPP_TRAIL  # the territory is gone from this view
    assert territory[5, 5] == MY_TERRITORY               # but still in the layer


def test_combine_keeps_unknown_cells_unknown():
    territory = np.full(SHAPE, UNKNOWN, dtype=np.int8)
    trail = np.full(SHAPE, UNKNOWN, dtype=np.int8)
    assert (combine(territory, trail) == UNKNOWN).all()


def test_layer_bitboards_place_each_layer_and_skip_unknown():
    territory, trail = layers()
    territory[1, 2] = MY_TERRITORY
    territory[3, 4] = OPP_TERRITORY
    trail[1, 2] = OPP_TRAIL
    trail[5, 6] = MY_TRAIL
    territory[9, 9] = UNKNOWN
    trail[9, 9] = UNKNOWN
    bbs = layer_bitboards(territory, trail)
    assert bitboard.to_cells(bbs.my_territory) == [flat(1, 2)]
    assert bitboard.to_cells(bbs.opp_territory) == [flat(3, 4)]
    assert bitboard.to_cells(bbs.my_trail) == [flat(5, 6)]
    assert bitboard.to_cells(bbs.opp_trail) == [flat(1, 2)]


def test_layer_problems_accepts_sound_layers():
    territory, trail = layers()
    territory[0, 0], trail[1, 1] = MY_TERRITORY, OPP_TRAIL
    assert layer_problems(territory, trail, SHAPE, allow_unknown=False) == []


def test_layer_problems_rejects_codes_in_the_wrong_layer():
    territory, trail = layers()
    territory[0, 0] = MY_TRAIL  # a trail code in the territory layer
    trail[1, 1] = OPP_TERRITORY  # a territory code in the trail layer
    problems = layer_problems(territory, trail, SHAPE, allow_unknown=False)
    assert len(problems) == 2


def test_layer_problems_allows_unknown_only_when_asked():
    territory, trail = layers()
    territory[0, 0] = UNKNOWN
    assert layer_problems(territory, trail, SHAPE, allow_unknown=False)
    assert layer_problems(territory, trail, SHAPE, allow_unknown=True) == []


def test_layer_problems_rejects_wrong_dtype_and_shape():
    territory, trail = layers()
    assert layer_problems(territory.astype(np.int64), trail, SHAPE, allow_unknown=False)
    assert layer_problems(territory, trail[:-1], SHAPE, allow_unknown=False)
