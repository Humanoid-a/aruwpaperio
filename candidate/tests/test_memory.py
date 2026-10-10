import dataclasses

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.cells import (
    FREE,
    MY_TERRITORY,
    MY_TRAIL,
    OPP_TERRITORY,
    OPP_TRAIL,
    UNKNOWN,
)
from candidate.agent.data.memory import NEVER_SEEN, MemoryMap
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.tests.builders import make_observation, square


def test_new_memory_knows_nothing():
    memory = MemoryMap.new()
    memory.validate()
    assert (memory.territory == UNKNOWN).all() and (memory.trail == UNKNOWN).all()
    assert (memory.last_seen == NEVER_SEEN).all()
    assert not memory.known_mask().any()
    assert bitboard.popcount(memory.known_bitboard()) == 0
    assert (memory.age(turn=10) == NEVER_SEEN).all()


def test_update_writes_both_layers_and_stamps_the_window_only():
    obs = make_observation(
        pos=(25, 4),
        turn=7,
        territory={c: MY_TERRITORY for c in square(25, 4)},
        trail={(24, 4): OPP_TRAIL},
        opp_at=(23, 5),
    )
    memory = MemoryMap.new()
    memory.update(obs)
    memory.validate()

    assert memory.territory[24, 4] == MY_TERRITORY and memory.trail[24, 4] == OPP_TRAIL
    assert memory.territory[25, 3] == MY_TERRITORY and memory.trail[25, 3] == FREE
    assert memory.agrees_with(obs)

    rows, cols = obs.window_slices
    inside = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=bool)
    inside[rows, cols] = True
    assert (memory.last_seen[inside] == 7).all()
    assert (memory.last_seen[~inside] == NEVER_SEEN).all()
    assert (memory.territory[~inside] == UNKNOWN).all()
    assert np.array_equal(memory.known_mask(), inside)
    assert bitboard.popcount(memory.known_bitboard()) == obs.window_territory.size


def test_a_later_observation_overwrites_the_overlap_and_keeps_older_cells():
    first = make_observation(pos=(25, 10), turn=1, territory={(25, 12): OPP_TERRITORY})
    second = make_observation(pos=(25, 14), turn=2, territory={(25, 12): MY_TERRITORY})
    memory = MemoryMap.new()
    memory.update(first)
    memory.update(second)

    # (25, 12) is in both windows: the newer sighting wins.
    assert memory.territory[25, 12] == MY_TERRITORY and memory.last_seen[25, 12] == 2
    # (25, 5) was only in the first window: kept, with its old stamp.
    assert memory.territory[25, 5] == FREE and memory.last_seen[25, 5] == 1
    # (25, 19) was only in the second window.
    assert memory.last_seen[25, 19] == 2
    assert memory.agrees_with(second)
    assert not memory.agrees_with(first)
    assert memory.age(turn=5)[25, 5] == 4 and memory.age(turn=5)[25, 19] == 3
    assert memory.age(turn=5)[0, 0] == NEVER_SEEN


def test_a_stale_cell_is_kept_even_if_the_world_has_since_changed():
    # Memory never guesses: it holds the last thing seen until the cell is seen again.
    memory = MemoryMap.new()
    memory.update(make_observation(pos=(25, 10), turn=1, territory={(25, 6): OPP_TERRITORY}))
    memory.update(make_observation(pos=(25, 30), turn=9))  # (25, 6) is now far out of view
    assert memory.territory[25, 6] == OPP_TERRITORY and memory.last_seen[25, 6] == 1


def test_clipped_corner_window_updates_cleanly():
    obs = make_observation(pos=(0, 0), respawn=(0, 0), turn=3, territory={(0, 0): MY_TERRITORY})
    memory = MemoryMap.new()
    memory.update(obs)
    memory.validate()
    assert memory.territory[0, 0] == MY_TERRITORY
    assert bitboard.popcount(memory.known_bitboard()) == 6 * 6


def test_layer_bitboards_hold_only_known_cells():
    memory = MemoryMap.new()
    memory.update(
        make_observation(
            pos=(25, 4),
            territory={c: MY_TERRITORY for c in square(25, 4)},
            trail={(24, 4): OPP_TRAIL},
            opp_at=(23, 4),
        )
    )
    bbs = memory.bitboards()
    assert bitboard.popcount(bbs.my_territory) == 9
    assert bitboard.to_cells(bbs.opp_trail) == [flat(24, 4)]
    assert bitboard.popcount(bbs.opp_territory) == 0 and bitboard.popcount(bbs.my_trail) == 0


def test_my_own_trail_in_view_is_remembered_as_my_trail():
    memory = MemoryMap.new()
    memory.update(make_observation(pos=(25, 7), my_trail=[(25, 6), (25, 7)]))
    assert memory.trail[25, 6] == MY_TRAIL and memory.trail[25, 7] == MY_TRAIL


def test_cells_view_hides_territory_under_a_trail_but_the_layers_keep_it():
    memory = MemoryMap.new()
    memory.update(make_observation(pos=(25, 4), territory={(24, 4): MY_TERRITORY}, trail={(24, 4): OPP_TRAIL}))
    assert memory.cells[24, 4] == OPP_TRAIL
    assert memory.territory[24, 4] == MY_TERRITORY
    assert memory.cells[0, 0] == UNKNOWN


def test_copy_is_independent():
    memory = MemoryMap.new()
    memory.update(make_observation(pos=(25, 4), turn=2))
    clone = memory.copy()
    clone.territory[0, 0] = MY_TERRITORY
    clone.last_seen[0, 0] = 99
    assert memory.territory[0, 0] == UNKNOWN and memory.last_seen[0, 0] == NEVER_SEEN


def test_problems_flags_unknown_that_disagrees_with_last_seen():
    memory = MemoryMap.new()
    memory.last_seen[3, 3] = 5  # stamped, yet the layers still say UNKNOWN
    assert any("last_seen" in p for p in memory.problems())
    memory = MemoryMap.new()
    memory.territory[3, 3] = FREE  # known territory with no stamp
    assert memory.problems()
    memory = MemoryMap.new()
    memory.last_seen = memory.last_seen.astype(np.int64)
    assert any("int32" in p for p in memory.problems())


def test_agrees_with_detects_tampering():
    obs = make_observation(pos=(25, 4), turn=4)
    memory = MemoryMap.new()
    memory.update(obs)
    assert memory.agrees_with(obs)
    memory.territory[25, 4] = OPP_TERRITORY
    assert not memory.agrees_with(obs)
    same_but_later = dataclasses.replace(obs, turn=5)
    assert not memory.agrees_with(same_but_later)
