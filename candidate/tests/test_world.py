import numpy as np
import pytest

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.cells import FREE, MY_TERRITORY, OPP_TERRITORY, OPP_TRAIL
from candidate.agent.data.world import WorldState
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.tests.builders import make_world


def only_problem(world: WorldState, **kwargs) -> str:
    problems = world.problems(**kwargs)
    assert len(problems) == 1, problems
    return problems[0]


def test_empty_world_is_blank_and_valid():
    world = WorldState.empty(my_pos=flat(25, 4), opp_pos=flat(25, 45))
    assert world.territory.shape == world.trail.shape == (BOARD_HEIGHT, BOARD_WIDTH)
    assert world.territory.dtype == world.trail.dtype == np.int8
    assert not world.territory.any() and not world.trail.any()
    assert world.my_trail == [] and world.opp_trail == []
    world.validate()


def test_hand_built_midgame_world_is_valid():
    make_world().validate()


def test_territory_and_trail_overlap_is_representable_and_lossless():
    world = make_world()
    assert world.territory[24, 4] == MY_TERRITORY
    assert world.trail[24, 4] == OPP_TRAIL
    assert world.cells[24, 4] == OPP_TRAIL  # the single-code view hides the land under it
    bbs = world.bitboards()
    assert bitboard.has_cell(bbs.my_territory, flat(24, 4))
    assert bitboard.has_cell(bbs.opp_trail, flat(24, 4))
    assert not bitboard.has_cell(bbs.my_trail, flat(24, 4))


def test_bitboards_match_the_dense_layers():
    world = make_world()
    bbs = world.bitboards()
    assert bitboard.popcount(bbs.my_territory) == world.my_count == 9
    assert bitboard.popcount(bbs.opp_territory) == world.opp_count == 9
    assert bitboard.to_cells(bbs.my_trail) == sorted(world.my_trail)
    assert bitboard.to_cells(bbs.opp_trail) == sorted(world.opp_trail)
    assert all(bitboard.is_valid(b) for b in bbs)


def test_copy_is_deep_and_equal():
    world = make_world()
    clone = world.copy()
    assert clone == world and clone is not world
    clone.territory[0, 0] = MY_TERRITORY
    clone.my_trail.append(0)
    clone.my_pos = 0
    assert world.territory[0, 0] == FREE
    assert world.my_trail != clone.my_trail
    assert world != clone


def test_equality_is_by_value_and_world_is_unhashable():
    assert make_world() == make_world()
    other = make_world()
    other.opp_count += 1
    assert other != make_world()
    assert make_world() != "not a world"
    with pytest.raises(TypeError):
        hash(make_world())


# ---- invariants ---------------------------------------------------------


def test_trail_must_not_repeat_a_cell():
    world = make_world()
    world.my_trail = world.my_trail + [world.my_trail[0]]
    # Repeating a cell also breaks adjacency and the head position, so expect several problems.
    assert any("repeats" in p for p in world.problems())


def test_trail_cells_must_be_adjacent_in_walking_order():
    world = make_world()
    world.my_trail = [world.my_trail[1], world.my_trail[0], world.my_trail[2]]  # (25,7),(25,6),(26,7)
    assert any("not adjacent" in p for p in world.problems())


def test_trail_list_and_trail_layer_must_agree():
    world = make_world()
    world.trail[25, 6] = FREE  # drop a cell from the layer only
    assert any("does not match the cells marked my trail" in p for p in world.problems())
    world = make_world()
    world.trail[10, 10] = OPP_TRAIL  # extra marked cell with no list entry
    assert any("opp_trail does not match" in p for p in world.problems())


def test_trail_must_end_at_the_agent():
    world = make_world()
    world.my_pos = flat(25, 7)  # on the trail, but not its last cell
    assert "my_trail does not end at its agent's cell" in only_problem(world)


def test_a_trail_never_lies_on_its_own_owners_territory():
    world = make_world()
    world.territory[25, 6] = MY_TERRITORY
    world.my_count += 1
    assert any("own territory" in p for p in world.problems())


def test_agents_cannot_share_a_cell_or_leave_the_board():
    world = make_world()
    world.opp_pos = world.my_pos
    assert any("same cell" in p for p in world.problems())
    world = make_world()
    world.my_pos = -1
    assert any("off the board" in p for p in world.problems())


def test_counts_are_checked_against_the_board_unless_disabled():
    world = make_world()
    world.my_count = 10
    assert "my_count is 10" in only_problem(world)
    assert world.problems(check_counts=False) == []
    with pytest.raises(ValueError, match="my_count"):
        world.validate()
    world.validate(check_counts=False)


def test_structural_errors_are_reported_before_semantic_ones():
    world = make_world()
    world.territory = world.territory.astype(np.int64)
    problems = world.problems()
    assert len(problems) == 1 and "int8" in problems[0]
    with pytest.raises(ValueError, match="invalid WorldState"):
        world.validate()


def test_an_agent_with_no_trail_may_stand_on_enemy_land():
    # After the other player captures a region around it, an agent can be
    # on enemy territory with an empty trail. That is a legal state.
    world = WorldState.empty(my_pos=flat(10, 10), opp_pos=flat(40, 40))
    world.territory[10, 10] = OPP_TERRITORY
    world.opp_count = 1
    world.validate()
