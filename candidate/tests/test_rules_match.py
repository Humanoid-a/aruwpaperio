import pytest

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, STARTING_TERRITORY_COUNT
from candidate.agent.data.cells import FREE, MY_TERRITORY, OPP_TERRITORY
from candidate.agent.data.types import Side
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat, to_yx
from candidate.agent.rules.match import MatchRules, RoleRules, starting_cells


def test_starting_territory_is_the_3x3_square_around_the_respawn():
    cells = starting_cells(flat(25, 4))
    assert len(cells) == STARTING_TERRITORY_COUNT
    assert set(map(to_yx, cells)) == {(y, x) for y in (24, 25, 26) for x in (3, 4, 5)}
    assert cells == tuple(sorted(cells))  # row-major, like the framework's list


def test_starting_territory_is_clipped_at_the_board_edge():
    assert set(map(to_yx, starting_cells(flat(0, 0)))) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert len(starting_cells(flat(BOARD_HEIGHT - 1, 10))) == 6


def test_role_rules_carry_a_matching_bitboard_and_a_read_only_copy():
    role = RoleRules.build(flat(25, 4))
    assert bitboard.to_cells(role.start_bb) == sorted(role.start_cells)
    with pytest.raises(ValueError):
        role.start_bb[0] = 1


def test_alternate_respawn_is_the_nearest_start_cell_with_ties_to_the_north():
    role = RoleRules.build(flat(25, 4))
    assert role.alt_respawn == flat(24, 4)  # distance 1, smallest (y, x) among the four neighbours
    corner = RoleRules.build(flat(0, 0))
    assert corner.alt_respawn == flat(0, 1)  # (0, 1) beats (1, 0) on y


def test_alternate_respawn_falls_back_to_the_board_when_there_is_no_other_start_cell():
    role = RoleRules.build(flat(25, 4), size=1)
    assert role.start_cells == (flat(25, 4),)
    assert role.alt_respawn == flat(24, 4)


def test_default_match_mirrors_the_opponent_across_the_board():
    rules = MatchRules.mirrored(flat(25, 4))
    assert rules.opp.respawn == flat(25, BOARD_WIDTH - 1 - 4) == flat(25, 45)
    assert rules.role(Side.ME) is rules.me and rules.role(Side.OPP) is rules.opp


def test_overlapping_starting_territories_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        MatchRules.from_respawns(flat(25, 20), flat(25, 22))
    MatchRules.from_respawns(flat(25, 20), flat(25, 23))  # touching but disjoint is fine


def test_initial_world_places_both_agents_on_their_own_start_land():
    rules = MatchRules.from_respawns(flat(25, 4), flat(25, 45))
    world = rules.initial_world()
    world.validate()
    assert (world.my_pos, world.opp_pos) == (flat(25, 4), flat(25, 45))
    assert world.my_count == world.opp_count == STARTING_TERRITORY_COUNT
    assert world.territory[25, 4] == MY_TERRITORY and world.territory[25, 45] == OPP_TERRITORY
    assert world.territory[25, 25] == FREE
    assert not world.trail.any()
