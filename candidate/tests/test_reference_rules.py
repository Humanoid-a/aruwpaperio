"""Hand-built move scenarios. Every one is run through the reference rules,
the fast bitboard rules, and the framework's real MoveResolver, which must
all agree exactly; the assertions then check that the agreed answer means
what the rules say."""

import numpy as np
import pytest

from candidate.agent.data.cells import FREE, MY_TERRITORY, MY_TRAIL, OPP_TERRITORY, OPP_TRAIL
from candidate.agent.data.types import Direction, MoveResult, Side
from candidate.agent.data.world import WorldState
from candidate.agent.geometry.grid import flat
from candidate.agent.rules.match import MatchRules
from candidate.agent.rules.reference import Outcome, apply_move
from candidate.agent.rules.transition import BitWorld, step
from candidate.tests.builders import put_agent, put_territory, put_trail, square
from candidate.tests.rules_oracle import real_apply

RULES = MatchRules.from_respawns(flat(25, 4), flat(25, 45))
N, E, S, W = Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST
ME, OPP = Side.ME, Side.OPP


def agree(world: WorldState, side: Side, direction: Direction, rules: MatchRules = RULES) -> Outcome:
    """Run the move through all three implementations, require identical
    outcomes, and return the reference outcome for further assertions."""
    world.validate()
    before = world.copy()
    ref = apply_move(world, rules, side, direction)
    real = real_apply(world, rules, side, direction)
    assert world == before, "the input world must not be modified"
    assert ref.world is not world
    assert ref.world == real.world
    assert (ref.result, ref.killed) == (real.result, real.killed)
    fast = step(BitWorld.from_world(world), rules, side, [direction])
    assert fast.world.matches(0, ref.world)
    assert (int(fast.result[0]), bool(fast.killed[0])) == (ref.result.value, ref.killed)
    assert fast.world.problems() == []
    ref.world.validate()
    return ref


def loop_world() -> WorldState:
    """I hold a wall of land (column 5, rows 20..30) plus my start square. My
    trail has left the wall at the top, gone round two columns of land and
    nine rows, and its head is at (30, 6), one step from the wall's bottom."""
    world = RULES.initial_world()
    put_territory(world, ME, [(y, 5) for y in range(20, 31)])
    path = [(20, 6), (20, 7), (20, 8)] + [(y, 8) for y in range(21, 30)] + [(30, 8), (30, 7), (30, 6)]
    put_trail(world, ME, path)
    return world


INTERIOR = {(y, x) for y in range(21, 30) for x in (6, 7)}


# ---- ordinary movement --------------------------------------------------


def test_moving_inside_my_land_leaves_no_trail():
    out = agree(RULES.initial_world(), ME, E)
    assert out.result == MoveResult.NORMAL and not out.killed
    assert out.world.my_pos == flat(25, 5) and out.world.my_trail == []
    assert not out.world.trail.any()


def test_leaving_my_land_starts_a_trail_on_the_new_cell():
    world = RULES.initial_world()
    put_agent(world, ME, (25, 5))
    out = agree(world, ME, E)
    assert out.result == MoveResult.NORMAL
    assert out.world.my_trail == [flat(25, 6)] and out.world.my_pos == flat(25, 6)
    assert out.world.trail[25, 6] == MY_TRAIL and out.world.my_count == 9


def test_the_opponent_moves_by_the_same_rules():
    world = RULES.initial_world()
    put_agent(world, OPP, (25, 44))
    out = agree(world, OPP, W)
    assert out.world.opp_trail == [flat(25, 43)] and out.world.trail[25, 43] == OPP_TRAIL


def test_a_trail_over_enemy_land_keeps_both_layers():
    world = RULES.initial_world()
    put_trail(world, ME, [(25, 42), (25, 43)])
    out = agree(world, ME, E)  # onto (25, 44), the opponent's start
    assert out.world.territory[25, 44] == OPP_TERRITORY and out.world.trail[25, 44] == MY_TRAIL
    assert out.world.cells[25, 44] == MY_TRAIL
    assert out.world.opp_count == 9 and out.world.my_trail[-1] == flat(25, 44)


# ---- invalid moves ------------------------------------------------------


@pytest.mark.parametrize("direction", [N, W])
def test_stepping_off_the_board_is_invalid_and_changes_nothing(direction):
    rules = MatchRules.from_respawns(flat(0, 0), flat(25, 45))
    world = rules.initial_world()
    out = agree(world, ME, direction, rules)
    assert out.result == MoveResult.INVALID and not out.killed
    assert out.world == world


def test_stepping_onto_the_other_agent_is_invalid():
    world = RULES.initial_world()
    put_agent(world, OPP, (25, 5))
    out = agree(world, ME, E)
    assert out.result == MoveResult.INVALID and out.world == world


# ---- death --------------------------------------------------------------


def exposed_world() -> WorldState:
    world = RULES.initial_world()
    put_territory(world, ME, square(11, 11))  # far-away land that dying will cost me
    put_trail(world, ME, [(25, 6), (25, 7), (26, 7), (26, 6)])
    return world


def test_stepping_on_my_own_trail_kills_me_and_resets_me():
    world = exposed_world()
    assert world.my_count == 18
    out = agree(world, ME, N)  # onto (25, 6)
    assert out.result == MoveResult.DIED and not out.killed
    w = out.world
    assert w.my_pos == flat(25, 4) and w.my_trail == []
    assert not (w.trail == MY_TRAIL).any()
    assert (w.territory == MY_TERRITORY).sum() == w.my_count == 9
    assert w.territory[11, 11] == FREE  # the far land is gone
    assert w.opp_count == 9


def test_dying_while_the_opponent_stands_on_my_respawn_cell_displaces_me():
    world = exposed_world()
    put_trail(world, OPP, [(25, 4)])  # the opponent is standing in my start
    out = agree(world, ME, N)
    assert out.result == MoveResult.DIED
    assert out.world.my_pos == RULES.me.alt_respawn == flat(24, 4)
    assert out.world.trail[25, 4] == OPP_TRAIL  # their trail is not touched


def test_dying_takes_back_my_start_land_from_whoever_holds_it():
    world = exposed_world()
    world.territory[25, 3] = OPP_TERRITORY  # a cell of my start that the opponent captured
    put_territory(world, OPP, [])
    out = agree(world, ME, N)
    assert out.world.territory[25, 3] == MY_TERRITORY


# ---- capture ------------------------------------------------------------


def test_closing_a_loop_captures_the_trail_and_the_land_inside():
    world = loop_world()
    before_mine = world.my_count
    out = agree(world, ME, W)  # onto the wall at (30, 5)
    assert out.result == MoveResult.CAPTURED and not out.killed
    w = out.world
    assert w.my_trail == [] and not (w.trail == MY_TRAIL).any()
    assert w.my_pos == flat(30, 5)
    assert all(w.territory[c] == MY_TERRITORY for c in INTERIOR)
    assert w.my_count == before_mine + 15 + len(INTERIOR)


def test_capturing_flips_enemy_land_but_leaves_enemy_trail_in_place():
    world = loop_world()
    put_territory(world, OPP, [(25, 6), (25, 7)])  # enemy land inside the loop
    put_trail(world, OPP, [(23, 7)])               # and an enemy trail and agent
    out = agree(world, ME, W)
    w = out.world
    assert out.result == MoveResult.CAPTURED
    assert w.territory[25, 6] == MY_TERRITORY and w.territory[25, 7] == MY_TERRITORY
    assert w.opp_count == 9
    assert w.trail[23, 7] == OPP_TRAIL and w.territory[23, 7] == MY_TERRITORY  # trail now on my land
    assert w.opp_trail == [flat(23, 7)] and w.opp_pos == flat(23, 7)


def test_my_trail_crossing_enemy_land_flips_it_when_the_loop_closes():
    world = loop_world()
    world.territory[20, 7] = OPP_TERRITORY  # under my own trail
    put_territory(world, OPP, [])
    out = agree(world, ME, W)
    assert out.world.territory[20, 7] == MY_TERRITORY and out.world.opp_count == 9


def test_a_region_that_was_already_enclosed_is_not_captured():
    world = loop_world()
    ring = set(square(40, 40)) - {(40, 40)}
    put_territory(world, ME, ring)
    put_territory(world, OPP, [(40, 40)])  # an enemy island inside my ring
    out = agree(world, ME, W)
    assert out.result == MoveResult.CAPTURED
    assert out.world.territory[40, 40] == OPP_TERRITORY
    assert all(out.world.territory[c] == MY_TERRITORY for c in INTERIOR)


def test_stepping_onto_free_land_instead_of_closing_the_loop_captures_nothing():
    world = loop_world()
    out = agree(world, ME, N)  # (29, 6): free land beside the head, not the wall
    assert out.result == MoveResult.NORMAL
    assert out.world.my_trail[-1] == flat(29, 6) and len(out.world.my_trail) == 16
    assert (out.world.territory[21:30, 6:8] == FREE).all()


# ---- cutting the opponent's trail --------------------------------------


def test_stepping_on_the_opponents_trail_sends_them_home_and_extends_my_trail():
    world = RULES.initial_world()
    put_trail(world, ME, [(25, x) for x in range(6, 12)])
    put_trail(world, OPP, [(25, 12), (25, 13)])
    put_territory(world, OPP, [(10, 10)])  # extra enemy land that dies with them
    out = agree(world, ME, E)
    w = out.world
    assert out.result == MoveResult.NORMAL and out.killed
    assert w.opp_pos == flat(25, 45) and w.opp_trail == [] and not (w.trail == OPP_TRAIL).any()
    assert (w.territory == OPP_TERRITORY).sum() == w.opp_count == 9
    assert w.my_trail[-1] == flat(25, 12) and w.trail[25, 12] == MY_TRAIL and len(w.my_trail) == 7


def test_cutting_a_trail_that_lies_on_my_own_land_leaves_my_trail_alone():
    world = RULES.initial_world()  # I am at home with no trail
    put_trail(world, OPP, [(24, 4), (23, 4)])  # an intruder's trail across my land
    out = agree(world, ME, N)
    assert out.result == MoveResult.NORMAL and out.killed
    assert out.world.my_trail == [] and out.world.my_pos == flat(24, 4)
    assert out.world.opp_pos == flat(25, 45) and out.world.opp_trail == []


def test_closing_a_loop_and_cutting_the_trail_on_the_same_step_does_both():
    world = loop_world()
    put_trail(world, OPP, [(30, 5), (30, 4)])  # an intruder crosses the wall's bottom cell
    out = agree(world, ME, W)
    w = out.world
    assert out.result == MoveResult.CAPTURED and out.killed
    assert w.opp_pos == flat(25, 45) and w.opp_trail == [] and w.my_trail == []
    assert all(w.territory[c] == MY_TERRITORY for c in INTERIOR)


def test_the_victim_gets_its_start_land_back_even_where_i_just_held_it():
    world = RULES.initial_world()
    put_territory(world, ME, [(25, 45), (25, 46)])           # land of theirs that I had captured
    put_trail(world, OPP, [(25, 45), (25, 46), (25, 47)])    # their trail runs back across it
    put_agent(world, ME, (25, 44))
    out = agree(world, ME, E)
    w = out.world
    assert out.killed and w.territory[25, 45] == OPP_TERRITORY and w.territory[25, 46] == OPP_TERRITORY
    assert w.my_count == 9 and w.opp_count == 9


def test_a_victim_whose_respawn_cell_is_under_me_is_displaced():
    world = RULES.initial_world()
    put_territory(world, ME, [(25, 45), (25, 46)])
    put_trail(world, OPP, [(25, 45), (25, 46), (25, 47)])
    put_agent(world, ME, (25, 44))
    out = agree(world, ME, E)  # I step onto (25, 45), their respawn cell
    assert out.killed and out.world.my_pos == flat(25, 45)
    assert out.world.opp_pos == RULES.opp.alt_respawn == flat(24, 45)


# ---- both sides ---------------------------------------------------------


@pytest.mark.parametrize("side", [ME, OPP])
@pytest.mark.parametrize("direction", list(Direction))
def test_every_first_move_from_the_starting_position_agrees(side, direction):
    agree(RULES.initial_world(), side, direction)


def test_the_opponent_can_die_capture_and_cut_just_like_me():
    # Mirror of the loop scenario with roles swapped: the opponent closes a loop.
    world = RULES.initial_world()
    put_territory(world, OPP, [(y, 40) for y in range(20, 31)])
    path = [(20, 39), (20, 38), (20, 37)] + [(y, 37) for y in range(21, 30)] + [(30, 37), (30, 38), (30, 39)]
    put_trail(world, OPP, path)
    out = agree(world, OPP, E)
    assert out.result == MoveResult.CAPTURED and out.world.opp_count > world.opp_count + 15
    assert np.all(out.world.territory[21:30, 38:40] == OPP_TERRITORY)
