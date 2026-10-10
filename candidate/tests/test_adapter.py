"""The adapter against the framework's real GameApi, on hand-built boards."""

import numpy as np
import pytest

from candidate.agent import adapter
from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.data.cells import FREE, MY_TERRITORY, MY_TRAIL, OPP_TERRITORY, OPP_TRAIL
from candidate.agent.data.types import Direction, MoveResult, Side
from candidate.agent.geometry.grid import flat
from candidate.agent.rules.match import MatchRules
from candidate.tests.builders import put_agent, put_territory, put_trail, square
from candidate.tests.live import LiveGame
from territorygame.api.direction import Direction as FrameworkDirection
from territorygame.api.grid_position import GridPosition
from territorygame.api.move_result import MoveResult as FrameworkMoveResult

ME, OPP = Side.ME, Side.OPP
RULES = MatchRules.from_respawns(flat(25, 4), flat(25, 45))


def observe(world, turn=0, last_result=None):
    return adapter.observation_from_game(LiveGame(world, RULES).me, turn, last_result)


# ---- vocabulary ---------------------------------------------------------


def test_every_direction_and_result_maps_one_to_one():
    assert {adapter.direction_to_framework(d) for d in Direction} == set(FrameworkDirection)
    for direction in Direction:
        assert adapter.direction_from_framework(adapter.direction_to_framework(direction)) == direction
    assert {adapter.result_from_framework(r) for r in FrameworkMoveResult} == set(MoveResult)
    assert adapter.result_from_framework(FrameworkMoveResult.MOVED) == MoveResult.NORMAL


def test_flat_cells_and_grid_positions_convert_both_ways():
    assert adapter.to_flat(GridPosition(x=3, y=2)) == flat(2, 3)
    assert adapter.from_flat(flat(2, 3)) == GridPosition(x=3, y=2)
    assert all(adapter.to_flat(adapter.from_flat(c)) == c for c in range(CELL_COUNT))


def test_a_move_in_our_north_goes_up_the_framework_board():
    live = LiveGame(RULES.initial_world(), RULES)
    live.me.reset_for_new_turn()
    assert adapter.send_move(live.me, Direction.NORTH) == MoveResult.NORMAL
    assert live.me.get_agent_position() == GridPosition(x=4, y=24)


def test_a_second_move_in_the_same_turn_is_refused_by_the_engine():
    live = LiveGame(RULES.initial_world(), RULES)
    live.me.reset_for_new_turn()
    assert adapter.send_move(live.me, Direction.EAST) == MoveResult.NORMAL
    assert adapter.send_move(live.me, Direction.EAST) == MoveResult.INVALID


def test_the_fallback_tries_moves_until_one_is_accepted_then_reports_none():
    rules = MatchRules.from_respawns(flat(0, 10), flat(25, 45))
    live = LiveGame(rules.initial_world(), rules)
    live.me.reset_for_new_turn()
    assert adapter.fallback_move(live.me) == MoveResult.NORMAL  # NORTH is off the board, the next one works
    assert adapter.fallback_move(live.me) is None                # the turn's move has been made


def test_check_board_accepts_the_real_board_and_rejects_another_size():
    adapter.check_board(LiveGame(RULES.initial_world(), RULES).me)

    class Small:
        def get_board_width(self):
            return 20

        def get_board_height(self):
            return 20

    with pytest.raises(adapter.AdapterError, match="20x20"):
        adapter.check_board(Small())


# ---- observations -------------------------------------------------------


def test_the_observation_matches_the_true_board_in_every_visible_cell():
    world = RULES.initial_world()
    put_territory(world, OPP, [(25, 6), (25, 7)])                       # enemy land beside my start
    put_trail(world, ME, [(25, 6), (25, 7), (26, 7)])                  # my trail across it
    put_trail(world, OPP, [(23, 5), (23, 4), (24, 4)])                  # theirs across my land
    obs = observe(world, turn=7, last_result=MoveResult.NORMAL)
    rows, cols = obs.window_slices
    assert obs.turn == 7 and obs.last_result == MoveResult.NORMAL
    assert np.array_equal(obs.window_territory, world.territory[rows, cols])
    differing = np.argwhere(obs.window_trail != world.trail[rows, cols])
    assert all(obs.hidden_mask[dy, dx] for dy, dx in differing)
    assert (obs.my_pos, list(obs.my_trail)) == (world.my_pos, world.my_trail)
    assert (obs.my_count, obs.opp_count) == (world.my_count, world.opp_count)
    assert obs.opp_seen_at == world.opp_pos == flat(24, 4)


def test_the_window_is_clipped_at_a_corner():
    rules = MatchRules.from_respawns(flat(0, 0), flat(25, 45))
    obs = adapter.observation_from_game(LiveGame(rules.initial_world(), rules).me, 0, None)
    assert obs.window_territory.shape == (6, 6) and obs.window_origin == (0, 0)


def test_the_opponent_on_my_land_is_assumed_to_stand_on_its_trail_and_is_hidden():
    world = RULES.initial_world()
    put_trail(world, OPP, [(24, 4)])  # the opponent has stepped onto my land
    obs = observe(world)
    assert obs.opp_seen_at == flat(24, 4) and flat(24, 4) in obs.trail_hidden
    assert obs.window_cell(flat(24, 4)) == (MY_TERRITORY, OPP_TRAIL)


def test_the_opponent_on_its_own_land_is_assumed_to_have_no_trail_there():
    world = RULES.initial_world()
    put_agent(world, OPP, (25, 8))
    world.territory[25, 8] = OPP_TERRITORY
    world.opp_count += 1
    obs = observe(world)
    assert obs.opp_seen_at == flat(25, 8)
    assert obs.window_cell(flat(25, 8)) == (OPP_TERRITORY, FREE) and flat(25, 8) in obs.trail_hidden


def test_my_own_cell_is_hidden_unless_it_is_on_my_trail():
    world = RULES.initial_world()
    obs = observe(world)
    assert obs.trail_hidden == (world.my_pos,)
    put_trail(world, ME, [(25, 6), (25, 7)])
    obs = observe(world)
    assert obs.trail_hidden == () and obs.window_cell(world.my_pos)[1] == MY_TRAIL


def test_a_respawn_onto_the_opponents_trail_is_reported_free_but_flagged_hidden():
    world = RULES.initial_world()
    put_trail(world, OPP, [(25, 4), (24, 4)])          # their trail crosses my respawn cell, where I stand
    obs = observe(world)
    assert obs.window_cell(flat(25, 4)) == (MY_TERRITORY, FREE)   # what the framework lets us see
    assert world.trail[25, 4] == OPP_TRAIL                        # what is really there
    assert flat(25, 4) in obs.trail_hidden                        # so the observation says it is a guess


def test_the_trail_layer_agrees_with_my_active_trail_even_when_most_of_it_is_out_of_view():
    world = RULES.initial_world()
    path = [(25, x) for x in range(6, 22)]
    put_trail(world, ME, path)
    obs = observe(world)
    assert len(obs.my_trail) == 16 and obs.covers(world.my_pos) and not obs.covers(flat(25, 6))
    assert obs.problems() == []
