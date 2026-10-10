"""The one-move predictor against the framework's real rules."""

import pytest

from candidate.agent import adapter
from candidate.agent.data.types import Direction, MoveResult, Side
from candidate.agent.geometry.grid import flat
from candidate.agent.prediction import predict_move
from candidate.agent.rules.match import MatchRules
from candidate.tests.builders import put_territory, put_trail
from candidate.tests.live import LiveGame
from candidate.tests.rules_oracle import real_apply
from candidate.tests.test_reference_rules import RULES, loop_world

N, E, S, W = Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST


def predict(world, rules, direction):
    obs = adapter.observation_from_game(LiveGame(world, rules).me, 0, None)
    return obs, predict_move(obs, world.bitboards().my_territory, rules, direction)


def test_a_plain_step_extends_the_trail_or_stays_home():
    world = RULES.initial_world()
    _, inside = predict(world, RULES, E)
    assert inside.result == MoveResult.NORMAL and inside.new_trail == () and inside.new_pos == flat(25, 5)
    world.my_pos = flat(25, 5)
    _, leaving = predict(world, RULES, E)
    assert leaving.new_trail == (flat(25, 6),) and leaving.dest == flat(25, 6)


def test_closing_a_loop_predicts_the_capture_and_its_size():
    obs, pred = predict(loop_world(), RULES, W)
    assert pred.result == MoveResult.CAPTURED and pred.new_trail == () and pred.new_pos == flat(30, 5)
    assert pred.gain == 15 + 18 and bitboard_count(pred.region) == 18


def bitboard_count(bb):
    from candidate.agent.geometry import bitboard

    return bitboard.popcount(bb)


def test_stepping_on_my_trail_predicts_death_and_the_respawn_cell():
    world = RULES.initial_world()
    put_trail(world, Side.ME, [(25, 6), (25, 7), (26, 7), (26, 6)])
    _, pred = predict(world, RULES, N)
    assert pred.result == MoveResult.DIED and pred.new_pos == flat(25, 4) and not pred.respawn_displaced


def test_death_with_the_opponent_on_my_respawn_cell_predicts_the_alternate_cell():
    world = RULES.initial_world()
    put_trail(world, Side.ME, [(25, 6), (25, 7), (26, 7), (26, 6)])
    put_trail(world, Side.OPP, [(25, 4)])
    _, pred = predict(world, RULES, N)
    assert pred.respawn_displaced and pred.new_pos == RULES.me.alt_respawn


def test_off_the_board_and_onto_the_opponent_are_invalid():
    rules = MatchRules.from_respawns(flat(0, 10), flat(25, 45))
    world = rules.initial_world()
    put_trail(world, Side.OPP, [(0, 12)])
    world.my_pos = flat(0, 11)
    _, off = predict(world, rules, N)
    _, onto = predict(world, rules, E)
    assert off.result == onto.result == MoveResult.INVALID and off.dest is None and onto.dest == flat(0, 12)


def test_stepping_on_the_opponents_trail_is_flagged_as_a_kill():
    world = RULES.initial_world()
    put_trail(world, Side.ME, [(25, 6), (25, 7)])
    put_trail(world, Side.OPP, [(25, 8), (24, 8)])
    _, pred = predict(world, RULES, E)
    assert pred.result == MoveResult.NORMAL and pred.killed


def test_predictions_match_the_engine_for_all_four_moves_on_random_positions(fuzz_records):
    checked = 0
    for before, rules, _side, _direction, _real in fuzz_records[:: 11][:350]:
        obs = adapter.observation_from_game(LiveGame(before, rules).me, 0, None)
        my_land = before.bitboards().my_territory  # as if memory were perfect
        for direction in Direction:
            pred = predict_move(obs, my_land, rules, direction)
            truth = real_apply(before, rules, Side.ME, direction)
            assert pred.result == truth.result, (direction, pred.result, truth.result)
            if truth.result == MoveResult.INVALID:
                continue
            assert pred.killed == truth.killed
            if truth.result == MoveResult.DIED:
                if obs.covers(rules.me.respawn):  # the opponent on my respawn cell is only seen if it is in view
                    assert pred.new_pos == truth.world.my_pos
            else:
                assert pred.new_pos == truth.world.my_pos
                assert pred.new_trail == tuple(truth.world.my_trail)
            if truth.result == MoveResult.CAPTURED and not truth.killed:
                assert pred.gain == truth.world.my_count - before.my_count
            checked += 1
    assert checked > 700
