"""Full-engine check of the move rules: after every real turn of real
matches, the board must equal what the reference and fast transitions predict
from the board before it and the move the controller made."""

import pytest

from candidate.agent.data.types import MoveResult, Side
from candidate.agent.geometry.grid import flat
from candidate.agent.rules.match import MatchRules
from candidate.agent.rules.reference import apply_move
from candidate.agent.rules.transition import BitWorld, step
from candidate.examples.basic_state_machine import BasicStateMachine
from candidate.examples.random_state_machine import RandomStateMachine
from candidate.tests.engine_support import MoveLogger, play_match, world_from_snapshot
from territorygame.controller.enemy_state_machine import EnemyStateMachine

TURNS = 200
# Respawns (y, x) per player, close enough that the players keep meeting.
MATCHES = {
    "enemy_vs_basic": (lambda: [EnemyStateMachine(), BasicStateMachine()], [(25, 18), (25, 31)]),
    "basic_vs_random": (lambda: [BasicStateMachine(), RandomStateMachine()], [(20, 22), (30, 28)]),
}


@pytest.fixture(scope="module")
def logged_matches() -> dict:
    out = {}
    for name, (make, respawns) in MATCHES.items():
        attempts: list = []
        controllers = [MoveLogger(c, i, attempts) for i, c in enumerate(make())]
        out[name] = (play_match(controllers, TURNS, respawns), attempts, respawns)
    return out


@pytest.mark.parametrize("name", MATCHES)
def test_every_real_turn_is_one_application_of_the_rules(logged_matches, name):
    snapshots, attempts, respawns = logged_matches[name]
    assert all(s.error_message is None for s in snapshots), "a forfeited turn would break the alignment"
    moves = [a for a in attempts if a[2] != MoveResult.INVALID]
    assert len(moves) == len(snapshots) - 1

    cells = [flat(y, x) for y, x in respawns]
    results = set()
    for j, (player, direction, result) in enumerate(moves):
        rules = MatchRules.from_respawns(cells[player], cells[1 - player])
        before = world_from_snapshot(snapshots[j], player)
        expected = world_from_snapshot(snapshots[j + 1], player)

        outcome = apply_move(before, rules, Side.ME, direction)
        assert outcome.world == expected, f"turn {j}: {direction.name} by player {player}"
        assert outcome.result == result
        fast = step(BitWorld.from_world(before), rules, Side.ME, [direction])
        assert fast.world.matches(0, expected)
        assert int(fast.result[0]) == result.value
        results.add(result)

    assert {MoveResult.NORMAL, MoveResult.CAPTURED, MoveResult.DIED} <= results


@pytest.mark.parametrize("name", MATCHES)
def test_the_opponents_view_of_each_turn_is_predicted_too(logged_matches, name):
    """The same turns seen from the other player's side (they are 'ME' in their
    own world), so a mover is predicted from either perspective."""
    snapshots, attempts, respawns = logged_matches[name]
    moves = [a for a in attempts if a[2] != MoveResult.INVALID]
    cells = [flat(y, x) for y, x in respawns]
    for j in range(0, len(moves), 3):
        player, direction, _ = moves[j]
        other = 1 - player
        rules = MatchRules.from_respawns(cells[other], cells[player])
        before = world_from_snapshot(snapshots[j], other)  # the bystander's view
        expected = world_from_snapshot(snapshots[j + 1], other)
        assert apply_move(before, rules, Side.OPP, direction).world == expected
        fast = step(BitWorld.from_world(before), rules, Side.OPP, [direction])
        assert fast.world.matches(0, expected)
