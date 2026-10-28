"""NoisyEnemy: the framework's EnemyStateMachine, copied, with seeded per-game habits."""

from candidate.agent.controller import Controller
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.q.strategy import QStrategy
from candidate.examples.random_state_machine import RandomStateMachine
from candidate.offline import match
from candidate.offline.noisy_enemy import NoisyEnemy
from territorygame.controller.enemy_state_machine import EnemyStateMachine


def test_at_randomness_zero_it_is_the_original_move_for_move():
    assert NoisyEnemy().habits() == dict(randomness=0.0, bite=7, lookahead=2, vertical=1.0, aim=0.0, strike=0.0, wander=0.0)
    tapes = []
    for enemy in (EnemyStateMachine(), NoisyEnemy()):
        ours = match.Taped(match.AgentAdapter(Controller(QStrategy(), Diagnostics())))
        theirs = match.Taped(enemy)
        match.play([theirs, ours], 300, match.CLOSE)
        tapes.append((theirs.tape, ours.tape))
    assert tapes[0] == tapes[1]


def test_with_randomness_each_seed_has_its_own_habits_and_games_and_none_fails():
    habits = [NoisyEnemy(1.0, seed).habits() for seed in range(20)]
    assert len({(h["bite"], h["lookahead"], h["aim"]) for h in habits}) > 10
    assert all(h["bite"] in (5, 7, 9, 11) and 1 <= h["lookahead"] <= 3 for h in habits)
    tapes = {}
    for seed in (0, 0, 1):
        enemy = match.Taped(NoisyEnemy(1.0, seed))
        result = match.play([enemy, RandomStateMachine()], 300)
        assert result.failed_turns == 0
        tapes.setdefault(seed, []).append(enemy.tape)
    assert tapes[0][0] == tapes[0][1] and tapes[0][0] != tapes[1][0]
