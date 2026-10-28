"""Behaviour randomness for training games (agent/behavior/style.py): none at 0, varied but never less safe above it."""

import dataclasses

from candidate.agent.behavior import expand, style
from candidate.agent.behavior.common import PERPENDICULAR, in_land
from candidate.agent.behavior.config import BEHAVIOR, BehaviorConfig
from candidate.agent.behavior.frame import Frame, Hints
from candidate.agent.behavior.snapshot import build_snapshot
from candidate.agent.controller import Controller
from candidate.agent.data.types import Direction
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.geometry.grid import flat, step
from candidate.agent.q.strategy import QStrategy
from candidate.agent.q.version import inputs_version
from candidate.tests.live import LiveGame
from candidate.tests.q_support import CLOSE, FAR, scene, world_with

E, W = Direction.EAST, Direction.WEST


def play(strategy, turns=150):
    controller = Controller(strategy, Diagnostics())
    live = LiveGame(FAR.initial_world(), FAR)
    moves = []
    for _ in range(turns):
        live.me.reset_for_new_turn()
        controller.take_turn(live.me)
        moves.append(strategy.last.direction)
        live.opponent_moves(E if live.world().opp_pos == flat(25, 45) else W)
    return moves, controller.diag


def test_randomness_zero_draws_nothing_and_changes_nothing():
    assert style.draw(BEHAVIOR, 5) == (BEHAVIOR, None)
    plain, _ = play(QStrategy(), 80)
    zero, _ = play(QStrategy(behavior=BehaviorConfig(randomness=0.0), seed=9), 80)
    assert plain == zero


def test_a_draw_is_seeded_varied_and_never_less_cautious_than_configured():
    cfg = BehaviorConfig(randomness=1.0)
    a, sa = style.draw(cfg, 1)
    b, _ = style.draw(cfg, 1)
    assert a == b
    draws = [style.draw(cfg, seed)[0] for seed in range(200)]
    assert len({(round(d.hunt_min_chance, 6), round(d.switch_cost, 6)) for d in draws}) > 50
    for d in draws:
        assert d.safety_threshold >= cfg.safety_threshold and d.safety_floor >= cfg.safety_floor
        assert 0.2 <= d.hunt_min_chance <= 0.8 and d.return_margin >= cfg.return_margin
        assert inputs_version(behavior=dataclasses.replace(d, randomness=0.0)) == inputs_version()   # inputs unchanged
    assert sa.changed and "randomness" not in sa.changed and sa.exit_bias.shape == (50, 50)


def test_randomness_varies_the_game_with_the_seed_and_stays_legal():
    games = [play(QStrategy(behavior=BehaviorConfig(randomness=1.0), seed=s)) for s in (1, 1, 2, 3)]
    moves = [m for m, _ in games]
    assert moves[0] == moves[1]                                             # same seed, same game
    assert moves[0] != moves[2] or moves[0] != moves[3]                     # other seeds, other games
    for _, diag in games:
        assert diag.counts["exceptions"] == 0 and diag.counts["move_INVALID"] == 0


def test_expand_jogs_sideways_out_in_the_open_only_when_the_style_says_so():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21)])                 # heading east, head at (25, 21)
    view = scene(world, CLOSE)
    snap = build_snapshot(view)
    _, drawn = style.draw(BehaviorConfig(randomness=1.0), 0)
    always = dataclasses.replace(drawn, jog_rate=1.0)
    never = dataclasses.replace(drawn, jog_rate=0.0)
    assert expand.move(Frame(view, snap, Hints(E))) == E
    assert expand.move(Frame(view, snap, Hints(E), style=never)) == E
    for _ in range(10):
        frame = Frame(view, snap, Hints(E), style=always)
        jog = expand.move(frame)
        assert jog in PERPENDICULAR[E] and not in_land(frame, step(frame.pos, jog))
