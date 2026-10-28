"""The safety look-ahead: the margin a move would leave, checked against what really happens next frame."""

import dataclasses
import math

import numpy as np

from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat, step
from candidate.agent.behavior import safety
from candidate.agent.behavior.config import BEHAVIOR, BehaviorConfig
from candidate.agent.behavior.snapshot import build_snapshot
from candidate.tests.q_support import CLOSE, belief_of, scene, world_with


def snap_of(world, enemy_at, rules=CLOSE):
    return build_snapshot(scene(world, rules, belief_of([dict(pos=flat(*enemy_at), weight=1.0)])))


def test_home_is_always_safe_and_a_kill_is_scored_from_the_enemys_respawn():
    snap = snap_of(world_with(CLOSE, my_trail=[(25, 20), (25, 21)], opp_trail=[(24, 23), (24, 22)]), (24, 22))
    assert safety.margin_after(snap, flat(25, 19), BEHAVIOR) == math.inf              # home
    # cutting its trail at (24, 23) sends it to its respawn (25, 31); the nearest of my exposed cells (that one) is 9
    # from there, it moves once before I look again, and my way home from (24, 23) is 6 steps: margin (9 - 1) - 6
    assert safety.kills(snap, flat(24, 23))
    assert safety.margin_after(snap, flat(24, 23), BEHAVIOR) == (9 - 1) - 6


def test_a_step_toward_the_enemy_costs_up_to_three_turns_of_margin_and_a_step_home_costs_none():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21)])                  # head (25, 21); home is 3 away via row 24
    snap = snap_of(world, (25, 27))
    assert snap.d1 == 6 and snap.d4 == 3 and snap.d7 == 3
    # east, toward the enemy: it is 5 from (25, 22) and then moves; I am 4 from home from there: 3 -> 0
    assert safety.margin_after(snap, flat(25, 22), BEHAVIOR) == 0.0
    # north, toward home: the margin stays what it was
    assert safety.margin_after(snap, flat(24, 21), BEHAVIOR) == snap.d7


def test_the_margin_offset_shifts_the_prediction_like_every_other_margin():
    snap = snap_of(world_with(CLOSE, my_trail=[(25, 20), (25, 21)]), (25, 27))
    assert safety.margin_after(snap, flat(25, 22), BehaviorConfig(margin_offset=-1)) == -1.0


def test_a_cell_with_no_way_home_is_unsafe():
    snap = snap_of(world_with(CLOSE, my_trail=[(25, 20), (25, 21)]), (25, 40))
    home_dist = snap.home_dist.copy()
    home_dist[flat(25, 22)] = 9999                                            # a dead end
    boxed = dataclasses.replace(snap, home_dist=home_dist)
    assert safety.margin_after(boxed, flat(25, 22), BEHAVIOR) < 0 <= safety.margin_after(snap, flat(25, 22), BEHAVIOR)


def test_the_prediction_never_promises_more_than_the_next_frame_delivers():
    """Property: whatever the enemy then does, the next frame's real margin is at least the predicted one."""
    rng = np.random.default_rng(11)
    checked = tight = 0
    for _ in range(40):
        n = int(rng.integers(0, 4))
        trail = [(25, 19 + k) for k in range(1, n + 1)]
        head = (25, 19 + n) if n else (25, 19)
        enemy = (int(rng.integers(19, 32)), int(rng.integers(head[1] + 3, head[1] + 12)))
        world = world_with(CLOSE, my_trail=trail) if trail else world_with(CLOSE, me=head)
        snap = snap_of(world, enemy)
        for d in Direction:
            dest = step(flat(*head), d)
            if dest is None or bitboard.has_cell(snap.my_trail_bb, dest) or dest == flat(*enemy):
                continue
            predicted = safety.margin_after(snap, dest, BEHAVIOR)
            if math.isinf(predicted):
                continue
            dy, dx = divmod(dest, 50)
            after_trail = trail + [(dy, dx)]
            worst = None
            for reply in Direction:
                nxt = step(flat(*enemy), reply)
                if nxt is None or nxt == dest:
                    continue
                real = build_snapshot(scene(world_with(CLOSE, my_trail=after_trail), CLOSE, belief_of([dict(pos=nxt, weight=1.0)])))
                worst = real.safety_margin if worst is None else min(worst, real.safety_margin)
            assert worst is not None and worst >= predicted, (trail, enemy, d, predicted, worst)
            checked += 1
            tight += worst == predicted
    assert checked > 60 and tight > checked // 3          # and it is not vacuous: often exactly what the worst reply does


# ---- regressions from a recorded game: the hunter blocking my way home ----------------------------------------------------

# my land in that game: the 3x3 start plus a strip to the north-west
RECORDED_LAND = [(y, x) for y in range(20, 24) for x in (16, 17)]


def test_a_land_cell_the_enemy_stands_on_is_not_a_way_home():
    # I am at (23, 19) with a trail (23, 18), (23, 19); the hunter stands on (24, 19), the land cell right below me
    world = world_with(CLOSE, my_land=RECORDED_LAND, my_trail=[(23, 18), (23, 19)], opp=(24, 19))
    snap = build_snapshot(scene(world, CLOSE))
    assert flat(24, 19) not in (snap.home_path or [])
    assert snap.d4 == 3 and snap.home_path == [flat(22, 19), flat(22, 18), flat(22, 17)]
    assert snap.d7 == 1 - 3                                    # the margin was really -2, not the 0 it seemed


def test_the_look_ahead_counts_the_land_cell_my_way_home_enters_by():
    # one turn earlier: I am at (23, 18), one cell out of my land; the hunter is at (25, 19)
    world = world_with(CLOSE, my_land=RECORDED_LAND, my_trail=[(23, 18)], opp=(25, 19))
    snap = build_snapshot(scene(world, CLOSE))
    # going east to (23, 19): home from there enters by (24, 19), which the hunter reaches in one move
    assert safety.margin_after(snap, flat(23, 19), BEHAVIOR) < 0
    # going home is fine
    assert safety.margin_after(snap, flat(24, 18), BEHAVIOR) == math.inf


def test_a_kill_next_to_the_enemys_respawn_is_not_safe():
    # my trail runs east along row 24 up to (24, 28), beside the enemy's land (cols 30-32); its trail starts at (24, 29)
    trail = [(24, x) for x in range(20, 29)]
    world = world_with(CLOSE, my_trail=trail, opp=(22, 29), opp_trail=[(24, 29), (23, 29), (22, 29)])
    snap = build_snapshot(scene(world, CLOSE))
    kill = flat(24, 29)
    assert safety.kills(snap, kill)
    margin = safety.margin_after(snap, kill, BEHAVIOR)
    assert math.isfinite(margin) and margin < 0          # it reappears at (25, 31), 3 from the cell I stand on, and I am far from home
    assert safety.margin_after(snap, flat(24, 19), BEHAVIOR) == math.inf     # home is still always safe
