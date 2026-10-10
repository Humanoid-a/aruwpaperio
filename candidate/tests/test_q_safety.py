"""The safety look-ahead: the margin a move would leave, checked against what really happens next frame."""

import dataclasses
import math

import numpy as np
import pytest

from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat, step
from candidate.agent.q import safety
from candidate.agent.q.config import Q, QConfig
from candidate.agent.q.snapshot import build_snapshot
from candidate.tests.q_support import CLOSE, belief_of, scene, world_with


def snap_of(world, enemy_at, rules=CLOSE):
    return build_snapshot(scene(world, rules, belief_of([dict(pos=flat(*enemy_at), weight=1.0)])))


def test_stepping_onto_my_own_land_or_an_enemy_trail_cell_is_always_safe():
    snap = snap_of(world_with(CLOSE, my_trail=[(25, 20), (25, 21)], opp_trail=[(24, 23), (24, 22)]), (24, 22))
    assert safety.margin_after(snap, flat(25, 19), Q) == math.inf              # home
    assert safety.margin_after(snap, flat(24, 23), Q) == math.inf              # a known enemy trail cell: it would die


def test_a_step_toward_the_enemy_costs_up_to_three_turns_of_margin_and_a_step_home_costs_none():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21)])                  # head (25, 21); home is 3 away via row 24
    snap = snap_of(world, (25, 27))
    assert snap.d1 == 6 and snap.d4 == 3 and snap.d7 == 3
    # east, toward the enemy: it is 5 from (25, 22) and then moves; I am 4 from home from there: 3 -> 0
    assert safety.margin_after(snap, flat(25, 22), Q) == 0.0
    # north, toward home: the margin stays what it was
    assert safety.margin_after(snap, flat(24, 21), Q) == snap.d7


def test_the_margin_offset_shifts_the_prediction_like_every_other_margin():
    snap = snap_of(world_with(CLOSE, my_trail=[(25, 20), (25, 21)]), (25, 27))
    assert safety.margin_after(snap, flat(25, 22), QConfig(margin_offset=-1)) == -1.0


def test_a_cell_with_no_way_home_is_unsafe():
    snap = snap_of(world_with(CLOSE, my_trail=[(25, 20), (25, 21)]), (25, 40))
    home_dist = snap.home_dist.copy()
    home_dist[flat(25, 22)] = 9999                                            # a dead end
    boxed = dataclasses.replace(snap, home_dist=home_dist)
    assert safety.margin_after(boxed, flat(25, 22), Q) < 0 <= safety.margin_after(snap, flat(25, 22), Q)


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
            predicted = safety.margin_after(snap, dest, Q)
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
                worst = real.d7 if worst is None else min(worst, real.d7)
            assert worst is not None and worst >= predicted, (trail, enemy, d, predicted, worst)
            checked += 1
            tight += worst == predicted
    assert checked > 60 and tight > checked // 3          # and it is not vacuous: often exactly what the worst reply does
