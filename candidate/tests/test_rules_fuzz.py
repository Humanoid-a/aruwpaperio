"""Randomised comparison of the reference and fast rules with the framework's
real rules, on thousands of moves from randomly played games."""

from collections import Counter, defaultdict

import numpy as np
import pytest

from candidate.agent.data.types import Direction, Side
from candidate.agent.rules.reference import apply_move
from candidate.agent.rules.transition import BitWorld, step
from candidate.tests.rules_oracle import classify_event, real_apply


def test_every_intermediate_state_of_real_play_satisfies_the_world_invariants(fuzz_records):
    for before, *_ in fuzz_records:
        assert before.problems() == []


def test_reference_rules_reproduce_the_real_rules_on_every_move(fuzz_records):
    for before, rules, side, direction, real in fuzz_records:
        ref = apply_move(before, rules, side, direction)
        assert ref.world == real.world
        assert (ref.result, ref.killed) == (real.result, real.killed)


def test_fast_rules_reproduce_the_real_rules_on_every_move(fuzz_records):
    for before, rules, side, direction, real in fuzz_records:
        fast = step(BitWorld.from_world(before), rules, side, [direction])
        assert fast.world.matches(0, real.world)
        assert (int(fast.result[0]), bool(fast.killed[0])) == (real.result.value, real.killed)
        assert fast.world.problems() == []


def test_the_random_games_cover_the_rules_worth_checking(fuzz_records):
    events: Counter = Counter()
    for before, rules, side, _, real in fuzz_records:
        events.update(classify_event(before, side, real, rules))
    for needed in ("NORMAL", "INVALID", "DIED", "CAPTURED", "killed"):
        assert events[needed] > 20, (needed, dict(events))


def _by_match(records):
    groups = defaultdict(list)
    for before, rules, *_ in records:
        groups[id(rules)].append((before, rules))
    return groups.values()


@pytest.mark.parametrize("side", [Side.ME, Side.OPP])
@pytest.mark.parametrize("direction", list(Direction))
def test_a_whole_batch_in_one_call_matches_the_reference_for_a_move_nobody_made(fuzz_records, side, direction):
    """Batch step() over many boards at once, including the moves the random
    players did not choose: mixed captures, deaths, kills and invalid moves in
    one call."""
    checked = 0
    for group in _by_match(fuzz_records):
        rules = group[0][1]
        worlds = [w for w, _ in group[::3]]
        bw = BitWorld.from_worlds(worlds)
        result = step(bw, rules, side, np.full(len(worlds), direction))
        for i, world in enumerate(worlds):
            ref = apply_move(world, rules, side, direction)
            assert result.world.matches(i, ref.world), (i, side, direction)
            assert int(result.result[i]) == ref.result.value and bool(result.killed[i]) == ref.killed
            checked += 1
    assert checked > 1000


def test_untaken_moves_agree_with_the_real_rules_too(fuzz_records):
    sample = fuzz_records[:: max(1, len(fuzz_records) // 80)]
    for before, rules, *_ in sample:
        for side in Side:
            for direction in Direction:
                real = real_apply(before, rules, side, direction)
                ref = apply_move(before, rules, side, direction)
                assert ref.world == real.world and (ref.result, ref.killed) == (real.result, real.killed)
