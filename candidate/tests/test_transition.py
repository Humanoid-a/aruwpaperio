import numpy as np
import pytest

from candidate.agent.config import BOARD_WIDTH
from candidate.agent.data.cells import MY_TERRITORY, MY_TRAIL
from candidate.agent.data.types import Direction, MoveResult, Side
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.agent.geometry.zobrist import DEFAULT_KEYS
from candidate.agent.rules.match import MatchRules
from candidate.agent.rules.reference import apply_move
from candidate.agent.rules.transition import BitWorld, classify_moves, step
from candidate.tests.builders import put_agent, put_territory, put_trail
from candidate.tests.test_reference_rules import RULES, loop_world

ME, OPP = Side.ME, Side.OPP


def sample_worlds(fuzz_records, every=7):
    """Worlds from real play, all from the same match so they share one set of rules."""
    first_rules = fuzz_records[0][1]
    return first_rules, [w for w, rules, *_ in fuzz_records if rules is first_rules][::every]


# ---- BitWorld -----------------------------------------------------------


def test_round_trip_through_bitboards_keeps_layers_positions_and_counts():
    world = loop_world()
    bw = BitWorld.from_world(world)
    assert bw.n == 1 and bw.matches(0, world)
    assert bw.problems() == []
    my_count, opp_count = bw.counts()
    assert (int(my_count[0]), int(opp_count[0])) == (world.my_count, world.opp_count)
    assert bw.my_pos.dtype == np.int32 and bw.my_territory.dtype == np.int64


def test_matches_notices_any_difference():
    world = loop_world()
    bw = BitWorld.from_world(world)
    other = world.copy()
    other.territory[0, 0] = MY_TERRITORY
    assert not bw.matches(0, other)
    other = world.copy()
    other.opp_pos += 1
    assert not bw.matches(0, other)


def test_take_concatenate_and_copy_are_independent_copies():
    a, b = BitWorld.from_world(loop_world()), BitWorld.from_world(RULES.initial_world())
    both = BitWorld.concatenate([a, b])
    assert both.n == 2 and both.matches(0, loop_world()) and both.matches(1, RULES.initial_world())
    picked = both.take(np.array([1]))
    picked.my_territory[0, 0] = 5
    assert both.my_territory[1, 0] == 0
    original = int(both.my_pos[0])
    clone = both.copy()
    clone.my_pos[0] = 0
    assert int(both.my_pos[0]) == original != 0


@pytest.mark.parametrize(
    "corrupt,message",
    [
        (lambda bw: bw.opp_territory.__setitem__((0, 25), bw.my_territory[0, 25]), "same cell"),
        (lambda bw: bw.my_trail.__setitem__((0, 25), np.int64(1) << 4), "trail lies on my territory"),
        (lambda bw: bw.opp_pos.__setitem__(0, bw.my_pos[0]), "same cell"),
        (lambda bw: bw.my_territory.__setitem__((0, 3), np.int64(1) << 55), "valid"),
        (lambda bw: bw.my_pos.__setitem__(0, -1), "on-board"),
    ],
)
def test_problems_flags_a_corrupt_board(corrupt, message):
    bw = BitWorld.from_world(RULES.initial_world())
    corrupt(bw)
    assert any(message in p for p in bw.problems()), bw.problems()


def test_problems_requires_the_head_to_be_on_the_trail():
    world = loop_world()
    bw = BitWorld.from_world(world)
    bw.my_pos[0] = flat(40, 40)
    assert any("does not include the agent's cell" in p for p in bw.problems())


def test_hash_matches_the_zobrist_state_hash_and_changes_with_the_state():
    world = loop_world()
    bw = BitWorld.from_world(world)
    layers = world.bitboards()
    expected = DEFAULT_KEYS.hash_state(
        world.my_pos, world.opp_pos, layers.my_territory, layers.opp_territory, layers.my_trail, layers.opp_trail
    )
    assert bw.hash()[0] == expected
    assert BitWorld.from_world(world.copy()).hash()[0] == expected
    moved = step(bw, RULES, ME, [Direction.WEST]).world
    assert moved.hash()[0] != expected


# ---- classification -----------------------------------------------------


def test_classify_moves_on_the_loop_world():
    bw = BitWorld.from_world(loop_world())  # head (30, 6): west is the wall (30, 5), north is (29, 6)
    c = classify_moves(bw, ME)
    W_, N_, E_, S_ = Direction.WEST, Direction.NORTH, Direction.EAST, Direction.SOUTH
    assert c.legal[0].all()
    assert c.dest[0, W_] == flat(30, 5) and c.enters_own[0, W_] and c.captures[0, W_]
    assert not c.captures[0, N_] and not c.enters_own[0, N_]
    assert c.self_trail[0, E_] and c.dest[0, E_] == flat(30, 7)  # my own trail
    assert not c.self_trail[0, S_] and not c.cuts_other[0].any()


def test_classify_moves_marks_off_board_and_opponent_cells_illegal():
    rules = MatchRules.from_respawns(flat(0, 0), flat(25, 45))
    world = rules.initial_world()
    put_agent(world, OPP, (0, 1))
    c = classify_moves(BitWorld.from_world(world), ME)
    assert not c.legal[0, Direction.NORTH] and not c.legal[0, Direction.WEST]
    assert not c.legal[0, Direction.EAST]   # the opponent is there
    assert c.legal[0, Direction.SOUTH]
    assert c.dest[0, Direction.NORTH] == flat(0, 0)  # clamped to the mover's own cell
    assert not c.captures[0].any() and not c.self_trail[0].any()


def test_classification_agrees_with_what_step_then_does(fuzz_records):
    rules, worlds = sample_worlds(fuzz_records)
    bw = BitWorld.from_worlds(worlds)
    for side in Side:
        classes = classify_moves(bw, side)
        for direction in Direction:
            result = step(bw, rules, side, np.full(bw.n, direction))
            codes = result.result
            assert np.array_equal(codes == MoveResult.INVALID.value, ~classes.legal[:, direction])
            assert np.array_equal(codes == MoveResult.DIED.value, classes.self_trail[:, direction])
            assert np.array_equal(codes == MoveResult.CAPTURED.value, classes.captures[:, direction])
            assert np.array_equal(result.killed, classes.cuts_other[:, direction])


# ---- step ---------------------------------------------------------------


def test_step_does_not_modify_its_input_and_returns_int8_codes():
    bw = BitWorld.from_world(loop_world())
    before = bw.copy()
    result = step(bw, RULES, ME, [Direction.WEST])
    for field in ("my_territory", "opp_territory", "my_trail", "opp_trail", "my_pos", "opp_pos"):
        assert np.array_equal(getattr(bw, field), getattr(before, field))
    assert result.result.dtype == np.int8 and result.killed.dtype == np.bool_
    assert result.world is not bw


def test_an_invalid_move_leaves_the_board_unchanged():
    rules = MatchRules.from_respawns(flat(0, 0), flat(25, 45))
    bw = BitWorld.from_world(rules.initial_world())
    result = step(bw, rules, ME, [Direction.NORTH])
    assert result.result[0] == MoveResult.INVALID.value and not result.killed[0]
    assert result.world.matches(0, rules.initial_world())


@pytest.mark.parametrize("bad", [[4], [-1], [0, 1], np.zeros((1, 1), dtype=int)])
def test_step_rejects_malformed_directions(bad):
    bw = BitWorld.from_world(RULES.initial_world())
    with pytest.raises(ValueError):
        step(bw, RULES, ME, bad)


def test_step_handles_an_empty_batch():
    bw = BitWorld.from_worlds([RULES.initial_world()]).take(np.array([], dtype=int))
    result = step(bw, RULES, ME, np.array([], dtype=int))
    assert result.world.n == 0 and result.result.shape == (0,)


def test_boards_in_one_batch_do_not_affect_each_other():
    capture, start = loop_world(), RULES.initial_world()
    bw = BitWorld.from_worlds([capture, start, capture])
    result = step(bw, RULES, ME, [Direction.WEST, Direction.EAST, Direction.NORTH])
    assert list(result.result) == [MoveResult.CAPTURED.value, MoveResult.NORMAL.value, MoveResult.NORMAL.value]
    assert result.world.matches(0, apply_move(capture, RULES, ME, Direction.WEST).world)
    assert result.world.matches(1, apply_move(start, RULES, ME, Direction.EAST).world)
    assert result.world.matches(2, apply_move(capture, RULES, ME, Direction.NORTH).world)


def test_moves_along_every_board_edge_never_wrap():
    # An agent at the far east column stepping east, and at column 0 stepping west, is invalid.
    rules = MatchRules.from_respawns(flat(20, 0), flat(25, 45))
    world = rules.initial_world()
    put_territory(world, ME, [(20, 0), (20, BOARD_WIDTH - 1)])
    put_agent(world, ME, (20, BOARD_WIDTH - 1))
    bw = BitWorld.from_world(world)
    assert step(bw, rules, ME, [Direction.EAST]).result[0] == MoveResult.INVALID.value
    put_agent(world, ME, (20, 0))
    bw = BitWorld.from_world(world)
    assert step(bw, rules, ME, [Direction.WEST]).result[0] == MoveResult.INVALID.value


def test_a_batch_of_1000_boards_steps_quickly(fuzz_records):
    import time

    rules, worlds = sample_worlds(fuzz_records, every=1)
    worlds = (worlds * (1000 // len(worlds) + 1))[:1000]
    bw = BitWorld.from_worlds(worlds)
    directions = np.random.default_rng(0).integers(0, 4, size=1000)
    start = time.perf_counter()
    step(bw, rules, ME, directions)
    assert time.perf_counter() - start < 2.0  # generous: it takes a few milliseconds
