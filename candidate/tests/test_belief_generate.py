"""Step 1, generate: every move of every hypothesis resolved by the game rules, against the verified rules."""

import numpy as np

from candidate.agent.belief.generate import Children, build_children, classify, final_positions
from candidate.agent.data.cells import OPP_TERRITORY
from candidate.agent.data.hypotheses import FLAG_CAPTURED, FLAG_DIED, FLAG_KILLED_ME, HypothesisStore
from candidate.agent.data.types import Direction, Side
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.agent.rules.transition import BitWorld, step
from candidate.tests.belief_support import belief_state_of, situation_of, stale_base
from candidate.tests.builders import put_territory, put_trail
from candidate.tests.test_reference_rules import RULES, loop_world

OPP = Side.OPP


def children_of(world, rules, opp_base=None):
    state = belief_state_of(world, rules, opp_base=opp_base)
    exp = classify(state, situation_of(state, world))
    parents, moves = np.nonzero(exp.batch.legal)
    return state, exp, build_children(state, exp, parents, moves), moves


def test_every_child_equals_what_the_verified_rules_produce(fuzz_records):
    rng = np.random.default_rng(3)
    checked = captures = deaths = kills = 0
    for before, rules, *_ in fuzz_records[::7][:400]:
        base = stale_base(before, rng)
        state, exp, kids, moves = children_of(before, rules, base)
        for i, d in enumerate(moves):
            truth = step(BitWorld.from_world(before), rules, OPP, [Direction(d)])
            assert int(truth.result[0]) != 3, "a legal pair must not be an INVALID move"
            child = kids.store.take([i])
            assert child.problems(state.opp_diffs, state.my_diffs) == []
            next_world = truth.world
            assert int(child.pos[0]) == int(next_world.opp_pos[0])
            assert np.array_equal(child.trail_bb[0], next_world.opp_trail[0])
            assert np.array_equal(state.opp_land(child.opp_land_id)[0], next_world.opp_territory[0])
            assert int(child.opp_count[0]) == int(bitboard.popcount(next_world.opp_territory[0]))
            killed_me = bool(exp.batch.kills_me[0, d])
            assert killed_me == bool(truth.killed[0])   # the rules report that the opponent cut my trail
            if not killed_me:  # when I die my own territory resets, which is not the opponent's loss
                lost = before.my_count - int(bitboard.popcount(next_world.my_territory[0]))
                assert int(kids.my_loss[i]) == lost
                # and my land under the child is exactly my land in the verified rules
                assert np.array_equal(state.my_land(child.my_land_id)[0], next_world.my_territory[0])
            assert bool(child.flags[0] & FLAG_KILLED_ME) == killed_me
            checked += 1
            captures += bool(child.flags[0] & FLAG_CAPTURED)
            deaths += bool(child.flags[0] & FLAG_DIED)
            kills += killed_me
    assert checked > 1000 and captures > 0 and deaths > 0 and kills > 0


def test_the_children_know_where_they_came_from_and_start_with_no_weight():
    state, exp, kids, moves = children_of(loop_world(), RULES)
    assert kids.n == int(exp.batch.legal.sum())
    assert (kids.parent == 0).all() and kids.move.tolist() == moves.tolist()
    assert (kids.store.weight == 0).all()          # weights are belief.weigh's job


def test_a_capture_that_takes_my_land_reports_the_loss():
    world = RULES.initial_world()
    put_territory(world, OPP, [(y, 40) for y in range(20, 31)])
    put_territory(world, Side.ME, [(25, 38), (25, 39)])
    put_trail(world, OPP, [(20, 39), (20, 38), (20, 37)] + [(y, 37) for y in range(21, 30)] + [(30, 37), (30, 38), (30, 39)])
    state, exp, kids, moves = children_of(world, RULES, bitboard.from_mask(world.territory == OPP_TERRITORY))
    east = list(moves).index(Direction.EAST)
    assert exp.batch.captures[0, Direction.EAST]
    assert kids.store.flags[east] & FLAG_CAPTURED and kids.my_loss[east] == 2
    assert kids.store.trail_len[east] == 0 and kids.store.last_dir[east] == -1
    assert kids.store.opp_count[east] == world.opp_count + 15 + 18
    # the two cells it took are no longer mine under that child, and still are under the others
    assert bitboard.popcount(state.my_land(kids.store.my_land_id)[east]) == bitboard.popcount(state.my_land([0])[0]) - 2


def test_stepping_on_its_own_trail_sends_it_home_with_its_start_land_only():
    world = RULES.initial_world()
    put_trail(world, OPP, [(25, 43), (25, 42), (26, 42), (26, 43)])
    base = bitboard.from_mask(world.territory == OPP_TERRITORY)
    state, exp, kids, moves = children_of(world, RULES, base)
    north = list(moves).index(Direction.NORTH)   # (25, 43), the oldest trail cell
    assert exp.batch.self_trail[0, Direction.NORTH]
    child = kids.store.take([north])
    assert child.flags[0] & FLAG_DIED and int(child.pos[0]) == RULES.opp.respawn
    assert child.trail_len[0] == 0 and child.opp_count[0] == 9
    assert np.array_equal(state.opp_land(child.opp_land_id)[0], RULES.opp.start_bb)


def test_death_with_my_agent_on_its_respawn_cell_lands_on_the_alternate_cell():
    world = RULES.initial_world()
    put_trail(world, OPP, [(25, 43), (25, 42), (26, 42), (26, 43)])
    world.my_pos = RULES.opp.respawn   # I am standing on its respawn cell
    base = bitboard.from_mask(world.territory == OPP_TERRITORY)
    state = belief_state_of(world, RULES, opp_base=base)
    sit = situation_of(state, world)
    assert sit.opponent_respawn == RULES.opp.alt_respawn
    exp = classify(state, sit)
    parents, moves = np.nonzero(exp.batch.legal)
    kids = build_children(state, exp, parents, moves)
    north = list(moves).index(Direction.NORTH)
    assert int(kids.store.pos[north]) == RULES.opp.alt_respawn
    assert int(final_positions(exp)[0, Direction.NORTH]) == RULES.opp.alt_respawn


def test_moves_off_the_board_or_onto_my_agent_are_not_legal():
    world = RULES.initial_world()
    put_trail(world, OPP, [(0, 20)])
    world.my_pos = flat(0, 21)
    state = belief_state_of(world, RULES)
    exp = classify(state, situation_of(state, world))
    legal = exp.batch.legal[0]
    assert not legal[Direction.NORTH]   # off the board
    assert not legal[Direction.EAST]    # my agent
    assert legal[Direction.WEST] and legal[Direction.SOUTH]


def test_children_inside_a_batch_do_not_interfere():
    world = loop_world()
    state = belief_state_of(world, RULES)
    second = state.single(RULES.opp.respawn, RULES.opp.start_bb, bitboard.empty(), 9, my_land_id=0)   # same tables
    state.store = HypothesisStore.concatenate([state.store, second])
    sit = situation_of(state, world)
    exp = classify(state, sit)
    parents, moves = np.nonzero(exp.batch.legal)
    kids = build_children(state, exp, parents, moves)
    assert kids.store.problems(state.opp_diffs, state.my_diffs) == []
    assert kids.n == int(exp.batch.legal.sum())


def test_no_chosen_pair_gives_no_children():
    state, exp, kids, _ = children_of(loop_world(), RULES)
    nothing = build_children(state, exp, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    assert isinstance(nothing, Children) and nothing.n == 0
