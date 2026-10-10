"""The belief filter, step by step. Each test sets up a small situation, drives the filter the way the runtime
does (memory takes the window, the belief observes, the agent's move is applied at the end of the frame) and
checks what the evidence did.

  start       the starting state is exactly right
  generate    the moves of each hypothesis (against the verified rules: test_belief_generate)
  cut         position, my death, the territory counts, the window
  weigh       K is applied after the cuts, and the probabilities are over all the legal moves
  settle      merging, pruning into the residual, rehydration
"""

import numpy as np
import pytest

from candidate.agent.belief.config import BeliefConfig, ProportionalBudget
from candidate.agent.belief.cut import Evidence, cut_children
from candidate.agent.belief.generate import Children, Situation, build_children, classify
from candidate.agent.belief.motion import ConstantMotionModel, MotionModel
from candidate.agent.belief.weigh import assign_weights
from candidate.agent.data.belief_view import BeliefEvent
from candidate.agent.data.cells import MY_TERRITORY, OPP_TERRITORY, OPP_TRAIL
from candidate.agent.data.hypotheses import HypothesisStore
from candidate.agent.data.territory import TerritoryDiffs
from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.agent.rules.match import MatchRules
from candidate.tests.belief_rig import FAR, NEAR, Rig, plain_move, prime
from candidate.tests.builders import make_observation, square

N, E, S, W = Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST


def opponent_land(rules, window_pos):
    """The cells of the opponent's start that fall in the window of an agent at window_pos."""
    obs = make_observation(window_pos, respawn=window_pos)
    return {divmod(c, 50): OPP_TERRITORY for c in rules.opp.start_cells if obs.covers(c)}


def sound(rig):
    state = rig.state
    assert state.store.problems(state.opp_diffs, state.my_diffs) == []
    assert rig.total_mass() == pytest.approx(1.0)
    assert rig.belief.view.position.sum() == pytest.approx(1.0, abs=1e-5)


# ---- start ----------------------------------------------------------------------------------------------


def test_a_new_match_starts_with_one_certain_hypothesis_at_the_respawn():
    rig = Rig(FAR)
    store = rig.state.store
    assert store.n == 1 and int(store.pos[0]) == FAR.opp.respawn and store.weight[0] == 1.0
    assert store.opp_count[0] == 9 and store.trail_len[0] == 0
    assert store.problems(rig.state.opp_diffs, rig.state.my_diffs) == []


def test_the_starting_hypothesis_owns_exactly_the_start_lands_and_so_do_the_bases():
    rig = Rig(NEAR)
    state = rig.state
    assert np.array_equal(state.opp_land(state.store.opp_land_id)[0], NEAR.opp.start_bb)
    assert np.array_equal(state.my_land(state.store.my_land_id)[0], NEAR.me.start_bb)
    assert np.array_equal(state.opp_base, NEAR.opp.start_bb) and np.array_equal(state.my_base, NEAR.me.start_bb)
    assert state.me.pos == NEAR.me.respawn and state.me.count == 9 and state.me.trail_bb.sum() == 0
    assert state.first and not state.needs_reseed and state.residual.total() == 0


def test_the_starting_state_agrees_with_the_boards_initial_world():
    world = FAR.initial_world()
    rig = Rig(FAR)
    state = rig.state
    assert state.me.pos == world.my_pos and int(state.store.pos[0]) == world.opp_pos
    assert state.me.count == world.my_count and int(state.store.opp_count[0]) == world.opp_count


def test_the_first_mover_expects_the_opponent_not_to_have_moved_yet():
    # My respawn is the left one, which the default config gives the player who moves first.
    rig = Rig(FAR)
    view = rig.step(make_observation((25, 4), turn=0, respawn=(25, 4)))
    assert view.probability_at(FAR.opp.respawn) == pytest.approx(0.95)
    for cell in [(24, 45), (26, 45), (25, 44), (25, 46)]:
        assert view.probability_at(flat(*cell)) == pytest.approx(0.0125)         # it moved once: unlikely
    sound(rig)


def test_the_second_mover_expects_the_opponent_to_have_moved_once():
    rig = Rig(MatchRules.from_respawns(flat(25, 45), flat(25, 4)))
    view = rig.step(make_observation((25, 45), turn=0, respawn=(25, 45)))
    assert view.probability_at(flat(25, 4)) == pytest.approx(0.05)
    for cell in [(24, 4), (26, 4), (25, 3), (25, 5)]:
        assert view.probability_at(flat(*cell)) == pytest.approx(0.2375)         # 0.95 / 4
    sound(rig)


def test_the_seat_is_only_a_belief_so_the_unlikely_order_keeps_some_probability():
    rig = Rig(FAR, BeliefConfig(seat_confidence=0.8))
    view = rig.step(make_observation((25, 4), turn=0, respawn=(25, 4)))
    assert view.probability_at(FAR.opp.respawn) == pytest.approx(0.8)


def test_an_explicit_setting_overrides_the_inferred_seat():
    rig = Rig(FAR, BeliefConfig(first_update_stay=0.5))
    view = rig.step(make_observation((25, 4), turn=0, respawn=(25, 4)))
    assert view.probability_at(FAR.opp.respawn) == pytest.approx(0.5)


def test_afterwards_every_hypothesis_moves_exactly_once_per_update():
    rig = Rig(FAR)
    rig.step(make_observation((25, 4), turn=0, respawn=(25, 4)))
    view = rig.step(make_observation((25, 4), turn=1, respawn=(25, 4)))
    cells = np.flatnonzero(view.position.reshape(-1) > 0)
    # one more step from {respawn, its neighbours}: everything is two steps away or the respawn region again
    assert flat(25, 45) in cells and flat(25, 47) in cells and flat(23, 45) in cells
    assert all(abs(c // 50 - 25) + abs(c % 50 - 45) <= 2 for c in cells)
    sound(rig)


def test_a_first_observation_that_is_not_a_start_rebuilds_the_belief_from_what_is_seen():
    rig = Rig(FAR, seed_memory=False)
    mid_match = make_observation((25, 9), turn=0, respawn=(25, 4), my_count=31, opp_count=17)
    rig.belief.begin_match(FAR, mid_match)
    assert rig.state.needs_reseed
    view = rig.step(mid_match)
    assert BeliefEvent.RESEEDED in view.events and view.position.sum() == pytest.approx(1.0, abs=1e-5)


# ---- cut: position ------------------------------------------------------------------------------------------


def test_seeing_the_opponent_puts_all_the_belief_on_its_cell():
    rig = Rig(NEAR)
    land = opponent_land(NEAR, (25, 10))
    view = rig.step(make_observation((25, 10), turn=0, respawn=(25, 10), territory=land, opp_at=(25, 15), opp_count=9))
    assert view.sighted_at == flat(25, 15) and view.probability_at(flat(25, 15)) == pytest.approx(1.0)
    assert BeliefEvent.SIGHTED in view.events and BeliefEvent.RESEEDED not in view.events
    assert view.n_hypotheses >= 1 and (rig.state.store.pos == flat(25, 15)).all()
    sound(rig)


def test_not_seeing_the_opponent_removes_every_cell_of_the_window():
    rig = Rig(NEAR)
    land = opponent_land(NEAR, (25, 10))
    obs = make_observation((25, 10), turn=0, respawn=(25, 10), territory=land, opp_count=9)
    view = rig.step(obs)
    window = np.zeros((50, 50), dtype=bool)
    rows, cols = obs.window_slices
    window[rows, cols] = True
    assert view.position[window].sum() == 0
    assert view.position.sum() == pytest.approx(1.0, abs=1e-5)
    sound(rig)


def test_a_sighting_nothing_in_the_belief_can_explain_rebuilds_it_from_the_sighting():
    rig = Rig(FAR)
    # The opponent appears right beside me on turn 0, where nothing could have put it.
    obs = make_observation((25, 4), turn=0, respawn=(25, 4), opp_at=(25, 8), opp_count=13, trail_hidden=(flat(25, 8),),
                           trail={(25, 8): OPP_TRAIL})
    view = rig.step(obs)
    assert BeliefEvent.RESEEDED in view.events and rig.belief.stats.reseeds == 1
    assert view.probability_at(flat(25, 8)) == pytest.approx(1.0) and view.n_hypotheses == 1
    assert int(rig.state.store.opp_count[0]) == 13 and int(rig.state.store.trail_len[0]) == 1
    sound(rig)


def test_a_forced_rebuild_without_a_sighting_spreads_over_where_it_could_have_got_to():
    rig = Rig(FAR)
    rig.step(make_observation((25, 10), turn=0, respawn=(25, 10), opp_at=(25, 15), opp_count=9, trail_hidden=(flat(25, 15),)))
    view = rig.step(make_observation((25, 4), turn=4, respawn=(25, 4)), force_reseed=True)
    assert BeliefEvent.RESEEDED in view.events and view.n_hypotheses == 0
    cells = np.flatnonzero(view.position.reshape(-1) > 0)
    assert all(abs(c // 50 - 25) + abs(c % 50 - 15) <= 4 for c in cells)   # within four moves of the last sighting
    assert view.position.sum() == pytest.approx(1.0, abs=1e-5)


def test_without_any_sighting_a_rebuild_covers_the_whole_board_outside_the_window():
    rig = Rig(FAR)
    obs = make_observation((25, 4), turn=3, respawn=(25, 4))
    view = rig.step(obs, force_reseed=True)
    rows, cols = obs.window_slices
    assert view.position[rows, cols].sum() == 0 and view.support.sum() > 2000 and view.n_hypotheses == 0


# ---- cut: stepping on my trail ----------------------------------------------------------------------------------


# My trail is (25, 11), (25, 12) with my head on (25, 12). The opponent cannot step onto my head (that
# is an invalid move); stepping onto (25, 11), behind it, cuts the trail and kills me.
MY_TRAIL_CELLS = [(25, 11), (25, 12)]


def home_and_enemy_land():
    """What the window around my respawn shows: my start land and the edge of the opponent's."""
    land = {c: MY_TERRITORY for c in square(25, 10)}
    land.update(opponent_land(NEAR, (25, 10)))
    return land


def test_if_i_did_not_die_no_survivor_stepped_onto_my_trail():
    rig = Rig(NEAR)
    prime(rig, [(flat(24, 11), [(24, 11)], 0.4, 9), (flat(26, 11), [(26, 11)], 0.3, 9), (flat(24, 13), [(24, 13)], 0.3, 9)])
    rig.move(plain_move(MY_TRAIL_CELLS))
    view = rig.step(make_observation((25, 12), turn=5, respawn=(25, 10), my_trail=MY_TRAIL_CELLS, opp_count=9))
    assert view.probability_at(flat(25, 11)) == 0      # it cannot have stepped there, or I would be dead
    assert not (rig.state.store.flags & 4).any()
    assert view.position.sum() == pytest.approx(1.0, abs=1e-5)


def test_if_the_opponent_killed_me_only_moves_onto_my_trail_survive():
    rig = Rig(NEAR)
    prime(rig, [(flat(24, 11), [(24, 11)], 0.4, 9), (flat(26, 11), [(26, 11)], 0.3, 9), (flat(24, 13), [(24, 13)], 0.3, 9)])
    rig.move(plain_move(MY_TRAIL_CELLS))
    # The opponent stepped south from (24, 11) onto (25, 11), cutting my trail; I am back home at (25, 10).
    # Its own trail is now (24, 11), (25, 11), and I can see the first of those cells.
    obs = make_observation(
        (25, 10), turn=5, respawn=(25, 10), opp_at=(25, 11), opp_count=9, trail_hidden=(flat(25, 11),),
        trail={(24, 11): OPP_TRAIL, (25, 11): OPP_TRAIL}, territory=home_and_enemy_land(),
    )
    view = rig.step(obs, killed=True)
    store = rig.state.store
    assert view.probability_at(flat(25, 11)) == pytest.approx(1.0) and BeliefEvent.RESEEDED not in view.events
    assert store.n == 1 and store.flags[0] & 4                       # it is flagged as the move that killed me
    assert sorted(bitboard.to_cells(store.trail_bb[0])) == [flat(24, 11), flat(25, 11)]
    sound(rig)


def test_a_killing_move_by_a_hypothesis_that_cannot_reach_my_trail_is_ruled_out():
    rig = Rig(NEAR)
    # Nothing here is next to my trail, so if I died it was not by any of them: the belief must be rebuilt.
    prime(rig, [(flat(24, 14), [(24, 14)], 0.5, 9), (flat(26, 14), [(26, 14)], 0.5, 9)])
    rig.move(plain_move(MY_TRAIL_CELLS))
    obs = make_observation(
        (25, 10), turn=5, respawn=(25, 10), opp_at=(25, 11), opp_count=9, trail_hidden=(flat(25, 11),),
        trail={(25, 11): OPP_TRAIL}, territory=home_and_enemy_land(),
    )
    view = rig.step(obs, killed=True)
    assert BeliefEvent.RESEEDED in view.events


# ---- cut: history ----------------------------------------------------------------------------------------------------


def test_a_territory_count_nothing_can_reach_rules_out_every_hypothesis():
    rig = Rig(FAR)
    rig.step(make_observation((25, 4), turn=0, respawn=(25, 4)))
    before = rig.belief.stats.cut_by_count
    view = rig.step(make_observation((25, 4), turn=1, respawn=(25, 4), opp_count=12))   # +3 with no loop to close
    assert rig.belief.stats.cut_by_count > before
    assert BeliefEvent.RESEEDED in view.events or view.hypothesis_mass < 1.0


def test_a_visible_trail_that_a_hypothesis_does_not_have_rules_it_out():
    rig = Rig(NEAR)
    # Both could step to (25, 13), but only the first has the trail cell (25, 14) that I can see there.
    prime(rig, [(flat(25, 14), [(25, 14)], 0.5, 9), (flat(25, 12), [(25, 12)], 0.5, 9)])
    obs = make_observation(
        (25, 10), turn=3, respawn=(25, 10), opp_at=(25, 13), opp_count=9, trail_hidden=(flat(25, 13),),
        trail={(25, 14): OPP_TRAIL, (25, 13): OPP_TRAIL}, territory=opponent_land(NEAR, (25, 10)),
    )
    rig.step(obs)
    survivors = rig.state.store
    assert survivors.n == 1 and int(survivors.pos[0]) == flat(25, 13)
    assert sorted(bitboard.to_cells(survivors.trail_bb[0])) == [flat(25, 13), flat(25, 14)]
    assert rig.belief.stats.cut_by_window > 0


def test_territory_the_window_shows_is_enforced_on_the_hypotheses():
    rig = Rig(NEAR)
    # A hypothesis that believes the opponent owns the start land, which I can see is not its land.
    prime(rig, [(flat(25, 17), [], 1.0, 9)])
    state = rig.state
    state.opp_diffs = TerritoryDiffs()
    state.opp_base = bitboard.empty()
    state.store.opp_land_id[:] = state.opp_diffs.intern_territory(NEAR.opp.start_bb, state.opp_base)[0]
    obs = make_observation((25, 10), turn=3, respawn=(25, 10), opp_count=9)    # ... but the window shows no such land
    before = rig.belief.stats.cut_by_window
    rig.step(obs)
    assert rig.belief.stats.cut_by_window > before


def children_for_gate(my_loss, my_exact):
    """Two stored hypotheses, as children with the given loss of my land and exactness of its view."""
    rig = Rig(NEAR)
    prime(rig, [(flat(40, 40), [], 0.5, 9), (flat(41, 40), [], 0.5, 9)])
    rig.move(plain_move([], pos_after=flat(25, 10), count=9))
    obs = make_observation((25, 10), turn=3, respawn=(25, 10), my_count=8, territory=opponent_land(NEAR, (25, 10)))
    sit = Situation.of(rig.state, obs)
    kids = Children(
        rig.state.store.take([0, 1]), np.array([0, 1]), np.array([0, 0]), np.array(my_loss), np.array(my_exact)
    )
    return rig, sit, kids, obs


def test_my_count_is_held_against_what_each_childs_move_takes_from_me():
    rig, sit, kids, obs = children_for_gate([0, 1], [True, True])
    # my count was 9 after my move; the observation says 8: only the hypothesis that takes one cell fits
    keep = cut_children(kids, sit, Evidence(obs, False), rig.belief.stats)
    assert keep.tolist() == [False, True]


def test_a_view_of_my_land_that_does_not_add_up_is_not_judged_by_my_count():
    rig, sit, kids, obs = children_for_gate([0, 1], [False, True])
    assert cut_children(kids, sit, Evidence(obs, False), rig.belief.stats).tolist() == [True, True]


def test_my_count_is_not_checked_when_the_opponent_killed_me_or_when_it_is_unknown():
    rig, sit, kids, obs = children_for_gate([0, 1], [True, True])
    assert cut_children(kids, sit, Evidence(obs, True), rig.belief.stats).all()
    unknown = Situation.of(rig.state, obs)
    rig.state.me = type(rig.state.me)(rig.state.me.pos, rig.state.me.trail_bb, None, None)
    unknown = Situation.of(rig.state, obs)
    assert cut_children(kids, unknown, Evidence(obs, False), rig.belief.stats).all()


# ---- weigh: K comes after the cuts, and its probabilities are over all the legal moves -----------------------------------


def test_the_motion_model_is_shown_only_the_hypotheses_that_still_have_a_child():
    seen = []

    class Recorder(ConstantMotionModel):
        def score(self, batch):
            seen.append(batch)
            return super().score(batch)

    rig = Rig(NEAR, motion=Recorder())
    # Three hypotheses. The opponent is seen at (25, 13) with a trail along (25, 14): the first can have made that
    # step; the second can step there but would have a different trail, which the window contradicts; the third
    # cannot get there at all.
    prime(rig, [(flat(25, 14), [(25, 14)], 0.4, 9), (flat(24, 13), [(24, 13)], 0.4, 9), (flat(30, 30), [], 0.2, 9)])
    obs = make_observation(
        (25, 10), turn=3, respawn=(25, 10), opp_at=(25, 13), opp_count=9, trail_hidden=(flat(25, 13),),
        trail={(25, 14): OPP_TRAIL, (25, 13): OPP_TRAIL}, territory=opponent_land(NEAR, (25, 10)),
    )
    rig.step(obs)
    assert len(seen) == 1 and seen[0].n == 1 and seen[0].pos.tolist() == [flat(25, 14)]   # not three
    assert seen[0].legal.shape == (1, 4)                                                   # but all four of its moves
    sound(rig)


def test_the_probability_of_a_move_is_over_all_the_legal_moves_not_only_the_survivors():
    rig = Rig(FAR)
    state = rig.state
    state.first = False

    sit = Situation.of(state, make_observation((25, 4), turn=0, respawn=(25, 4)))
    exp = classify(state, sit)
    kids = build_children(state, exp, np.array([0]), np.array([int(E)]))     # only one of its four moves survived
    assign_weights(kids, exp, ConstantMotionModel(), BeliefConfig(), None)
    assert kids.store.weight[0] == pytest.approx(0.25)                         # P(east | four equal moves), not 1.0


def test_a_hypothesis_that_has_not_moved_keeps_its_share_of_the_first_look():
    rig = Rig(FAR)
    sit = Situation.of(rig.state, make_observation((25, 4), turn=0, respawn=(25, 4)))
    exp = classify(rig.state, sit)
    kids = Children.concatenate(
        build_children(rig.state, exp, np.array([0]), np.array([int(E)])),
        Children(exp.store.take([0]), np.array([0]), np.array([-1]), np.array([0]), np.array([True])),
    )
    assign_weights(kids, exp, ConstantMotionModel(), BeliefConfig(), 0.95)
    assert kids.store.weight.tolist() == pytest.approx([0.05 * 0.25, 0.95])


# ---- settle: merging, pruning, rehydration ------------------------------------------------------------------------------


def test_pruning_keeps_the_budget_per_position_and_hands_the_rest_to_the_residual():
    from candidate.agent.belief.settle import prune

    config = BeliefConfig(budget=ProportionalBudget(total=10, min_keep=1, max_keep=2))
    rig = Rig(FAR, config)
    entries = [(flat(30, 30), [(30, 29), (30, 30)][: (i % 2) + 1], 0.1 * (i + 1), 9 + i) for i in range(5)]
    entries += [(flat(40, 40), [], 0.05, 9)]
    prime(rig, entries)
    state = rig.state
    store = state.store
    total_before = store.mass_by_cell() + state.residual.mass
    kept = prune(state, store, config, rig.belief.stats)
    after = kept.mass_by_cell() + state.residual.mass
    assert (kept.pos == flat(30, 30)).sum() == 2 and (kept.pos == flat(40, 40)).sum() == 1
    assert np.allclose(total_before, after)                       # the probability of each position is unchanged
    assert sorted(kept.weight[kept.pos == flat(30, 30)]) == pytest.approx(sorted(store.weight[store.pos == flat(30, 30)])[-2:])
    assert state.residual.total() == pytest.approx(float(store.weight.sum() - kept.weight.sum()))


def test_pruning_never_empties_a_position_that_has_hypotheses():
    from candidate.agent.belief.settle import prune

    config = BeliefConfig(budget=ProportionalBudget(total=1, min_keep=1, max_keep=1))
    rig = Rig(FAR, config)
    prime(rig, [(flat(y, 20), [], 1.0, 9) for y in range(10, 30)])
    assert prune(rig.state, rig.state.store, config, rig.belief.stats).n == 20


def test_identical_hypotheses_are_merged_and_their_weights_added():
    from candidate.agent.belief.settle import merge

    rig = Rig(FAR)
    prime(rig, [(flat(30, 30), [], 0.2, 9), (flat(30, 30), [], 0.3, 9), (flat(30, 31), [], 0.5, 9)])
    merged = merge(rig.state, rig.state.store, rig.belief.stats)
    assert merged.n == 2 and rig.belief.stats.merged == 1
    assert merged.weight[merged.pos == flat(30, 30)][0] == pytest.approx(0.5)
    assert merged.weight.sum() == pytest.approx(1.0)


def test_hypotheses_that_differ_in_their_trail_are_not_merged():
    from candidate.agent.belief.settle import merge

    rig = Rig(FAR)
    prime(rig, [(flat(30, 30), [(30, 29), (30, 30)], 0.5, 9), (flat(30, 30), [(31, 30), (30, 30)], 0.5, 9)])
    assert merge(rig.state, rig.state.store, rig.belief.stats).n == 2


def test_the_residual_may_have_died_and_respawned_so_some_of_it_teleports_home():
    config = BeliefConfig(residual_respawn_rate=0.1, rehydrate_below=0.0)
    rig = Rig(FAR, config)
    rig.state.store = HypothesisStore.empty()
    rig.state.residual.mass[10, 10] = 1.0
    rig.state.first = False
    rig.state.opp_count = 9
    view = rig.step(make_observation((25, 4), turn=2, respawn=(25, 4)))
    assert view.probability_at(FAR.opp.respawn) == pytest.approx(0.1)
    assert view.position.sum() == pytest.approx(1.0, abs=1e-5)


def test_when_few_hypotheses_remain_part_of_the_residual_is_given_guessed_histories():
    config = BeliefConfig(rehydrate_below=0.5, rehydrate_fraction=0.5)
    rig = Rig(FAR, config)
    rig.state.store = HypothesisStore.empty()
    rig.state.residual.mass[10, 10] = 0.5
    rig.state.residual.mass[12, 12] = 0.5
    rig.state.first = False
    rig.state.opp_count = 9
    before = rig.belief.stats.rehydrated
    rig.step(make_observation((25, 4), turn=2, respawn=(25, 4)))
    # Hypotheses are placed where the residual is after the opponent's move (it has spread from the two
    # cells it was on), not where it was before.
    assert rig.belief.stats.rehydrated > before + 2
    assert rig.belief.view.n_hypotheses > 0 and rig.belief.view.residual_mass > 0   # the residual keeps the rest
    sound(rig)


def test_rehydration_can_be_switched_off():
    rig = Rig(FAR, BeliefConfig(rehydrate_below=0.0))
    rig.state.store = HypothesisStore.empty()
    rig.state.residual.mass[10, 10] = 1.0
    rig.state.first = False
    rig.state.opp_count = 9
    view = rig.step(make_observation((25, 4), turn=2, respawn=(25, 4)))
    assert view.n_hypotheses == 0 and rig.belief.stats.rehydrated == 0


# ---- the motion model is pluggable ----------------------------------------------------------------------------------------


def test_a_model_that_prefers_a_direction_moves_the_belief_that_way():
    class Eastward(MotionModel):
        def score(self, batch):
            out = np.zeros((batch.n, 4))
            out[:, int(E)] = 6.0
            return out

    plain, eager = Rig(FAR, BeliefConfig(first_update_stay=0.5)), Rig(FAR, BeliefConfig(first_update_stay=0.0), Eastward())
    obs = make_observation((25, 4), turn=0, respawn=(25, 4))
    plain.step(obs)
    view = eager.step(obs)
    east = flat(25, 46)
    assert view.probability_at(east) > 0.9
    assert plain.belief.view.probability_at(east) == pytest.approx(0.125)


def test_the_model_is_shown_every_hypothesis_and_every_move():
    seen = []

    class Recorder(ConstantMotionModel):
        def score(self, batch):
            seen.append(batch)
            return super().score(batch)

    rig = Rig(FAR, motion=Recorder())
    rig.step(make_observation((25, 4), turn=0, respawn=(25, 4)))
    batch = seen[0]
    assert batch.n == 1 and batch.dest.shape == (1, 4) and batch.legal.all()
    assert batch.dest[0].tolist() == [flat(24, 45), flat(25, 46), flat(26, 45), flat(25, 44)]
    assert batch.my_pos == flat(25, 4) and not batch.captures.any() and batch.capture_gain.sum() == 0


# ---- general ------------------------------------------------------------------------------------------------------------------


def test_the_belief_stays_normalised_and_consistent_over_many_updates():
    rig = Rig(FAR)
    for turn in range(30):
        rig.step(make_observation((25, 4), turn=turn, respawn=(25, 4)))
        sound(rig)
    assert rig.belief.stats.updates == 30 and rig.belief.stats.reseeds == 0


def test_published_views_stay_valid_after_the_next_update():
    rig = Rig(FAR)
    first = rig.step(make_observation((25, 4), turn=0, respawn=(25, 4)))
    snapshot = first.position.copy()
    marginal = first.trail_marginal().copy()
    rig.step(make_observation((25, 4), turn=1, respawn=(25, 4)))
    assert np.array_equal(first.position, snapshot) and np.array_equal(first.trail_marginal(), marginal)


def test_two_filters_given_the_same_evidence_agree_exactly():
    a, b = Rig(FAR), Rig(FAR)
    for turn in range(8):
        obs = make_observation((25, 4), turn=turn, respawn=(25, 4))
        va, vb = a.step(obs), b.step(obs)
        assert np.array_equal(va.position, vb.position) and va.n_hypotheses == vb.n_hypotheses


def test_the_hypotheses_never_exceed_the_configured_capacity():
    config = BeliefConfig(capacity=50, budget=ProportionalBudget(total=5000, min_keep=1, max_keep=5000))
    rig = Rig(FAR, config)
    for turn in range(12):
        view = rig.step(make_observation((25, 4), turn=turn, respawn=(25, 4)))
        assert view.n_hypotheses <= 50
    sound(rig)
