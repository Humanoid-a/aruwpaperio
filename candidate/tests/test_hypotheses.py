"""The belief's data structures."""

import numpy as np
import pytest

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.data.belief_view import BeliefView
from candidate.agent.data.hypotheses import HypothesisStore
from candidate.agent.data.residual import ResidualGrid
from candidate.agent.data.territory import TerritoryDiffs
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.agent.geometry.zobrist import DEFAULT_KEYS


def bb(*cells):
    return bitboard.from_cells(flat(y, x) for y, x in cells)


def store_of(entries):
    """entries: (pos, trail cells as (y, x), weight, count); head must be the last trail cell."""
    pos, trail, length, last_dir, weight, count, hash_ = [], [], [], [], [], [], []
    for p, cells, w, c in entries:
        t = bb(*cells)
        pos.append(p)
        trail.append(t)
        length.append(len(cells))
        last_dir.append(0 if cells else -1)
        weight.append(w)
        count.append(c)
        hash_.append(DEFAULT_KEYS.hash_position("opp_pos", p) ^ DEFAULT_KEYS.hash_board("opp_trail", t))
    n = len(entries)
    return HypothesisStore.build(
        pos=pos, last_dir=last_dir, trail_len=length, trail_bb=np.stack(trail), opp_land_id=[0] * n,
        opp_count=count, flags=[0] * n, weight=weight, hash=hash_,
    )


# ---- TerritoryDiffs ----------------------------------------------------------


def test_the_empty_difference_is_id_zero_and_is_never_stored_twice():
    diffs = TerritoryDiffs()
    assert diffs.intern(bitboard.empty(), bitboard.empty()) == 0
    first = diffs.intern(bb((1, 1)), bitboard.empty())
    again = diffs.intern(bb((1, 1)), bitboard.empty())
    assert first == again != 0 and len(diffs) == 2


def test_many_differences_are_deduplicated_in_one_call():
    diffs = TerritoryDiffs()
    added = np.stack([bb((1, 1)), bb((2, 2)), bb((1, 1)), bitboard.empty()])
    removed = np.stack([bitboard.empty(), bitboard.empty(), bitboard.empty(), bitboard.empty()])
    ids = diffs.intern_many(added, removed)
    assert ids[0] == ids[2] and ids[0] != ids[1] and ids[3] == 0 and len(diffs) == 3
    assert diffs.intern_many(added[:0], removed[:0]).size == 0


def test_the_territory_under_an_id_is_the_base_with_the_difference_applied():
    diffs = TerritoryDiffs()
    base = bb((5, 5), (5, 6), (5, 7))
    ident = diffs.intern(bb((9, 9)), bb((5, 6)))
    territory = diffs.territory(np.array([0, ident]), base)
    assert np.array_equal(territory[0], base)
    assert bitboard.to_cells(territory[1]) == sorted([flat(5, 5), flat(5, 7), flat(9, 9)])


def test_the_store_grows_past_its_initial_capacity_and_keeps_every_difference():
    diffs = TerritoryDiffs(capacity=4)
    cells = [(y, x) for y in range(10) for x in range(10)]
    ids = diffs.intern_many(np.stack([bb(c) for c in cells]), np.zeros((len(cells), BOARD_HEIGHT), dtype=np.int64))
    assert len(set(ids.tolist())) == 100 and len(diffs) == 101
    assert np.array_equal(diffs.added[ids[37]], bb(cells[37]))


def test_a_difference_has_a_content_key_independent_of_its_id():
    a, b = TerritoryDiffs(), TerritoryDiffs()
    b.intern(bb((3, 3)), bitboard.empty())            # shifts the ids in b
    ia = a.intern(bb((7, 7)), bb((2, 2)))
    ib = b.intern(bb((7, 7)), bb((2, 2)))
    assert ia != ib and a.key[ia] == b.key[ib] != 0
    assert a.key[0] == 0


# ---- HypothesisStore ----------------------------------------------------------


def test_a_well_formed_store_has_no_problems():
    store = store_of([(flat(10, 10), [], 0.6, 9), (flat(11, 10), [(10, 10), (11, 10)], 0.4, 9)])
    assert store.n == 2 and store.problems(TerritoryDiffs()) == []


def test_take_and_concatenate_return_independent_copies():
    store = store_of([(flat(1, 1), [], 0.5, 9), (flat(2, 2), [], 0.5, 9)])
    one = store.take([1])
    one.weight[0] = 99
    assert store.weight[1] == 0.5 and one.n == 1
    both = HypothesisStore.concatenate([store, one])
    assert both.n == 3 and both.problems() == []


def test_the_empty_store_is_valid():
    empty = HypothesisStore.empty()
    assert empty.n == 0 and empty.problems() == [] and empty.mass_by_cell().sum() == 0


def test_mass_by_cell_adds_the_weights_at_each_position():
    store = store_of([(flat(4, 4), [], 0.25, 9), (flat(4, 4), [], 0.5, 9), (flat(8, 1), [], 0.25, 9)])
    mass = store.mass_by_cell()
    assert mass.shape == (BOARD_HEIGHT, BOARD_WIDTH) and mass[4, 4] == 0.75 and mass[8, 1] == 0.25


def test_frozen_stores_refuse_writes():
    store = store_of([(flat(1, 1), [], 1.0, 9)]).freeze()
    with pytest.raises(ValueError):
        store.weight[0] = 0.5


def test_the_merge_key_separates_states_that_differ_in_anything_that_matters():
    base = store_of([(flat(5, 5), [], 1.0, 9)])
    diffs = TerritoryDiffs()
    same = base.merge_key(diffs, diffs)
    assert (same == base.merge_key(diffs, diffs)).all()
    other_count = store_of([(flat(5, 5), [], 1.0, 10)])
    other_pos = store_of([(flat(5, 6), [], 1.0, 9)])
    other_trail = store_of([(flat(5, 6), [(5, 5), (5, 6)], 1.0, 9)])
    keys = {int(s.merge_key(diffs, diffs)[0]) for s in (base, other_count, other_pos, other_trail)}
    assert len(keys) == 4


@pytest.mark.parametrize(
    "corrupt,message",
    [
        (lambda s: s.weight.__setitem__(0, -1.0), "non-negative"),
        (lambda s: s.weight.__setitem__(0, np.nan), "non-negative"),
        (lambda s: s.pos.__setitem__(0, CELL_COUNT), "off the board"),
        (lambda s: s.trail_len.__setitem__(1, 7), "trail_len"),
        (lambda s: s.last_dir.__setitem__(0, 2), "last_dir"),
        (lambda s: s.opp_count.__setitem__(0, -3), "count"),
        (lambda s: s.hash.__setitem__(0, 12345), "hash"),
        (lambda s: s.opp_land_id.__setitem__(0, 40), "difference id"),
        (lambda s: s.trail_bb.__setitem__((1, 3), np.int64(1) << 55), "beyond the board"),
    ],
)
def test_every_invariant_has_a_failing_case(corrupt, message):
    store = store_of([(flat(10, 10), [], 0.6, 9), (flat(11, 10), [(10, 10), (11, 10)], 0.4, 9)])
    corrupt(store)
    assert any(message in p for p in store.problems(TerritoryDiffs())), store.problems(TerritoryDiffs())


def test_the_trail_must_contain_the_opponents_cell_exactly_when_it_has_one():
    headless = store_of([(flat(11, 10), [(10, 10), (10, 9)], 1.0, 9)])      # trail does not include pos
    assert any("include the opponent's cell" in p for p in headless.problems())
    stray = store_of([(flat(11, 10), [], 1.0, 9)])
    stray.trail_bb[0] = bb((11, 10))
    stray.trail_len[0] = 1
    assert any("include the opponent's cell" in p or "last_dir" in p for p in stray.problems())


# ---- ResidualGrid ---------------------------------------------------------------


def open_board():
    return np.ones((BOARD_HEIGHT, BOARD_WIDTH), dtype=bool)


def test_spreading_conserves_mass_and_the_opponent_always_moves():
    grid = ResidualGrid()
    grid.mass[20, 20] = 1.0
    grid.spread(open_board())
    assert grid.total() == pytest.approx(1.0)
    assert grid.mass[20, 20] == 0
    for cell in [(19, 20), (21, 20), (20, 19), (20, 21)]:
        assert grid.mass[cell] == pytest.approx(0.25)


def test_corner_mass_splits_between_its_only_two_neighbours():
    grid = ResidualGrid()
    grid.mass[0, 0] = 0.6
    grid.spread(open_board())
    assert grid.mass[0, 1] == pytest.approx(0.3) and grid.mass[1, 0] == pytest.approx(0.3)
    assert grid.total() == pytest.approx(0.6)


def test_mass_never_flows_onto_a_cell_that_is_not_a_legal_destination():
    legal = open_board()
    legal[20, 21] = False     # my agent is there
    grid = ResidualGrid()
    grid.mass[20, 20] = 1.0
    grid.spread(legal)
    assert grid.mass[20, 21] == 0 and grid.mass[20, 19] == pytest.approx(1 / 3) and grid.total() == pytest.approx(1.0)


def test_mass_with_nowhere_to_go_stays_put():
    legal = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=bool)
    grid = ResidualGrid()
    grid.mass[5, 5] = 0.4
    grid.spread(legal)
    assert grid.mass[5, 5] == pytest.approx(0.4)


def test_spreading_matches_a_slow_reference_on_random_mass():
    rng = np.random.default_rng(4)
    legal = rng.random((BOARD_HEIGHT, BOARD_WIDTH)) < 0.9
    mass = rng.random((BOARD_HEIGHT, BOARD_WIDTH))
    expected = np.zeros_like(mass)
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            targets = [(y + dy, x + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                       if 0 <= y + dy < BOARD_HEIGHT and 0 <= x + dx < BOARD_WIDTH and legal[y + dy, x + dx]]
            if not targets:
                expected[y, x] += mass[y, x]
            for t in targets:
                expected[t] += mass[y, x] / len(targets)
    grid = ResidualGrid()
    grid.mass = mass.copy()
    grid.spread(legal)
    assert np.allclose(grid.mass, expected)


def test_a_walk_has_checkerboard_parity_but_a_teleport_breaks_it():
    grid = ResidualGrid()
    grid.mass[10, 10] = 1.0
    for _ in range(3):
        grid.spread(open_board())
    ys, xs = np.nonzero(grid.mass > 0)
    assert {(y + x) % 2 for y, x in zip(ys, xs)} == {(10 + 10 + 3) % 2}
    grid.teleport(flat(25, 31), 0.1)
    assert grid.mass[25, 31] == pytest.approx(0.1) and grid.total() == pytest.approx(1.0)


# ---- BeliefView -------------------------------------------------------------------


def make_view(store, residual=None):
    mass = store.mass_by_cell() + (residual if residual is not None else 0.0)
    total = mass.sum()
    p = mass / total
    return BeliefView(
        turn=3, position=p.astype(np.float32), support=p > 0, n_hypotheses=store.n,
        hypothesis_mass=float(store.weight.sum() / total), residual_mass=float(1 - store.weight.sum() / total),
        entropy=0.0, sighted_at=None, events=(), _pos=store.pos, _weight=store.weight,
        _trail_bb=store.trail_bb, _trail_len=store.trail_len, _opp_count=store.opp_count,
    )


def test_the_view_answers_position_questions():
    store = store_of([(flat(4, 4), [], 0.5, 9), (flat(5, 5), [(5, 4), (5, 5)], 0.5, 9)])
    view = make_view(store)
    assert view.probability_at(flat(4, 4)) == pytest.approx(0.5)
    assert view.most_likely()[1] == pytest.approx(0.5)
    assert view.support.sum() == 2


def test_the_trail_marginal_weights_each_cell_by_the_hypotheses_that_cover_it():
    store = store_of([(flat(5, 5), [(5, 4), (5, 5)], 0.25, 9), (flat(6, 5), [(5, 5), (6, 5)], 0.75, 9)])
    marginal = make_view(store).trail_marginal()
    assert marginal[5, 4] == pytest.approx(0.25) and marginal[6, 5] == pytest.approx(0.75)
    assert marginal[5, 5] == pytest.approx(1.0) and marginal[0, 0] == 0


def test_exposure_and_count_distribution_summarise_the_hypotheses():
    store = store_of([(flat(4, 4), [], 0.25, 9), (flat(5, 5), [(5, 4), (5, 5)], 0.75, 13)])
    view = make_view(store)
    assert view.exposed_probability() == pytest.approx(0.75)
    assert view.count_distribution() == {9: pytest.approx(0.25), 13: pytest.approx(0.75)}


def test_a_view_with_no_hypotheses_says_nothing_about_trails():
    empty = HypothesisStore.empty()
    residual = np.zeros((BOARD_HEIGHT, BOARD_WIDTH))
    residual[3, 3] = 1.0
    view = make_view(empty, residual)
    assert view.exposed_probability() is None and view.trail_marginal().sum() == 0
    assert view.probability_at(flat(3, 3)) == pytest.approx(1.0)
