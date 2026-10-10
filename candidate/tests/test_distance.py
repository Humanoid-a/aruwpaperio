from collections import deque

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.distance import UNREACHABLE, bfs_distance, shortest_path, taxicab_distance
from candidate.agent.geometry.fill import BOARD
from candidate.agent.geometry.grid import flat, manhattan, neighbors

RNG = np.random.default_rng(77)


def bb_of(cells) -> np.ndarray:
    return bitboard.from_cells(flat(y, x) for y, x in cells)


def random_mask(density: float) -> np.ndarray:
    return RNG.random((BOARD_HEIGHT, BOARD_WIDTH)) < density


def ref_bfs(sources: np.ndarray, passable: np.ndarray) -> np.ndarray:
    dist = np.full((BOARD_HEIGHT, BOARD_WIDTH), UNREACHABLE, dtype=np.int16)
    queue = deque()
    for y, x in zip(*np.nonzero(sources)):
        dist[y, x] = 0
        queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < BOARD_HEIGHT and 0 <= nx < BOARD_WIDTH and passable[ny, nx] and dist[ny, nx] == UNREACHABLE:
                dist[ny, nx] = dist[y, x] + 1
                queue.append((ny, nx))
    return dist


# ---- bfs_distance -------------------------------------------------------


def test_open_board_distance_is_manhattan_distance():
    dist = bfs_distance(bb_of([(10, 20)]), BOARD)
    assert dist.dtype == np.int16
    for y, x in [(10, 20), (0, 0), (49, 49), (30, 5)]:
        assert dist[y, x] == abs(y - 10) + abs(x - 20)


def test_distance_matches_reference_bfs_with_random_walls():
    for density in (0.2, 0.4, 0.55):
        for _ in range(5):
            passable = random_mask(1 - density)
            sources = random_mask(0.004)
            got = bfs_distance(bitboard.from_mask(sources), bitboard.from_mask(passable))
            assert np.array_equal(got, ref_bfs(sources, passable))


def test_a_wall_cuts_off_cells_and_is_never_entered():
    wall = {(y, 5) for y in range(BOARD_HEIGHT)}
    passable = BOARD & ~bb_of(wall)
    dist = bfs_distance(bb_of([(10, 2)]), passable)
    assert dist[10, 2] == 0 and dist[10, 4] == 2
    assert dist[10, 5] == UNREACHABLE  # a wall cell is never entered
    assert dist[10, 6] == UNREACHABLE  # and the far side is cut off


def test_a_source_on_a_wall_is_distance_zero_and_can_step_off_it():
    wall = {(y, 5) for y in range(BOARD_HEIGHT)}
    passable = BOARD & ~bb_of(wall)
    dist = bfs_distance(bb_of([(20, 5)]), passable)
    assert dist[20, 5] == 0
    assert dist[20, 4] == 1 and dist[20, 6] == 1 and dist[10, 6] == 11
    assert dist[19, 5] == UNREACHABLE  # the rest of the wall still blocks


def test_empty_sources_reach_nothing():
    assert (bfs_distance(bitboard.empty(), BOARD) == UNREACHABLE).all()


# ---- shortest_path ------------------------------------------------------


def test_path_is_empty_when_already_on_a_target():
    assert shortest_path(flat(5, 5), bb_of([(5, 5)]), BOARD) == []


def test_path_steps_onto_cells_one_by_one_and_ends_on_a_target():
    path = shortest_path(flat(10, 10), bb_of([(10, 14)]), BOARD)
    assert path == [flat(10, x) for x in (11, 12, 13, 14)]


def test_path_ties_are_broken_north_first():
    path = shortest_path(flat(10, 10), bb_of([(8, 8)]), BOARD)
    assert len(path) == 4 and path[0] == flat(9, 10)


def test_path_goes_around_a_wall_and_never_touches_it():
    wall = {(y, 12) for y in range(5, 16)}
    passable = BOARD & ~bb_of(wall)
    path = shortest_path(flat(10, 10), bb_of([(10, 14)]), passable)
    assert path is not None
    assert not any(divmod(c, BOARD_WIDTH) in wall for c in path)
    steps = [flat(10, 10)] + path
    assert all(manhattan(a, b) == 1 for a, b in zip(steps, steps[1:]))
    assert len(path) == 4 + 2 * 6  # detour over the top: 6 up, across, 6 back down
    assert path[-1] == flat(10, 14)


def test_path_may_start_from_a_blocked_cell_like_an_exposed_head():
    # The head stands on its own trail (blocked), yet can still leave it.
    trail = {(10, 10), (10, 9), (10, 8)}
    passable = BOARD & ~bb_of(trail)
    target = bb_of([(10, 12)])
    assert shortest_path(flat(10, 10), target, passable) == [flat(10, 11), flat(10, 12)]


def test_path_is_none_when_the_target_is_walled_off():
    box = {(9, 9), (9, 10), (9, 11), (10, 9), (10, 11), (11, 9), (11, 10), (11, 11)}
    passable = BOARD & ~bb_of(box)
    assert shortest_path(flat(0, 0), bb_of([(10, 10)]), passable) is None


def test_path_length_matches_the_distance_map_on_random_boards():
    for _ in range(10):
        passable = random_mask(0.8)
        targets = random_mask(0.005)
        target_bb, passable_bb = bitboard.from_mask(targets), bitboard.from_mask(passable)
        dist = ref_bfs(targets, passable)
        for _ in range(5):
            start = int(RNG.integers(0, BOARD_HEIGHT * BOARD_WIDTH))
            path = shortest_path(start, target_bb, passable_bb)
            y, x = divmod(start, BOARD_WIDTH)
            if targets[y, x]:
                assert path == []
                continue
            best = min((int(dist.flat[n]) for n in neighbors(start)), default=UNREACHABLE)
            if best >= UNREACHABLE:
                assert path is None
            else:
                assert path is not None and len(path) == best + 1
                assert targets.flat[path[-1]] and all(passable.flat[c] for c in path[:-1])


# ---- taxicab_distance ---------------------------------------------------


def test_taxicab_from_one_source_is_manhattan_distance():
    dist = taxicab_distance(bb_of([(20, 30)]))
    assert dist.shape == (BOARD_HEIGHT, BOARD_WIDTH) and dist.dtype == np.int16
    ys, xs = np.mgrid[:BOARD_HEIGHT, :BOARD_WIDTH]
    assert np.array_equal(dist, np.abs(ys - 20) + np.abs(xs - 30))


def test_taxicab_from_many_sources_is_the_minimum_over_sources():
    sources = [(5, 5), (40, 12), (22, 47), (0, 49)]
    dist = taxicab_distance(bb_of(sources))
    ys, xs = np.mgrid[:BOARD_HEIGHT, :BOARD_WIDTH]
    brute = np.min([np.abs(ys - sy) + np.abs(xs - sx) for sy, sx in sources], axis=0)
    assert np.array_equal(dist, brute)


def test_taxicab_with_no_sources_is_unreachable_everywhere():
    assert (taxicab_distance(bitboard.empty()) == UNREACHABLE).all()


def test_taxicab_batch_matches_single_and_agrees_with_open_board_bfs():
    masks = [random_mask(0.003) for _ in range(5)]
    batch = taxicab_distance(bitboard.from_mask(np.stack(masks)))
    assert batch.shape == (5, BOARD_HEIGHT, BOARD_WIDTH)
    for i, mask in enumerate(masks):
        single = taxicab_distance(bitboard.from_mask(mask))
        assert np.array_equal(batch[i], single)
        assert np.array_equal(single, bfs_distance(bitboard.from_mask(mask), BOARD))


def test_taxicab_is_a_lower_bound_on_obstructed_distance():
    passable = random_mask(0.7)
    sources = random_mask(0.004)
    obstructed = ref_bfs(sources, passable)
    free = taxicab_distance(bitboard.from_mask(sources))
    reachable = obstructed < UNREACHABLE
    assert (free[reachable] <= obstructed[reachable]).all()
