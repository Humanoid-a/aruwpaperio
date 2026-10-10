from collections import deque

import numpy as np
import pytest

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.fill import BOARD, EDGE, capture_region, dilate, flood, holes
from candidate.agent.geometry.grid import flat
from candidate.tests.builders import square

RNG = np.random.default_rng(2026)


def bb_of(cells) -> np.ndarray:
    return bitboard.from_cells(flat(y, x) for y, x in cells)


def cells_of(bb) -> set[tuple[int, int]]:
    return {divmod(c, BOARD_WIDTH) for c in bitboard.to_cells(bb)}


def random_mask(density: float) -> np.ndarray:
    return RNG.random((BOARD_HEIGHT, BOARD_WIDTH)) < density


# ---- independent reference implementations (plain BFS on bool arrays) ----


def ref_flood(seed: np.ndarray, passable: np.ndarray) -> np.ndarray:
    reached = np.zeros_like(passable)
    queue = deque()
    for y, x in zip(*np.nonzero(seed & passable)):
        reached[y, x] = True
        queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < BOARD_HEIGHT and 0 <= nx < BOARD_WIDTH and passable[ny, nx] and not reached[ny, nx]:
                reached[ny, nx] = True
                queue.append((ny, nx))
    return reached


def ref_holes(blocked: np.ndarray) -> np.ndarray:
    passable = ~blocked
    border = np.zeros_like(blocked)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    return passable & ~ref_flood(border, passable)


# ---- constants ----------------------------------------------------------


def test_edge_is_exactly_the_border_cells():
    mask = bitboard.to_mask(EDGE)
    assert mask.sum() == 2 * BOARD_WIDTH + 2 * (BOARD_HEIGHT - 2)
    assert mask[0].all() and mask[-1].all() and mask[:, 0].all() and mask[:, -1].all()
    assert not mask[1:-1, 1:-1].any()
    assert bitboard.is_valid(EDGE) and bitboard.is_valid(BOARD)


# ---- dilate -------------------------------------------------------------


def test_dilate_grows_one_step_in_four_directions():
    assert cells_of(dilate(bb_of([(10, 10)]))) == {(10, 10), (9, 10), (11, 10), (10, 9), (10, 11)}


def test_dilate_is_clipped_at_corners_and_never_wraps_rows():
    assert cells_of(dilate(bb_of([(0, 0)]))) == {(0, 0), (0, 1), (1, 0)}
    assert cells_of(dilate(bb_of([(3, BOARD_WIDTH - 1)]))) == {
        (3, BOARD_WIDTH - 1), (3, BOARD_WIDTH - 2), (2, BOARD_WIDTH - 1), (4, BOARD_WIDTH - 1),
    }
    assert cells_of(dilate(bb_of([(BOARD_HEIGHT - 1, BOARD_WIDTH - 1)]))) == {
        (BOARD_HEIGHT - 1, BOARD_WIDTH - 1), (BOARD_HEIGHT - 2, BOARD_WIDTH - 1), (BOARD_HEIGHT - 1, BOARD_WIDTH - 2),
    }


def test_dilate_never_sets_bits_beyond_the_board():
    grown = dilate(bitboard.full())
    assert bitboard.is_valid(grown) and np.array_equal(grown, bitboard.full())


def test_dilate_respects_passable_and_does_not_mutate_its_input():
    seed = bb_of([(10, 10)])
    original = seed.copy()
    passable = bb_of([(10, 11), (9, 10), (20, 20)])
    assert cells_of(dilate(seed, passable)) == {(10, 11), (9, 10)}
    assert np.array_equal(seed, original)


def test_dilate_batch_matches_single():
    masks = [random_mask(0.05) for _ in range(6)]
    batch = bitboard.from_mask(np.stack(masks))
    grown = dilate(batch)
    for i, mask in enumerate(masks):
        assert np.array_equal(grown[i], dilate(bitboard.from_mask(mask)))


# ---- flood --------------------------------------------------------------


@pytest.mark.parametrize("density", [0.3, 0.5, 0.6, 0.7])
def test_flood_matches_reference_bfs_on_random_boards(density):
    for _ in range(8):
        passable = random_mask(density)
        seed = random_mask(0.01)
        got = bitboard.to_mask(flood(bitboard.from_mask(seed), bitboard.from_mask(passable)))
        assert np.array_equal(got, ref_flood(seed, passable))


def test_flood_ignores_seeds_outside_passable_and_empty_seeds():
    passable = bb_of(square(10, 10, 2))
    assert bitboard.popcount(flood(bb_of([(0, 0)]), passable)) == 0
    assert bitboard.popcount(flood(bitboard.empty(), passable)) == 0
    assert cells_of(flood(bb_of([(10, 10)]), passable)) == cells_of(passable)


def test_flood_does_not_cross_walls_or_jump_diagonally():
    passable = bb_of([(5, 5), (6, 6)])  # diagonal neighbours only
    assert cells_of(flood(bb_of([(5, 5)]), passable)) == {(5, 5)}


def test_flood_handles_a_long_winding_corridor():
    snake = set()
    for y in range(0, 20, 2):
        snake |= {(y, x) for x in range(30)}
        if y + 1 < 20:
            snake.add((y + 1, 29 if (y // 2) % 2 == 0 else 0))
    passable = bb_of(snake)
    assert cells_of(flood(bb_of([(0, 0)]), passable)) == snake


def test_flood_batch_matches_single():
    passables = [random_mask(0.55) for _ in range(5)]
    seeds = [random_mask(0.01) for _ in range(5)]
    batch = flood(bitboard.from_mask(np.stack(seeds)), bitboard.from_mask(np.stack(passables)))
    for i in range(5):
        assert np.array_equal(batch[i], flood(bitboard.from_mask(seeds[i]), bitboard.from_mask(passables[i])))


# ---- holes --------------------------------------------------------------


def ring(cy: int, cx: int) -> set[tuple[int, int]]:
    return set(square(cy, cx, 1)) - {(cy, cx)}


def test_a_closed_ring_encloses_its_centre():
    assert cells_of(holes(bb_of(ring(10, 10)))) == {(10, 10)}


def test_a_ring_with_a_gap_encloses_nothing():
    gapped = ring(10, 10) - {(9, 10)}
    assert cells_of(holes(bb_of(gapped))) == set()


def test_four_cells_in_a_diamond_seal_their_centre_under_4_connectivity():
    # Diagonal contact between the blockers is enough: movement is 4-connected.
    diamond = {(9, 10), (10, 9), (10, 11), (11, 10)}
    assert cells_of(holes(bb_of(diamond))) == {(10, 10)}


def test_free_space_touching_the_edge_is_never_a_hole():
    wall = {(y, 5) for y in range(BOARD_HEIGHT)}
    assert cells_of(holes(bb_of(wall))) == set()  # a full-height wall splits the board; both sides touch the border


def test_empty_and_full_boards_have_no_holes():
    assert bitboard.popcount(holes(bitboard.empty())) == 0
    assert bitboard.popcount(holes(bitboard.full())) == 0


def test_border_cells_are_never_holes_even_when_walled_in():
    # (0, 0) is cut off from the rest of the board but sits on the border itself.
    assert cells_of(holes(bb_of({(0, 1), (1, 0)}))) == set()


@pytest.mark.parametrize("density", [0.35, 0.5, 0.6])
def test_holes_match_reference_on_random_boards(density):
    for _ in range(8):
        blocked = random_mask(density)
        assert np.array_equal(bitboard.to_mask(holes(bitboard.from_mask(blocked))), ref_holes(blocked))


def test_holes_batch_matches_single():
    masks = np.stack([random_mask(0.5) for _ in range(6)])
    batch = holes(bitboard.from_mask(masks))
    for i in range(6):
        assert np.array_equal(batch[i], holes(bitboard.from_mask(masks[i])))


# ---- capture_region -----------------------------------------------------


def wall_and_loop():
    """A vertical wall of territory with a trail that leaves it at the top,
    runs around two columns and nine rows of free land, and returns at the bottom."""
    territory = {(y, 5) for y in range(20, 31)}
    trail = {(20, 6), (20, 7), (20, 8), (30, 6), (30, 7), (30, 8)} | {(y, 8) for y in range(21, 30)}
    interior = {(y, x) for y in range(21, 30) for x in (6, 7)}
    return territory, trail, interior


def test_closing_a_loop_captures_the_enclosed_land_but_not_the_trail_cells():
    territory, trail, interior = wall_and_loop()
    region = capture_region(bb_of(territory), bb_of(trail))
    assert cells_of(region) == interior
    gain = bitboard.popcount(bb_of(trail) & ~bb_of(territory)) + bitboard.popcount(region)
    assert gain == len(trail) + len(interior) == 15 + 18


def test_an_open_loop_captures_nothing():
    territory, trail, _ = wall_and_loop()
    trail = trail - {(25, 8)}  # a gap in the far side
    assert bitboard.popcount(capture_region(bb_of(territory), bb_of(trail))) == 0


def test_a_hole_that_already_existed_is_not_captured_by_an_unrelated_loop():
    territory, trail, interior = wall_and_loop()
    pocket = ring(40, 40)  # a ring of territory with one free cell inside it
    territory = territory | pocket
    assert (40, 40) in cells_of(holes(bb_of(territory)))
    region = capture_region(bb_of(territory), bb_of(trail))
    assert cells_of(region) == interior  # (40, 40) stays out
    assert (40, 40) in cells_of(holes(bb_of(territory | trail)))


def test_enemy_land_inside_the_loop_is_part_of_the_captured_region():
    # capture_region only knows the capturer's cells, so land held by anyone else
    # (or by no one) inside the loop is captured alike.
    territory, trail, interior = wall_and_loop()
    assert (25, 6) in interior
    assert (25, 6) in cells_of(capture_region(bb_of(territory), bb_of(trail)))


def test_a_trail_that_seals_an_open_channel_captures_it():
    # A U of territory open to the east: the trail closes the open side, so the
    # enclosed region did not exist before and is captured.
    territory = {(10, x) for x in range(10, 16)} | {(14, x) for x in range(10, 16)} | {(y, 10) for y in range(10, 15)}
    trail = {(y, 15) for y in range(11, 14)}
    assert not cells_of(holes(bb_of(territory)))
    region = capture_region(bb_of(territory), bb_of(trail))
    assert cells_of(region) == {(y, x) for y in range(11, 14) for x in range(11, 15)}


def test_capture_region_batch_matches_single():
    territory, trail, interior = wall_and_loop()
    t1, t2 = bb_of(territory), bb_of(trail)
    batch = capture_region(np.stack([t1, t1]), np.stack([t2, bitboard.empty()]))
    assert cells_of(batch[0]) == interior
    assert bitboard.popcount(batch[1]) == 0
