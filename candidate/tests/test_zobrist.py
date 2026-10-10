import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.agent.geometry.zobrist import DEFAULT_KEYS, ROLES, ZobristKeys

RNG = np.random.default_rng(5)


def random_board(density: float = 0.3) -> np.ndarray:
    return bitboard.from_mask(RNG.random((BOARD_HEIGHT, BOARD_WIDTH)) < density)


def brute_hash(keys: ZobristKeys, role: str, bb: np.ndarray) -> np.int64:
    h = np.int64(0)
    for c in bitboard.to_cells(bb):
        h ^= keys.key(role, c)
    return h


def test_keys_are_reproducible_per_seed_and_differ_between_seeds():
    a, b, c = ZobristKeys(1), ZobristKeys(1), ZobristKeys(2)
    assert a.key("my_trail", 123) == b.key("my_trail", 123)
    assert a.key("my_trail", 123) != c.key("my_trail", 123)


def test_every_role_and_cell_has_its_own_key():
    flat_keys = np.concatenate([DEFAULT_KEYS.hash_position(role, np.arange(BOARD_HEIGHT * BOARD_WIDTH)) for role in ROLES])
    assert len(set(flat_keys.tolist())) == len(flat_keys)


def test_empty_board_hashes_to_zero():
    for role in ("my_territory", "opp_trail"):
        assert DEFAULT_KEYS.hash_board(role, bitboard.empty()) == 0


def test_board_hash_equals_the_xor_of_the_cell_keys():
    for role in ("my_territory", "opp_territory", "my_trail", "opp_trail"):
        for _ in range(5):
            bb = random_board()
            assert DEFAULT_KEYS.hash_board(role, bb) == brute_hash(DEFAULT_KEYS, role, bb)


def test_batch_hash_matches_single_boards():
    boards = np.stack([random_board() for _ in range(8)])
    batch = DEFAULT_KEYS.hash_board("my_trail", boards)
    assert batch.shape == (8,)
    for i in range(8):
        assert batch[i] == DEFAULT_KEYS.hash_board("my_trail", boards[i])


def test_the_last_column_and_last_row_are_hashed():
    for y, x in [(0, 0), (0, BOARD_WIDTH - 1), (BOARD_HEIGHT - 1, 0), (BOARD_HEIGHT - 1, BOARD_WIDTH - 1)]:
        bb = bitboard.from_cells([flat(y, x)])
        assert DEFAULT_KEYS.hash_board("my_trail", bb) == DEFAULT_KEYS.key("my_trail", flat(y, x))


def test_updates_are_incremental_and_reversible():
    bb = random_board()
    h = DEFAULT_KEYS.hash_board("my_trail", bb)
    cell = flat(17, 31)
    had = bitboard.has_cell(bb, cell)
    bitboard.set_cell(bb, cell)
    if not had:
        assert DEFAULT_KEYS.hash_board("my_trail", bb) == h ^ DEFAULT_KEYS.key("my_trail", cell)
    bitboard.clear_cell(bb, cell)
    h_cleared = DEFAULT_KEYS.hash_board("my_trail", bb)
    assert h_cleared == (h ^ DEFAULT_KEYS.key("my_trail", cell) if had else h)


def test_distinct_random_boards_do_not_collide():
    hashes = {int(DEFAULT_KEYS.hash_board("my_territory", random_board())) for _ in range(3000)}
    assert len(hashes) == 3000


def test_role_matters_the_same_cells_hash_differently_as_territory_and_trail():
    bb = random_board()
    assert DEFAULT_KEYS.hash_board("my_territory", bb) != DEFAULT_KEYS.hash_board("my_trail", bb)


def test_state_hash_is_the_xor_of_its_parts_and_moves_incrementally():
    terr, opp_terr, trail, opp_trail = (random_board(0.1) for _ in range(4))
    h = DEFAULT_KEYS.hash_state(flat(5, 5), flat(40, 40), terr, opp_terr, trail, opp_trail)
    moved = DEFAULT_KEYS.hash_state(flat(5, 6), flat(40, 40), terr, opp_terr, trail, opp_trail)
    assert moved == h ^ DEFAULT_KEYS.key("my_pos", flat(5, 5)) ^ DEFAULT_KEYS.key("my_pos", flat(5, 6))


def test_state_hash_is_vectorised_over_positions_and_boards():
    n = 6
    boards = [np.stack([random_board(0.1) for _ in range(n)]) for _ in range(4)]
    my_pos = RNG.integers(0, 2500, n)
    opp_pos = RNG.integers(0, 2500, n)
    batch = DEFAULT_KEYS.hash_state(my_pos, opp_pos, *boards)
    for i in range(n):
        assert batch[i] == DEFAULT_KEYS.hash_state(my_pos[i], opp_pos[i], *(b[i] for b in boards))
