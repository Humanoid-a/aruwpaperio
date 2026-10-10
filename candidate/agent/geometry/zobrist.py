"""Zobrist hashing for board states, so duplicate states can be merged by a
dictionary lookup or np.unique on one int64 per state.

Every (role, cell) pair gets a fixed random 64-bit key. A state's hash is the
XOR of the keys of everything true about it, which makes updates incremental:
moving an agent XORs out the old position key and XORs in the new one, and
adding a trail cell XORs in one key. At 64 bits an accidental collision is
rare enough to accept (about 1 in 2**32 per pair of states compared).

Keys come from a seeded generator, so hashes are reproducible across runs.
Hashes are int64 (negative values are fine; they are just bit patterns).
"""

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.geometry import bitboard

ROLES = ("my_pos", "opp_pos", "my_territory", "opp_territory", "my_trail", "opp_trail", "terr_added", "terr_removed")
_BOARD_ROLES = ("my_territory", "opp_territory", "my_trail", "opp_trail", "terr_added", "terr_removed")
_CHUNK_BITS = 8
_CHUNKS = -(-BOARD_WIDTH // _CHUNK_BITS)  # ceil: 7 chunks of 8 bits cover 50 columns


class ZobristKeys:
    """Per-role random keys plus per-row byte tables for fast board hashing."""

    def __init__(self, seed: int = 20261008) -> None:
        rng = np.random.default_rng(seed)
        self._keys = {
            role: rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=CELL_COUNT, dtype=np.int64)
            for role in ROLES
        }
        self._tables = {role: self._build_table(self._keys[role]) for role in _BOARD_ROLES}
        # Keys for small integers that are part of a state's identity (see belief merging).
        self.count_keys = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=CELL_COUNT + 1, dtype=np.int64)
        self.dir_keys = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=5, dtype=np.int64)  # last_dir -1..3

    @staticmethod
    def _build_table(keys: np.ndarray) -> np.ndarray:
        """table[y, k, b] = XOR of the keys of the set bits of byte value b in
        chunk k of row y, so a row hashes with one gather per chunk."""
        grid = keys.reshape(BOARD_HEIGHT, BOARD_WIDTH)
        table = np.zeros((BOARD_HEIGHT, _CHUNKS, 1 << _CHUNK_BITS), dtype=np.int64)
        for k in range(_CHUNKS):
            for byte in range(1, 1 << _CHUNK_BITS):
                x = k * _CHUNK_BITS + (byte & -byte).bit_length() - 1  # column of the lowest set bit
                key_column = grid[:, x] if x < BOARD_WIDTH else 0      # bits past the board never occur
                table[:, k, byte] = table[:, k, byte & (byte - 1)] ^ key_column
        return table

    def key(self, role: str, cell: int) -> np.int64:
        """The key for one (role, cell). XOR it in or out to update a hash."""
        return self._keys[role][cell]

    def hash_board(self, role: str, bb: np.ndarray) -> np.ndarray | np.int64:
        """XOR of the keys of every cell set in the bitboard(s): an int64 for a
        single board, an array over the leading dimensions for a batch."""
        table = self._tables[role]
        bb = bitboard.as_board(bb)
        rows = np.arange(BOARD_HEIGHT)
        out = np.zeros(bb.shape[:-1], dtype=np.int64)
        for k in range(_CHUNKS):
            chunk = (bb >> (k * _CHUNK_BITS)) & ((1 << _CHUNK_BITS) - 1)
            out ^= np.bitwise_xor.reduce(table[rows, k, chunk], axis=-1)
        return out if bb.ndim > 1 else out[()]

    def hash_position(self, role: str, pos: np.ndarray | int) -> np.ndarray | np.int64:
        """Key(s) for agent position(s) given as flat indices."""
        return self._keys[role][pos]

    def hash_state(
        self,
        my_pos,
        opp_pos,
        my_territory: np.ndarray,
        opp_territory: np.ndarray,
        my_trail: np.ndarray,
        opp_trail: np.ndarray,
    ):
        """The full hash of a state (or a batch of states). Equal states hash
        equal; the hash is unaffected by the order of the trail cells."""
        return (
            self.hash_position("my_pos", my_pos)
            ^ self.hash_position("opp_pos", opp_pos)
            ^ self.hash_board("my_territory", my_territory)
            ^ self.hash_board("opp_territory", opp_territory)
            ^ self.hash_board("my_trail", my_trail)
            ^ self.hash_board("opp_trail", opp_trail)
        )


DEFAULT_KEYS = ZobristKeys()
