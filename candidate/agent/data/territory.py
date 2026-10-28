"""A player's territory as a small difference from a shared map.

Every hypothesis has its own idea of what the opponent owns, and of what I own
(the opponent's captures take land from me, and memory cannot know which cells
for land that is out of view). Storing a 50-word bitboard per hypothesis for each
would be wasteful when almost all of them agree with the map memory already has,
so a hypothesis stores only an id into a table of differences:

    territory under id  =  (base & ~removed[id]) | added[id]

`base` is the set of cells memory currently believes the player owns, the *frame*
the table is expressed in. Id 0 is the empty difference (exactly the base). The
table is deduplicated: hypotheses with the same territory share an id, which also
makes the id's content key part of what makes two hypotheses the same.

Differences are re-expressed against the new base after every window update (see
belief.land), so cells the window has since revealed drop out and the table
stays small.
"""

import numpy as np

from candidate.agent.config import BOARD_HEIGHT
from candidate.agent.geometry.zobrist import DEFAULT_KEYS, ZobristKeys

# Odd 64-bit multipliers, one per word of an (added, removed) row: a row's hash is the wrapping sum of word x multiplier.
_ROW_MIX = np.random.default_rng(0x7E11).integers(1, 2**63, size=2 * BOARD_HEIGHT, dtype=np.uint64) | np.uint64(1)


def distinct_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(unique rows, inverse) for a (k, m) int64 array, like np.unique(rows, axis=0, return_inverse=True) but in the
    order the rows first appear, and much faster: each row is hashed to one 64-bit number and only those are sorted
    (np.unique with an axis sorts whole rows field by field, which dominated the belief's time). Two different rows
    with the same hash would be merged, so the result is checked against the rows and, in that (astronomically
    unlikely) case, computed the slow way."""
    if rows.shape[0] == 0:
        return rows, np.zeros(0, dtype=np.int64)
    h = (rows.view(np.uint64) * _ROW_MIX[: rows.shape[1]]).sum(axis=1, dtype=np.uint64)
    _, first, inverse = np.unique(h, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")                    # unique rows in order of first appearance
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    unique, inverse = rows[first[order]], rank[inverse.reshape(-1)]
    if not np.array_equal(unique[inverse], rows):               # a hash collision: fall back on exact sorting
        unique, inverse = np.unique(rows, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
    return unique, inverse


class TerritoryDiffs:
    """Deduplicated (added, removed) bitboard pairs. Id 0 means no difference."""

    def __init__(self, keys: ZobristKeys = DEFAULT_KEYS, capacity: int = 64) -> None:
        self._keys = keys
        self.added = np.zeros((capacity, BOARD_HEIGHT), dtype=np.int64)
        self.removed = np.zeros((capacity, BOARD_HEIGHT), dtype=np.int64)
        self.key = np.zeros(capacity, dtype=np.int64)  # Zobrist key of the difference's content
        self.n = 1                                      # id 0 is the empty difference
        self._index: dict[bytes, int] = {}

    def __len__(self) -> int:
        return self.n

    def _add_many(self, added: np.ndarray, removed: np.ndarray) -> np.ndarray:
        """Store k new differences; returns their ids. The keys are hashed in one call."""
        k = added.shape[0]
        if self.n + k > self.added.shape[0]:
            grown = max(self.n + k, self.n * 2)
            for name in ("added", "removed"):
                bigger = np.zeros((grown, BOARD_HEIGHT), dtype=np.int64)
                bigger[: self.n] = getattr(self, name)[: self.n]
                setattr(self, name, bigger)
            key = np.zeros(grown, dtype=np.int64)
            key[: self.n] = self.key[: self.n]
            self.key = key
        ids = np.arange(self.n, self.n + k)
        self.added[ids], self.removed[ids] = added, removed
        self.key[ids] = self._keys.hash_board("terr_added", added) ^ self._keys.hash_board("terr_removed", removed)
        self.n += k
        return ids

    def intern_many(self, added: np.ndarray, removed: np.ndarray) -> np.ndarray:
        """Ids for k differences given as (k, 50) bitboards, creating the ones not seen before."""
        added, removed = np.asarray(added, dtype=np.int64), np.asarray(removed, dtype=np.int64)
        if added.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        rows = np.ascontiguousarray(np.concatenate([added, removed], axis=1))
        unique, inverse = distinct_rows(rows)
        ids = np.zeros(len(unique), dtype=np.int64)
        fresh: list[int] = []
        for i, row in enumerate(unique):
            if not row.any():
                continue  # id 0
            found = self._index.get(row.tobytes())
            if found is None:
                fresh.append(i)
            else:
                ids[i] = found
        if fresh:
            # new ids are handed out in the rows' sorted order (as np.unique(axis=0) would list them), so the ids, and
            # everything that later depends on their order, do not depend on how the rows were found
            fresh = [fresh[j] for j in np.lexsort(unique[fresh].T[::-1])]
            new_ids = self._add_many(unique[fresh, :BOARD_HEIGHT], unique[fresh, BOARD_HEIGHT:])
            for i, new_id in zip(fresh, new_ids):
                self._index[unique[i].tobytes()] = int(new_id)
                ids[i] = new_id
        return ids[inverse.reshape(-1)]

    def intern(self, added: np.ndarray, removed: np.ndarray) -> int:
        return int(self.intern_many(added[None], removed[None])[0])

    def intern_territory(self, territory: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Ids for k whole territories given as (k, 50) bitboards (or one (50,)), expressed against `base`."""
        territory = np.atleast_2d(np.asarray(territory, dtype=np.int64))
        return self.intern_many(territory & ~base[None, :], base[None, :] & ~territory)

    def territory(self, ids: np.ndarray, base: np.ndarray) -> np.ndarray:
        """The territory under each id, as (len(ids), 50) bitboards."""
        ids = np.asarray(ids)
        return (base[None, :] & ~self.removed[ids]) | self.added[ids]
