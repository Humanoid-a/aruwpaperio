"""A hypothesis: one complete guess about everything the opponent's moves can have changed.

The belief is a weighted set of hypotheses (plus a residual for probability mass
whose history has been thrown away; see data.residual). Each hypothesis says:

    where the opponent is                       pos
    its trail, and how long it is               trail_bb, trail_len
    the direction of its last step              last_dir
    what land it owns, and how much             opp_land_id, opp_count
    what land I own                             my_land_id
    how probable all that is                    weight

My land is part of the hypothesis because it depends on what the opponent did: a
hypothesis in which the opponent captured a loop around some of my cells knows
those cells are no longer mine, and memory alone cannot know that for cells that
are out of view. The two lands are stored as ids into difference tables
(data.territory); the counts are stored as numbers because the engine reports
both every turn and a count is the strongest evidence there is.

Conventions
  * The trail includes the cell the opponent stands on, as the engine does, so
    pos is in trail_bb exactly when trail_len > 0. (method1.0 states the
    opposite; the engine is the authority.) trail_len therefore equals the
    number of turns spent outside its land, so that is not stored separately.
  * last_dir is the direction of the last step while exposed, -1 when not.
  * Weights are float64: many tiny masses are normalised together.
  * A store is replaced, not edited, on each update, so a published view keeps
    valid arrays. Capacity is bounded by the pruning budget (belief.config).
"""

from dataclasses import dataclass

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.data.territory import TerritoryDiffs
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.zobrist import DEFAULT_KEYS, ZobristKeys

FLAG_DIED = 1       # stepped on its own trail and respawned
FLAG_CAPTURED = 2   # closed its trail onto its own territory
FLAG_KILLED_ME = 4  # stepped on my trail

FIELDS = ("pos", "last_dir", "trail_len", "trail_bb", "opp_land_id", "my_land_id", "opp_count", "flags", "weight", "hash")
_DTYPES = {
    "pos": np.int32,
    "last_dir": np.int8,
    "trail_len": np.int32,
    "trail_bb": np.int64,
    "opp_land_id": np.int64,
    "my_land_id": np.int64,
    "opp_count": np.int32,
    "flags": np.uint8,
    "weight": np.float64,
    "hash": np.int64,
}


@dataclass(eq=False)
class HypothesisStore:
    """Parallel arrays, one entry per hypothesis."""

    pos: np.ndarray          # (n,) flat index of the opponent
    last_dir: np.ndarray     # (n,) Direction of its last step while exposed, else -1
    trail_len: np.ndarray    # (n,) cells in its trail (head included)
    trail_bb: np.ndarray     # (n, 50) its trail as bitboards
    opp_land_id: np.ndarray  # (n,) its land: an id in the opponent's TerritoryDiffs
    my_land_id: np.ndarray   # (n,) my land under this hypothesis: an id in my TerritoryDiffs
    opp_count: np.ndarray    # (n,) the size of its land
    flags: np.ndarray        # (n,) FLAG_* of the move that produced this hypothesis
    weight: np.ndarray       # (n,) probability mass
    hash: np.ndarray         # (n,) Zobrist hash of its position and trail

    @property
    def n(self) -> int:
        return self.pos.shape[0]

    @classmethod
    def empty(cls) -> "HypothesisStore":
        return cls(**{f: np.zeros((0, BOARD_HEIGHT) if f == "trail_bb" else 0, dtype=_DTYPES[f]) for f in FIELDS})

    @classmethod
    def build(cls, **arrays) -> "HypothesisStore":
        """From arrays by field name. `my_land_id` may be left out: 0, meaning my land is what memory says."""
        if "my_land_id" not in arrays:
            arrays["my_land_id"] = np.zeros(np.shape(arrays["pos"])[0], dtype=np.int64)
        return cls(**{f: np.asarray(arrays[f], dtype=_DTYPES[f]) for f in FIELDS})

    def take(self, index) -> "HypothesisStore":
        """The hypotheses at `index` (indices or a boolean mask), copied: indexing with an array always copies."""
        index = np.asarray(index)
        return HypothesisStore(**{f: getattr(self, f)[index] for f in FIELDS})

    @classmethod
    def concatenate(cls, parts: list["HypothesisStore"]) -> "HypothesisStore":
        return cls(**{f: np.concatenate([getattr(p, f) for p in parts]) for f in FIELDS})

    def freeze(self) -> "HypothesisStore":
        for f in FIELDS:
            getattr(self, f).flags.writeable = False
        return self

    def mass_by_cell(self) -> np.ndarray:
        """Total weight at each cell, as (50, 50) float64."""
        return np.bincount(self.pos, weights=self.weight, minlength=CELL_COUNT).reshape(BOARD_HEIGHT, BOARD_WIDTH)

    def merge_key(
        self, opp_diffs: TerritoryDiffs, my_diffs: TerritoryDiffs, keys: ZobristKeys = DEFAULT_KEYS
    ) -> np.ndarray:
        """Hash of everything that makes two hypotheses the same hypothesis: position and trail (the
        stored hash), both lands, the opponent's count and its last direction."""
        return (
            self.hash
            ^ opp_diffs.key[self.opp_land_id]
            ^ my_diffs.key[self.my_land_id]
            ^ keys.count_keys[self.opp_count]
            ^ keys.dir_keys[self.last_dir.astype(np.int64) + 1]
        )

    def problems(
        self,
        opp_diffs: TerritoryDiffs | None = None,
        my_diffs: TerritoryDiffs | None = None,
        keys: ZobristKeys = DEFAULT_KEYS,
    ) -> list[str]:
        """Every broken invariant, as readable messages; empty when sound."""
        out: list[str] = []
        n = self.n
        for f in FIELDS:
            arr = getattr(self, f)
            want = (n, BOARD_HEIGHT) if f == "trail_bb" else (n,)
            if arr.shape != want or arr.dtype != _DTYPES[f]:
                out.append(f"{f} must be {np.dtype(_DTYPES[f]).name} with shape {want}, got {arr.dtype} {arr.shape}")
        if out or n == 0:
            return out
        if not (np.isfinite(self.weight).all() and (self.weight >= 0).all()):
            out.append("weights must be finite and non-negative")
        if ((self.pos < 0) | (self.pos >= CELL_COUNT)).any():
            out.append("a position is off the board")
            return out
        if not bitboard.is_valid(self.trail_bb):
            out.append("a trail bitboard has bits beyond the board")
            return out
        lengths = bitboard.popcount(self.trail_bb)
        if (lengths != self.trail_len).any():
            out.append("trail_len differs from the number of trail cells")
        exposed = self.trail_len > 0
        if (exposed != bitboard.test_cells(self.trail_bb, self.pos)).any():
            out.append("the trail must include the opponent's cell exactly when the trail is not empty")
        if ((self.last_dir < -1) | (self.last_dir > 3)).any() or ((self.last_dir >= 0) != exposed).any():
            out.append("last_dir must be 0..3 while exposed and -1 otherwise")
        if ((self.opp_count < 0) | (self.opp_count > CELL_COUNT)).any():
            out.append("a territory count is out of range")
        rebuilt = keys.hash_position("opp_pos", self.pos) ^ keys.hash_board("opp_trail", self.trail_bb)
        if (rebuilt != self.hash).any():
            out.append("a stored hash differs from the hash recomputed from position and trail")
        if opp_diffs is not None and ((self.opp_land_id < 0) | (self.opp_land_id >= len(opp_diffs))).any():
            out.append("a difference id for the opponent's land is unknown")
        if my_diffs is not None and ((self.my_land_id < 0) | (self.my_land_id >= len(my_diffs))).any():
            out.append("a difference id for my land is unknown")
        return out
