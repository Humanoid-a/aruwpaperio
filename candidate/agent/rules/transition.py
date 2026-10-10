"""Fast, batched move rules on bitboards.

A BitWorld holds N complete boards at once, each as four (50,) bitboards
(both players' territory and trail) plus the two agent positions. step()
advances every board by one move in a handful of vectorised operations:
every (board, move) pair is classified with bit tests, only the boards that
actually capture pay for a flood fill, and nothing is copied per board.

It implements exactly the rules of rules.reference (which mirrors the real
engine) and is checked against it. Differences in representation:

  * Trails are sets, not walking-order lists. The rules never need the
    order, but it cannot always be recovered from a set, so a BitWorld
    cannot be turned back into a WorldState; use rules.reference when the
    ordered trail of the result matters.
  * Counts are not stored; counts() takes a popcount.

Result codes are the MoveResult values: 0 NORMAL, 1 CAPTURED, 2 DIED,
3 INVALID. A move that is INVALID leaves its board unchanged.
"""

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.data.types import MoveResult, Side
from candidate.agent.data.world import WorldState
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.fill import capture_region
from candidate.agent.geometry.grid import DX, DY
from candidate.agent.geometry.zobrist import DEFAULT_KEYS, ZobristKeys
from candidate.agent.rules.match import MatchRules

_DY = np.array(DY, dtype=np.int32)
_DX = np.array(DX, dtype=np.int32)
_NORMAL = np.int8(MoveResult.NORMAL.value)
_CAPTURED = np.int8(MoveResult.CAPTURED.value)
_DIED = np.int8(MoveResult.DIED.value)
_INVALID = np.int8(MoveResult.INVALID.value)

_FIELDS = ("my_territory", "opp_territory", "my_trail", "opp_trail")


@dataclass(eq=False)
class BitWorld:
    """N complete boards in bitboard form: arrays of shape (N, 50) for the
    four layers and (N,) int32 for the agent positions (flat indices)."""

    my_territory: np.ndarray
    opp_territory: np.ndarray
    my_trail: np.ndarray
    opp_trail: np.ndarray
    my_pos: np.ndarray
    opp_pos: np.ndarray

    @property
    def n(self) -> int:
        return self.my_pos.shape[0]

    @classmethod
    def from_worlds(cls, worlds: Sequence[WorldState]) -> "BitWorld":
        layers = [w.bitboards() for w in worlds]
        return cls(
            np.stack([b.my_territory for b in layers]),
            np.stack([b.opp_territory for b in layers]),
            np.stack([b.my_trail for b in layers]),
            np.stack([b.opp_trail for b in layers]),
            np.array([w.my_pos for w in worlds], dtype=np.int32),
            np.array([w.opp_pos for w in worlds], dtype=np.int32),
        )

    @classmethod
    def from_world(cls, world: WorldState) -> "BitWorld":
        return cls.from_worlds([world])

    @classmethod
    def concatenate(cls, parts: Sequence["BitWorld"]) -> "BitWorld":
        return cls(*(np.concatenate([getattr(p, f) for p in parts]) for f in (*_FIELDS, "my_pos", "opp_pos")))

    def take(self, index) -> "BitWorld":
        """The boards at `index` (an array of indices or a boolean mask), as a new BitWorld."""
        return BitWorld(*(getattr(self, f)[index].copy() for f in (*_FIELDS, "my_pos", "opp_pos")))

    def copy(self) -> "BitWorld":
        return self.take(slice(None))

    def counts(self) -> tuple[np.ndarray, np.ndarray]:
        """(my territory sizes, opponent territory sizes), each of shape (N,)."""
        return bitboard.popcount(self.my_territory), bitboard.popcount(self.opp_territory)

    def hash(self, keys: ZobristKeys = DEFAULT_KEYS) -> np.ndarray:
        """Zobrist hash of every board, shape (N,)."""
        return keys.hash_state(
            self.my_pos, self.opp_pos, self.my_territory, self.opp_territory, self.my_trail, self.opp_trail
        )

    def matches(self, i: int, world: WorldState) -> bool:
        """Does board i hold the same layers and positions as `world`? (Trail
        order is not compared; a BitWorld does not have it.)"""
        layers = world.bitboards()
        return (
            all(np.array_equal(getattr(self, f)[i], getattr(layers, f)) for f in _FIELDS)
            and int(self.my_pos[i]) == world.my_pos
            and int(self.opp_pos[i]) == world.opp_pos
        )

    def problems(self) -> list[str]:
        """Broken invariants, as readable messages naming the first bad board."""
        out: list[str] = []
        for f in _FIELDS:
            arr = getattr(self, f)
            if not bitboard.is_valid(arr) or arr.ndim != 2:
                out.append(f"{f} is not a valid (N, {BOARD_HEIGHT}) bitboard array")
        for f in ("my_pos", "opp_pos"):
            arr = getattr(self, f)
            if arr.shape != (self.n,) or ((arr < 0) | (arr >= CELL_COUNT)).any():
                out.append(f"{f} must hold on-board flat indices, one per board")
        if out:
            return out

        def first(bad: np.ndarray, message: str) -> None:
            if bad.any():
                out.append(f"{message} (first at board {int(np.argmax(bad))})")

        first((self.my_territory & self.opp_territory).any(axis=1), "both players own the same cell")
        first((self.my_trail & self.my_territory).any(axis=1), "my trail lies on my territory")
        first((self.opp_trail & self.opp_territory).any(axis=1), "the opponent's trail lies on their territory")
        first(self.my_pos == self.opp_pos, "both agents are on the same cell")
        for name, trail, pos in (("my", self.my_trail, self.my_pos), ("opp", self.opp_trail, self.opp_pos)):
            has_trail = (trail != 0).any(axis=1)
            head_on_trail = _test(trail, pos[:, None])[:, 0]
            first(has_trail & ~head_on_trail, f"{name} trail does not include the agent's cell")
        return out


# ---- classification (the "predict" building block) ----------------------


class MoveClasses(NamedTuple):
    """For every board and each of the four moves (columns N, E, S, W):"""

    dest: np.ndarray        # (N, 4) int32 destination cell (the mover's own cell where off the board)
    legal: np.ndarray       # (N, 4) bool: on the board and not onto the other agent
    self_trail: np.ndarray  # (N, 4) bool: legal, onto the mover's own trail: the mover dies
    cuts_other: np.ndarray  # (N, 4) bool: legal, survives, onto the other's trail: the other dies
    enters_own: np.ndarray  # (N, 4) bool: legal, survives, onto the mover's own territory
    captures: np.ndarray    # (N, 4) bool: enters_own while holding a trail: the loop closes


_test = bitboard.test_cells


def _arrays(bw: BitWorld, side: Side):
    if side == Side.ME:
        return bw.my_territory, bw.my_trail, bw.my_pos, bw.opp_territory, bw.opp_trail, bw.opp_pos
    return bw.opp_territory, bw.opp_trail, bw.opp_pos, bw.my_territory, bw.my_trail, bw.my_pos


def _destinations(pos: np.ndarray, direction: np.ndarray):
    """Destination cell and on-board flag for moves `direction` (broadcast against pos)."""
    y, x = np.divmod(pos, BOARD_WIDTH)
    ny, nx = y + _DY[direction], x + _DX[direction]
    on_board = (ny >= 0) & (ny < BOARD_HEIGHT) & (nx >= 0) & (nx < BOARD_WIDTH)
    return np.where(on_board, ny * BOARD_WIDTH + nx, pos).astype(np.int32), on_board


def classify_moves(bw: BitWorld, side: Side) -> MoveClasses:
    """Classify all four moves of `side` on every board with bit tests only.
    No board is built or copied; step() reuses the same tests for the move it
    is given."""
    mt, mtr, mp, _, otr, op = _arrays(bw, side)
    all_moves = np.arange(4)[None, :]
    dest, on_board = _destinations(mp[:, None], all_moves)
    legal = on_board & (dest != op[:, None])
    self_trail = legal & _test(mtr, dest)
    survives = legal & ~self_trail
    cuts_other = survives & _test(otr, dest)
    enters_own = survives & _test(mt, dest)
    has_trail = (mtr != 0).any(axis=1)[:, None]
    return MoveClasses(dest, legal, self_trail, cuts_other, enters_own, enters_own & has_trail)


# ---- the transition ------------------------------------------------------


class StepResult(NamedTuple):
    world: BitWorld
    result: np.ndarray  # (N,) int8 MoveResult values
    killed: np.ndarray  # (N,) bool: the mover cut the other player's trail


def step(bw: BitWorld, rules: MatchRules, side: Side, directions) -> StepResult:
    """Advance every board by one move of `side`; directions[i] is the move
    for board i (Direction values 0..3). The input is not modified."""
    directions = np.asarray(directions)
    if directions.shape != (bw.n,) or ((directions < 0) | (directions > 3)).any():
        raise ValueError(f"directions must be {bw.n} values in 0..3")
    out = bw.copy()
    mt, mtr, mp, ot, otr, op = _arrays(out, side)
    mine, theirs = rules.role(side), rules.role(side.other)

    # Classify against the untouched boards, before anything below changes them.
    rows_all = np.arange(bw.n)
    dest, on_board = _destinations(mp, directions)
    legal = on_board & (dest != op)
    drow, dcol = np.divmod(dest, BOARD_WIDTH)
    bit = np.left_shift(np.int64(1), dcol.astype(np.int64))
    on_my_trail = (mtr[rows_all, drow] & bit) != 0
    on_other_trail = (otr[rows_all, drow] & bit) != 0
    on_my_terr = (mt[rows_all, drow] & bit) != 0
    died = legal & on_my_trail
    alive = legal & ~on_my_trail
    captured = alive & on_my_terr & (mtr != 0).any(axis=1)
    killed = alive & on_other_trail
    appends_trail = alive & ~on_my_terr  # the destination was not the mover's land (read before any change)

    # The mover steps onto an own-trail cell: dies and respawns.
    idx = np.nonzero(died)[0]
    if idx.size:
        mtr[idx] = 0
        mt[idx] = mine.start_bb
        ot[idx] = ot[idx] & ~mine.start_bb
        mp[idx] = np.where(op[idx] == mine.respawn, mine.alt_respawn, mine.respawn)

    # Otherwise the mover steps onto the destination ...
    mp[alive] = dest[alive]

    # ... and if it was home with a trail out, the loop closes.
    idx = np.nonzero(captured)[0]
    if idx.size:
        grown = mt[idx] | mtr[idx] | capture_region(mt[idx], mtr[idx])
        mt[idx] = grown
        mtr[idx] = 0
        ot[idx] = ot[idx] & ~grown

    # Cutting the other's trail sends them home, after any capture.
    idx = np.nonzero(killed)[0]
    if idx.size:
        otr[idx] = 0
        ot[idx] = theirs.start_bb
        mt[idx] = mt[idx] & ~theirs.start_bb
        op[idx] = np.where(mp[idx] == theirs.respawn, theirs.alt_respawn, theirs.respawn)

    # Leaving (or staying off) home extends the trail by the new cell.
    idx = np.nonzero(appends_trail)[0]
    if idx.size:
        mtr[idx, drow[idx]] |= bit[idx]

    result = np.full(bw.n, _NORMAL, dtype=np.int8)
    result[captured] = _CAPTURED
    result[died] = _DIED
    result[~legal] = _INVALID
    return StepResult(out, result, killed)
