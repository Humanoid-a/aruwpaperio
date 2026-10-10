"""Step 1 of a frame: generate the opponent's possible moves, resolved by the game rules.

classify() takes every hypothesis and all four directions and works out, with bit
tests and nothing copied, what each move would do: is it on the board, does it
step on its own trail (it dies), on my trail (I die), onto its own land, does it
close a loop and how much land would that capture. These are the rules of
rules.reference seen from the opponent's side, with the part the belief cannot
know for sure (whose land is where) carried as differences from memory. They are
checked against rules.transition, which is checked against the real engine.

Nothing is dropped here except moves that are illegal in the game itself: off the
board, or onto my agent. Everything the evidence says is the next step (belief.cut).

build_children() then turns chosen (hypothesis, move) pairs into new hypotheses.
It is called only for the pairs that survived the first cuts, so no hypothesis is
copied for a move the observation has already ruled out.

The frame all of this lives in: the stored hypotheses describe the world right after
my latest move, and `Situation` carries what is true while the opponent moves (where
my agent stands, my trail, the bases the lands are expressed against).
"""

from dataclasses import dataclass

import numpy as np

from candidate.agent.belief.motion import SuccessorBatch
from candidate.agent.belief.state import BeliefState
from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.hypotheses import FLAG_CAPTURED, FLAG_DIED, FLAG_KILLED_ME, HypothesisStore
from candidate.agent.data.observation import Observation
from candidate.agent.data.residual import ResidualGrid
from candidate.agent.data.territory import TerritoryDiffs
from candidate.agent.data.types import MoveResult
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.fill import capture_region
from candidate.agent.geometry.grid import DX, DY
from candidate.agent.rules.match import MatchRules

_DY = np.array(DY, dtype=np.int64)
_DX = np.array(DX, dtype=np.int64)


# ---- the situation the opponent moves in ---------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Situation:
    """What is true while the opponent moves, as far as I know it."""

    rules: MatchRules
    obs: Observation
    opp_base: np.ndarray        # (50,) the cells memory believes the opponent owns (the frame of its lands)
    opp_diffs: TerritoryDiffs
    my_pos: int                 # where my agent stands during the opponent's move
    my_trail_bb: np.ndarray     # (50,) my trail at that moment
    my_base: np.ndarray         # (50,) the cells memory believes are mine (the frame of my lands)
    my_diffs: TerritoryDiffs
    my_count: int | None        # the engine's count of my cells while the opponent moves, if known exactly

    @classmethod
    def of(cls, state: BeliefState, obs: Observation) -> "Situation":
        me = state.me
        return cls(
            state.rules, obs, state.opp_base, state.opp_diffs, me.pos, me.trail_bb, state.my_base, state.my_diffs, me.count
        )

    @property
    def opponent_respawn(self) -> int:
        """Where the opponent lands if it dies now: its respawn cell, or the alternate if my agent is on it."""
        role = self.rules.opp
        return role.alt_respawn if self.my_pos == role.respawn else role.respawn


@dataclass(eq=False)
class Expansion:
    """All four moves of every hypothesis, classified."""

    batch: SuccessorBatch           # what each move does (also what the motion model K is shown)
    situation: Situation
    store: HypothesisStore          # the parents
    capture_index: np.ndarray       # (n, 4) index into the capture_* arrays, or -1
    capture_territory: np.ndarray   # (k, 50) the opponent's land after each capture
    capture_taken: np.ndarray       # (k, 50) the trail and the enclosed cells: what the capture turns into its land
    capture_gain: np.ndarray        # (k,) cells it adds
    capture_loss: np.ndarray        # (k,) cells of mine it takes (under that hypothesis' view of my land)
    my_exact: np.ndarray            # (n,) the hypothesis' view of my land has exactly the engine's count of cells,
                                    #      so what it says I lose can be held against the count I now see


def classify(state: BeliefState, sit: Situation) -> Expansion:
    store, opp_diffs, my_diffs = state.store, sit.opp_diffs, sit.my_diffs
    n = store.n
    pos = store.pos.astype(np.int64)
    y, x = np.divmod(pos, BOARD_WIDTH)
    ny, nx = y[:, None] + _DY[None, :], x[:, None] + _DX[None, :]
    on_board = (ny >= 0) & (ny < BOARD_HEIGHT) & (nx >= 0) & (nx < BOARD_WIDTH)
    dest = np.where(on_board, ny * BOARD_WIDTH + nx, pos[:, None])
    legal = on_board & (dest != sit.my_pos)                       # the game refuses the rest

    self_trail = legal & bitboard.test_cells(store.trail_bb, dest)            # onto its own trail: it dies
    survives = legal & ~self_trail
    kills_me = survives & bitboard.test_cells(sit.my_trail_bb, dest)          # onto my trail: I die

    rows, cols = np.divmod(dest, BOARD_WIDTH)
    ids = store.opp_land_id[:, None]
    words = (sit.opp_base[rows] & ~opp_diffs.removed[ids, rows]) | opp_diffs.added[ids, rows]
    enters_own = survives & (((words >> cols) & 1).astype(bool))              # onto its own land
    captures = enters_own & (store.trail_len > 0)[:, None]                    # ... with a trail out: a loop closes

    capture_index = np.full((n, 4), -1, dtype=np.int64)
    capture_gain = np.zeros((n, 4), dtype=np.int64)
    parent, move = np.nonzero(captures)
    if parent.size:
        land = opp_diffs.territory(store.opp_land_id[parent], sit.opp_base)
        trail = store.trail_bb[parent]
        region = capture_region(land, trail)
        gain = store.trail_len[parent].astype(np.int64) + bitboard.popcount(region)
        taken = trail | region
        after = land | taken
        loss = bitboard.popcount(taken & my_diffs.territory(store.my_land_id[parent], sit.my_base))
        capture_index[parent, move] = np.arange(parent.size)
        capture_gain[parent, move] = gain
    else:
        after = taken = np.zeros((0, BOARD_HEIGHT), dtype=np.int64)
        gain = loss = np.zeros(0, dtype=np.int64)

    # Does this hypothesis' idea of my land add up to the count the engine reports for me?
    if sit.my_count is None or n == 0:
        my_exact = np.zeros(n, dtype=bool)
    else:
        unique, inverse = np.unique(store.my_land_id, return_inverse=True)
        cells = bitboard.popcount(my_diffs.territory(unique, sit.my_base))
        my_exact = (cells == sit.my_count)[inverse.reshape(-1)]

    batch = SuccessorBatch(
        n=n, pos=store.pos, trail_bb=store.trail_bb, trail_len=store.trail_len, last_dir=store.last_dir,
        opp_count=store.opp_count, weight=store.weight,
        dest=dest, legal=legal, self_trail=self_trail, kills_me=kills_me, enters_own=enters_own,
        captures=captures, capture_gain=capture_gain,
        my_pos=sit.my_pos, my_trail_bb=sit.my_trail_bb, opp_land_bb=sit.opp_base, obs=sit.obs,
    )
    return Expansion(batch, sit, store, capture_index, after, taken, gain, loss, my_exact)


def final_positions(exp: Expansion) -> np.ndarray:
    """(n, 4): where the opponent ends up after each move. A move that kills
    it puts it back at its respawn, not on the cell it stepped to."""
    return np.where(exp.batch.self_trail, exp.situation.opponent_respawn, exp.batch.dest)


# ---- building the children -------------------------------------------------------------------------------


@dataclass(eq=False)
class Children:
    """New hypotheses, with where each one came from. Their weights are filled in by belief.weigh."""

    store: HypothesisStore
    parent: np.ndarray    # (m,) row of the hypothesis each came from
    move: np.ndarray      # (m,) the direction taken, or -1 for "has not moved"
    my_loss: np.ndarray   # (m,) cells of my land this move takes (or gives back)
    my_exact: np.ndarray  # (m,) the parent's view of my land added up to the engine's count

    @property
    def n(self) -> int:
        return self.store.n

    def take(self, keep) -> "Children":
        return Children(self.store.take(keep), self.parent[keep], self.move[keep], self.my_loss[keep], self.my_exact[keep])

    @classmethod
    def empty(cls) -> "Children":
        return cls(HypothesisStore.empty(), *(np.zeros(0, dtype=t) for t in (np.int64, np.int64, np.int64, bool)))

    @classmethod
    def concatenate(cls, a: "Children", b: "Children") -> "Children":
        return cls(
            HypothesisStore.concatenate([a.store, b.store]),
            np.concatenate([a.parent, b.parent]), np.concatenate([a.move, b.move]),
            np.concatenate([a.my_loss, b.my_loss]), np.concatenate([a.my_exact, b.my_exact]),
        )


def build_children(state: BeliefState, exp: Expansion, parents: np.ndarray, moves: np.ndarray) -> Children:
    """The hypotheses that result from the chosen (parent, move) pairs. New land differences are added to the
    situation's tables. Weights are left at zero."""
    sit, batch, rules, keys, store = exp.situation, exp.batch, exp.situation.rules, state.keys, exp.store
    opp_diffs, my_diffs, opp_base, my_base = sit.opp_diffs, sit.my_diffs, sit.opp_base, sit.my_base
    m = parents.size
    if m == 0:
        return Children.empty()

    dest = batch.dest[parents, moves].astype(np.int64)
    death = batch.self_trail[parents, moves]
    capture = batch.captures[parents, moves]
    kills_me = batch.kills_me[parents, moves]
    enters = batch.enters_own[parents, moves]
    plain = ~death & ~capture
    appends = plain & ~enters

    parent_pos = store.pos[parents].astype(np.int64)
    pos = np.where(death, sit.opponent_respawn, dest)
    trail = store.trail_bb[parents].copy()
    trail_len = store.trail_len[parents].copy()
    opp_land_id = store.opp_land_id[parents].copy()
    my_land_id = store.my_land_id[parents].copy()
    opp_count = store.opp_count[parents].copy()
    my_loss = np.zeros(m, dtype=np.int64)

    # A step outside its own land lays one more trail cell (and a first step starts the trail).
    idx = np.nonzero(appends)[0]
    row, col = np.divmod(dest[idx], BOARD_WIDTH)
    trail[idx, row] |= np.left_shift(np.int64(1), col)
    trail_len[appends] += 1

    # Closing a loop: the trail and the land it encloses become its land, and stop being mine.
    idx = np.nonzero(capture)[0]
    if idx.size:
        k = exp.capture_index[parents[idx], moves[idx]]
        trail[idx] = 0
        trail_len[idx] = 0
        opp_count[idx] = opp_count[idx] + exp.capture_gain[k]
        opp_land_id[idx] = opp_diffs.intern_territory(exp.capture_territory[k], opp_base)
        my_loss[idx] = exp.capture_loss[k]
        mine = my_diffs.territory(my_land_id[idx], my_base) & ~exp.capture_taken[k]
        my_land_id[idx] = my_diffs.intern_territory(mine, my_base)

    # Stepping on its own trail kills it: back to its respawn with its starting land only.
    idx = np.nonzero(death)[0]
    if idx.size:
        start = rules.opp.start_bb
        trail[idx] = 0
        trail_len[idx] = 0
        opp_count[idx] = len(rules.opp.start_cells)
        opp_land_id[idx] = opp_diffs.intern_territory(start, opp_base)[0]
        mine = my_diffs.territory(my_land_id[idx], my_base)
        my_loss[idx] = bitboard.popcount(start & mine)
        my_land_id[idx] = my_diffs.intern_territory(mine & ~start, my_base)          # its respawn takes back its start land

    # Stepping on my trail kills me: I lose all my land but my start land, and my respawn takes that back from it.
    idx = np.nonzero(kills_me)[0]
    if idx.size:
        mine_start = rules.me.start_bb
        land = opp_diffs.territory(opp_land_id[idx], opp_base)
        opp_count[idx] = opp_count[idx] - bitboard.popcount(land & mine_start)
        opp_land_id[idx] = opp_diffs.intern_territory(land & ~mine_start, opp_base)
        my_land_id[idx] = my_diffs.intern_territory(mine_start, my_base)[0]

    exposed = trail_len > 0
    last_dir = np.where(exposed, moves, -1)
    flags = (
        np.where(death, FLAG_DIED, 0) | np.where(capture, FLAG_CAPTURED, 0) | np.where(kills_me, FLAG_KILLED_ME, 0)
    )

    new_pos_key = keys.hash_position("opp_pos", pos)
    moved_hash = (
        store.hash[parents]
        ^ keys.hash_position("opp_pos", parent_pos)
        ^ new_pos_key
        ^ np.where(appends, keys.hash_position("opp_trail", dest), 0)
    )
    hash_ = np.where(plain, moved_hash, new_pos_key)       # an empty trail hashes to zero

    made = HypothesisStore.build(
        pos=pos, last_dir=last_dir, trail_len=trail_len, trail_bb=trail, opp_land_id=opp_land_id, my_land_id=my_land_id,
        opp_count=opp_count, flags=flags, weight=np.zeros(m), hash=hash_,
    )
    return Children(made, parents.copy(), moves.copy(), my_loss, exp.my_exact[parents])


def unmoved_children(exp: Expansion, rows: np.ndarray) -> Children:
    """The hypotheses at `rows` as they are, for "the opponent has not moved yet" (only the first look can
    be that: if I move first, it has not)."""
    return Children(
        exp.store.take(rows), rows.copy(), np.full(rows.size, -1, dtype=np.int64),
        np.zeros(rows.size, dtype=np.int64), exp.my_exact[rows],
    )


# ---- the residual moves too -------------------------------------------------------------------------------


def advance_residual(state: BeliefState, sit: Situation, obs: Observation, respawn_rate: float) -> None:
    """The mass whose history is gone takes its step: it spreads to the neighbouring cells, a share of it
    (`respawn_rate`) goes back to the respawn (it may have died), and if its count fell to the start size
    with nothing of mine to explain it, all of it is at the respawn."""
    residual: ResidualGrid = state.residual
    if state.first:
        return
    legal = np.ones((BOARD_HEIGHT, BOARD_WIDTH), dtype=bool)
    legal.flat[sit.my_pos] = False
    residual.spread(legal)
    residual.teleport(sit.opponent_respawn, respawn_rate)
    start = len(state.rules.opp.start_cells)
    if state.opp_count is not None and state.opp_count > start and obs.opp_count == start:
        if state.me.result not in (MoveResult.CAPTURED, MoveResult.DIED):
            total = residual.total()
            residual.mass[:] = 0.0
            residual.mass.flat[sit.opponent_respawn] = total
