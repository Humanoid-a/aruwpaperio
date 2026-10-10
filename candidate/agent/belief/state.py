"""Everything the belief owns, and the correct state to start a match in.

A BeliefState is plain data. The steps of a frame (belief.generate, cut, weigh,
settle, after_move) are functions that read and change it; the BeliefFilter in
belief.filter only calls them in order.

What a frame is
---------------
    1. observe       generate -> cut -> weigh -> settle     (belief.filter.observe)
    2. the agent chooses and makes its move
    3. after my move: apply it to every hypothesis         (belief.after_move)

so the stored hypotheses always describe the world as it stands *right after my
latest move and before the opponent's reply*, which is exactly the world the
opponent moves in.

The starting state
------------------
At the start of a match nothing is hidden. The opponent stands on its respawn
cell on its 3x3 start land, with no trail and a count of 9; I stand on mine on
mine. start_state() builds exactly that: one hypothesis of weight 1, my land and
the opponent's land equal to the start lands. The only thing not known is who
moves first, and with it how many moves the opponent has made when I first look
(belief.weigh.first_look_stay).
"""

from dataclasses import dataclass, field

import numpy as np

from candidate.agent.config import BOARD_HEIGHT
from candidate.agent.data.belief_view import BeliefEvent
from candidate.agent.data.hypotheses import HypothesisStore
from candidate.agent.data.observation import Observation
from candidate.agent.data.residual import ResidualGrid
from candidate.agent.data.territory import TerritoryDiffs
from candidate.agent.data.types import MoveResult
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.zobrist import DEFAULT_KEYS, ZobristKeys
from candidate.agent.rules.match import MatchRules


@dataclass
class BeliefStats:
    """Running totals; the published summary reports the difference since the previous view."""

    updates: int = 0
    reseeds: int = 0
    children: int = 0            # successors generated that survived position-level evidence
    cut_by_position: int = 0     # successors ruled out by where the opponent is / whether it killed me
    cut_by_count: int = 0        # ... by its territory count
    cut_by_window: int = 0       # ... by the cells I can see
    cut_by_my_count: int = 0     # ... by the drop in my own territory count
    merged: int = 0              # duplicates folded together
    pruned_mass: float = 0.0     # probability moved into the residual by pruning
    resets: int = 0              # times my kill collapsed the belief to a single state
    rehydrated: int = 0          # hypotheses given a guessed history out of the residual


@dataclass(frozen=True, eq=False)
class MyState:
    """My own agent as the belief knows it. Exact, and the same under every hypothesis."""

    pos: int
    trail_bb: np.ndarray            # (50,) my trail
    count: int | None               # the engine's count of my cells; None when it cannot be told exactly
    result: MoveResult | None       # what my latest move did; None at the start or when none was accepted

    @classmethod
    def of(cls, obs: Observation) -> "MyState":
        """As the observation shows me (nothing is known about a move yet)."""
        return cls(obs.my_pos, bitboard.from_cells(int(c) for c in obs.my_trail), obs.my_count, None)


@dataclass(eq=False)
class BeliefState:
    rules: MatchRules
    keys: ZobristKeys
    store: HypothesisStore
    residual: ResidualGrid
    opp_diffs: TerritoryDiffs           # the opponent's land under each hypothesis ...
    opp_base: np.ndarray                # ... as a difference from these cells (what memory says is its land)
    my_diffs: TerritoryDiffs            # my land under each hypothesis ...
    my_base: np.ndarray                 # ... as a difference from these cells (what memory says is mine)
    me: MyState
    first: bool = True                  # nothing observed yet: the opponent may not have moved
    needs_reseed: bool = False          # everything was contradicted: rebuild at the end of this frame
    sighting: tuple[int, int] | None = None   # (cell, turn) of the latest sighting
    was_seen: bool = False
    opp_count: int | None = None        # the engine's count for the opponent at the last observation
    events: list[BeliefEvent] = field(default_factory=list)      # for the view being built
    pending: list[BeliefEvent] = field(default_factory=list)     # produced by my move, for the next frame

    def opp_land(self, ids) -> np.ndarray:
        """The opponent's land under each id: (len(ids), 50) bitboards."""
        return self.opp_diffs.territory(ids, self.opp_base)

    def my_land(self, ids) -> np.ndarray:
        """My land under each id: (len(ids), 50) bitboards."""
        return self.my_diffs.territory(ids, self.my_base)

    def single(
        self, pos: int, land: np.ndarray, trail: np.ndarray, count: int, my_land_id: int = 0
    ) -> HypothesisStore:
        """One hypothesis of weight 1 with the given state; `land` is its whole land as a bitboard."""
        length = int(bitboard.popcount(trail))
        return HypothesisStore.build(
            pos=[pos],
            last_dir=[0 if length else -1],
            trail_len=[length],
            trail_bb=trail[None, :],
            opp_land_id=self.opp_diffs.intern_territory(land, self.opp_base),
            my_land_id=[my_land_id],
            opp_count=[count],
            flags=[0],
            weight=[1.0],
            hash=[self.keys.hash_position("opp_pos", pos) ^ self.keys.hash_board("opp_trail", trail)],
        )


def is_fresh_start(rules: MatchRules, obs: Observation) -> bool:
    """Does the first observation look like the start of a match? I stand on my respawn with no trail and
    both of us own exactly our start lands. If not (a controller switched on in the middle of a match) the
    start state does not apply."""
    return (
        obs.my_pos == rules.me.respawn
        and obs.my_trail.size == 0
        and obs.my_count == len(rules.me.start_cells)
        and obs.opp_count == len(rules.opp.start_cells)
    )


def start_state(rules: MatchRules, keys: ZobristKeys = DEFAULT_KEYS) -> BeliefState:
    """The belief at the start of a match: one certain hypothesis, and both lands equal to the start lands."""
    state = BeliefState(
        rules=rules,
        keys=keys,
        store=HypothesisStore.empty(),
        residual=ResidualGrid(),
        opp_diffs=TerritoryDiffs(keys),
        opp_base=rules.opp.start_bb.copy(),
        my_diffs=TerritoryDiffs(keys),
        my_base=rules.me.start_bb.copy(),
        me=MyState(rules.me.respawn, bitboard.empty(), len(rules.me.start_cells), None),
    )
    state.store = state.single(rules.opp.respawn, rules.opp.start_bb, bitboard.empty(), len(rules.opp.start_cells))
    return state


def invalidate(state: BeliefState) -> None:
    """Nothing is consistent: forget the hypotheses and rebuild at the end of this frame, when the new memory is known."""
    state.store = HypothesisStore.empty()
    state.residual = ResidualGrid()
    state.needs_reseed = True
    state.first = False
