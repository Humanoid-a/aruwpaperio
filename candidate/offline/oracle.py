"""The true world as a belief, for training games: OracleBelief stands in for the belief filter.

Running the belief filter is about two thirds of a simulated game's time, and its guesses are noise in Q's inputs.
For most training games (all but QConfig.belief_share of them) the agent is instead handed the truth: every turn,
OracleBelief reads the engine's authoritative board and publishes it as a belief with exactly one hypothesis of weight
1 (the opponent's real cell, trail, land and count, and my real land) and no residual. Everything downstream (the
snapshot, the states, Q's inputs) is unchanged code reading a belief that happens to be certain, so it is exact and
cheap. In the remaining games the real filter runs, so Q also learns from the uncertain inputs it meets in a real
match; the probability map's layers are what let one network take both (one-hot when certain, spread when not).

What this changes beyond the inputs, and why it is accepted: with the truth, the states act on it too. HUNT and TRACK
TARGET know where the enemy is, and the worst-case safety margin is exact rather than conservative, so the agent plays
more boldly in these games than it can in a real match. The values learned from them describe a better-informed
agent; the belief games keep the network anchored to what is possible with the belief. Games are labelled (the
record's meta "oracle"), so the two kinds can be told apart, weighed or filtered when training.

Only for headless training games (offline/collect.py): it reads the framework engine's internal state, which a real
controller cannot see. It is never used during a real match.
"""

import dataclasses

import numpy as np

from candidate.agent.belief.publish import publish
from candidate.agent.belief.state import BeliefStats, MyState, start_state
from candidate.agent.config import BOARD_WIDTH
from candidate.agent.data.belief_view import BeliefEvent, BeliefView
from candidate.agent.data.cells import MY_TERRITORY, OPP_TERRITORY
from candidate.agent.data.territory import TerritoryDiffs
from candidate.agent.geometry import bitboard


@dataclasses.dataclass(frozen=True)
class TrueWorld:
    """The board as it really is, from one player's side. Cells are flat (50*y + x); lands are bitboards."""

    my_pos: int
    my_trail: tuple[int, ...]
    my_land: np.ndarray
    opp_pos: int
    opp_trail: tuple[int, ...]
    opp_land: np.ndarray


class EngineTruth:
    """Reads the true board from a running GameEngine. Attach it with match.play(..., on_engine=truth.attach)."""

    def __init__(self) -> None:
        self.engine = None

    def attach(self, engine) -> None:
        self.engine = engine

    def world(self, seat: int) -> TrueWorld:
        if self.engine is None:
            raise RuntimeError("the truth source was never attached to an engine")
        state = self.engine._state                       # the engine's own authoritative match state (read only)
        players = state.get_players()
        me, opp = players[seat], players[1 - seat]
        owner = np.array([[-1 if c._territory_owner is None else c._territory_owner.index for c in row]
                          for row in state.get_board()._cells], dtype=np.int8)

        def flat(p) -> int:
            return p.y * BOARD_WIDTH + p.x

        def trail(player) -> tuple[int, ...]:
            return tuple(flat(p) for p in player.get_agent().get_active_trail())

        return TrueWorld(
            my_pos=flat(me.get_agent().get_position()), my_trail=trail(me),
            my_land=bitboard.from_mask(owner == me.get_id().index),
            opp_pos=flat(opp.get_agent().get_position()), opp_trail=trail(opp),
            opp_land=bitboard.from_mask(owner == opp.get_id().index),
        )


class OracleBelief:
    """Has the three methods AgentRuntime calls on a BeliefFilter (begin_match, observe, after_my_move) and publishes
    a BeliefView, but the view is the true world, certain. Pass it as Controller(..., belief=OracleBelief(...))."""

    def __init__(self, truth: EngineTruth, seat: int) -> None:
        self.truth, self.seat = truth, seat
        self.rules = None
        self.state = None
        self.view: BeliefView | None = None
        self.stats = BeliefStats()
        self._mark = dataclasses.replace(self.stats)
        self._was_seen = False

    def begin_match(self, rules, obs=None) -> None:
        self.rules = rules
        self.stats = BeliefStats()
        self._mark = dataclasses.replace(self.stats)
        self._was_seen = False

    def observe(self, obs, memory, killed_by_opponent: bool = False, force_reseed: bool = False) -> BeliefView:
        world = self.truth.world(self.seat)
        state = start_state(self.rules)
        # The lands are stored as differences from memory's view, as the real belief stores them (data/territory.py).
        state.opp_base = bitboard.from_mask(memory.territory == OPP_TERRITORY)
        state.my_base = bitboard.from_mask(memory.territory == MY_TERRITORY)
        state.opp_diffs, state.my_diffs = TerritoryDiffs(), TerritoryDiffs()
        my_land_id = int(state.my_diffs.intern_territory(world.my_land, state.my_base)[0])
        state.store = state.single(world.opp_pos, world.opp_land, bitboard.from_cells(world.opp_trail),
                                   int(bitboard.popcount(world.opp_land)), my_land_id=my_land_id)
        state.me = MyState.of(obs)
        state.first = False
        events = []
        if obs.opp_seen_at is not None:
            events.append(BeliefEvent.SIGHTED)
        elif self._was_seen:
            events.append(BeliefEvent.LOST_SIGHT)
        self._was_seen = obs.opp_seen_at is not None
        self.stats.updates += 1
        self.state = state
        self.view = publish(state, obs, tuple(events), self.stats, self._mark)
        self._mark = dataclasses.replace(self.stats)
        return self.view

    def after_my_move(self, move) -> None:
        """Nothing to do: the next observe() reads the board again."""


