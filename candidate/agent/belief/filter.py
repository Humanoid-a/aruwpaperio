"""The belief filter: what the opponent might be doing, given everything seen.

This file only calls the steps in order. Each step is a module of its own:

    belief.state       the data, and the correct state to start a match in
    belief.generate    1. every move of every hypothesis, resolved by the game rules
    belief.cut         2. remove what the rules and this observation make impossible
    belief.weigh       3. apply the motion model K: P(move | hypothesis), child weight = parent x P
    belief.settle      4. merge, normalise, prune into the residual, re-base on memory, rehydrate
    belief.after_move  5. apply my own move to every hypothesis, at the end of the frame
    belief.publish     the read-only BeliefView

A frame, as the runtime drives it:

    begin_match(rules, obs)            once, at the start of a match
    ---- every one of my turns ----
    observe(obs, memory, killed)       steps 1-4, then publishes belief.view
    ... the agent chooses and makes its move ...
    after_my_move(move)                step 5

The hypotheses always describe the world as it stands right after my latest move
and before the opponent's reply: the world the opponent moves in. observe() makes
that reply for every hypothesis and compares the result with what is seen; step 5
then moves the world on by my move. Nothing needs to be applied at the start of a
frame, which is why step 5 is the last thing in the previous one.

Evidence only removes a hypothesis when the rules or an observation contradict it.
K never does, because move_probabilities keeps every legal move above zero. So if the
model of the game is right the true state is never removed, only pruned into the
residual, which keeps its position.
"""

import dataclasses

import numpy as np

from candidate.agent.belief.after_move import MyMove, apply_my_move
from candidate.agent.belief.config import BeliefConfig
from candidate.agent.belief.cut import Evidence, cut_children, cut_moves, gate_residual, position_ok_unmoved
from candidate.agent.belief.generate import (
    Children,
    Situation,
    advance_residual,
    build_children,
    classify,
    unmoved_children,
)
from candidate.agent.belief.motion import FeatureMotionModel, MotionModel
from candidate.agent.belief.publish import publish
from candidate.agent.belief.settle import rebase, settle
from candidate.agent.belief.state import BeliefState, BeliefStats, MyState, invalidate, is_fresh_start, start_state
from candidate.agent.belief.weigh import assign_weights, first_look_stay
from candidate.agent.data.belief_view import BeliefEvent, BeliefView
from candidate.agent.data.memory import MemoryMap
from candidate.agent.data.observation import Observation
from candidate.agent.geometry.zobrist import DEFAULT_KEYS, ZobristKeys
from candidate.agent.rules.match import MatchRules


class BeliefFilter:
    def __init__(
        self,
        config: BeliefConfig | None = None,
        motion: MotionModel | None = None,
        keys: ZobristKeys = DEFAULT_KEYS,
    ) -> None:
        self.config = config if config is not None else BeliefConfig()
        self.motion = motion if motion is not None else FeatureMotionModel()
        self.keys = keys
        self.state: BeliefState | None = None
        self.view: BeliefView | None = None
        self.stats = BeliefStats()
        self._stats_mark = dataclasses.replace(self.stats)   # the statistics at the last published view

    @property
    def rules(self) -> MatchRules | None:
        return None if self.state is None else self.state.rules

    # ---- the start of a match ------------------------------------------------------------------------------

    def begin_match(self, rules: MatchRules, obs: Observation | None = None) -> None:
        """A new match: the opponent is on its respawn with its start land and no trail, and so am I. If the
        first observation shows that is not so (a controller switched on in the middle of a match) the belief
        is built from that observation instead."""
        self.stats = BeliefStats()
        self._stats_mark = dataclasses.replace(self.stats)
        self.view = None
        self.state = start_state(rules, self.keys)
        if obs is not None and not is_fresh_start(rules, obs):
            invalidate(self.state)

    # ---- steps 1-4: observe -----------------------------------------------------------------------------------

    def observe(
        self, obs: Observation, memory: MemoryMap, killed_by_opponent: bool = False, force_reseed: bool = False
    ) -> BeliefView:
        """Bring the belief up to date with this turn's observation and publish it. `memory` must already
        have taken this observation's window (it is what the lands are re-based on)."""
        state, stats = self.state, self.stats
        stats.updates += 1
        events: list[BeliefEvent] = []
        if obs.opp_seen_at is not None:
            events.append(BeliefEvent.SIGHTED)
            state.sighting = (obs.opp_seen_at, obs.turn)
        elif state.was_seen:
            events.append(BeliefEvent.LOST_SIGHT)
        state.was_seen = obs.opp_seen_at is not None

        if force_reseed or state.needs_reseed:
            invalidate(state)
        else:
            events.extend(state.pending)
            if not self._advance(obs, killed_by_opponent):
                invalidate(state)
        state.pending = []
        state.events = events

        rebase(state, memory, obs, self.config, stats)
        state.me = MyState.of(obs)
        self.view = publish(state, obs, tuple(state.events), stats, self._stats_mark)
        self._stats_mark = dataclasses.replace(stats)
        return self.view

    def _advance(self, obs: Observation, killed_by_opponent: bool) -> bool:
        """The opponent's move, for every hypothesis. False if the evidence left nothing."""
        state, cfg, stats = self.state, self.config, self.stats
        ev = Evidence(obs, killed_by_opponent)
        sit = Situation.of(state, obs)

        exp = classify(state, sit)                                          # 1. generate
        keep = cut_moves(exp, ev, stats)                                    # 2. cut what the moves rule out ...
        parents, moves = np.nonzero(keep)
        kids = build_children(state, exp, parents, moves)
        first_stay = None
        if state.first:
            first_stay = first_look_stay(state.rules, cfg)
            if not killed_by_opponent:
                # If I moved first the opponent may not have moved at all yet.
                kids = Children.concatenate(kids, unmoved_children(exp, position_ok_unmoved(exp, ev)))
        stats.children += kids.n

        advance_residual(state, sit, obs, cfg.residual_respawn_rate)        # the residual moves and is cut too
        gate_residual(state, sit, ev)
        kids = kids.take(cut_children(kids, sit, ev, stats))                # ... then what the children rule out

        assign_weights(kids, exp, self.motion, cfg, first_stay)             # 3. K
        if not settle(state, kids, cfg, stats):                             # 4. store back
            return False
        state.first = False
        return True

    # ---- step 5: my move ------------------------------------------------------------------------------------------

    def after_my_move(self, move: MyMove) -> None:
        """Apply my move, which the engine has accepted, to every hypothesis. Call it at the end of the frame."""
        apply_my_move(self.state, move, self.stats)
