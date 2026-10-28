"""QStrategy: the Strategy (agent.controller.Strategy) the controller runs: the Q model choosing among the action-states.

This is where the two layers meet. The action-states (candidate/agent/behavior) know how to move and when each state
may or must be taken; the Q model (this package) knows what each state is worth. Per frame, in the order the brief
gives (the belief has already been updated by the runtime):

    behavior.snapshot -> Q's three inputs -> Q's five values -> behavior.moves.valid_states (masks)
    -> behavior.chooser (switching cost, hard rules) -> behavior.moves.move_for (the chosen state's move)
    -> the safety look-ahead (behavior.safety) and the guard -> a record of the frame

choose() may be called again in the same frame after the engine refused a move; then the state is kept, only the
move is recomputed with the refused cell excluded. The plan is advanced in on_result(), never in choose().

The Q model is anything with values(frame, inputs) -> (6,): HeuristicQ (the hand-set stand-in) or NetworkQ (the trained
network, agent.q.network).

Three things serve training (brief, "Labels and training"):
  * exploration: on `explore_rate` of the frames (3% when collecting) a random valid state is picked and held for 1 to
    `explore_hold` frames, or until it becomes invalid. The hard rules still apply. Without it the model never learns
    what the states it did not choose are worth. Off by default.
  * a GameRecorder (agent.q.record), if given, writes every frame's inputs, values, masks, states and the accepted move.
  * behaviour randomness (BehaviorConfig.randomness, behavior/style.py): each game, drawn from `seed`, gets its own
    variation of how the states behave, so that two of our agents (or ours and a deterministic opponent) do not play
    the same game every time. 0 by default and in a real match. `game_behavior` and `style` are this game's draw.
"""

import random
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from candidate.agent.behavior import chooser, moves, safety, style
from candidate.agent.behavior.config import BEHAVIOR, BehaviorConfig
from candidate.agent.behavior.frame import Frame, Hints
from candidate.agent.behavior.snapshot import Snapshot, build_snapshot
from candidate.agent.behavior.states import State
from candidate.agent.data.types import Direction, MoveResult
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import step
from candidate.agent.q.config import Q, QConfig
from candidate.agent.q.heuristic import HeuristicQ
from candidate.agent.q.inputs import QInputs, build_inputs
from candidate.agent.runtime import Event, TurnView


@dataclass(frozen=True, eq=False)
class FrameRecord:
    """What was decided in one frame: the inputs exactly as Q saw them, its values, the masks and the state taken."""

    turn: int
    inputs: QInputs | None
    snapshot: Snapshot
    values: np.ndarray
    valid: np.ndarray
    previous: State
    state: State
    forced: str | None
    direction: Direction | None


class QStrategy:
    def __init__(
        self,
        q_model=None,
        cfg: QConfig = Q,
        behavior: BehaviorConfig = BEHAVIOR,
        build_inputs: bool = True,
        recorder=None,
        explore_rate: float = 0.0,
        explore_hold: int | None = None,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg                  # the Q model's constants
        self.behavior = behavior        # the action-states' constants
        self.q_model = q_model if q_model is not None else HeuristicQ(cfg)
        self.build_inputs = build_inputs
        self.recorder = recorder
        self.explore_rate = float(explore_rate)
        self.explore_hold = int(explore_hold if explore_hold is not None else cfg.exploration_hold)
        self.seed = seed
        if (recorder is not None or getattr(self.q_model, "needs_inputs", False)) and not build_inputs:
            raise ValueError("recording and the network both need Q's inputs: build_inputs must be True")
        self.reset()

    def reset(self) -> None:
        """A new match. A game still being recorded (a reset in the middle of it) is saved as incomplete."""
        if getattr(self, "recorder", None) is not None and self.recorder.active:
            self.recorder.finish(complete=False)
        self.state = State.PATROL
        self._rng = random.Random(self.seed)
        self.game_behavior, self.style = style.draw(self.behavior, self.seed)
        self._explore_state: State | None = None
        self._explore_left = 0
        self._last_turn = False
        self.hints = Hints()
        self.last: FrameRecord | None = None
        self.stats: Counter = Counter()
        self.timing_ms: dict[str, list[float]] = {"snapshot": [], "inputs": [], "q": [], "move": []}
        self._turn: int | None = None
        self._snap: Snapshot | None = None
        self._record: FrameRecord | None = None

    # ---- the Strategy interface ----------------------------------------------------------------------------

    def choose(self, view: TurnView) -> Direction:
        if view.retry and self._turn == view.obs.turn and self._snap is not None:
            frame = Frame(view, self._snap, self.hints, self.game_behavior, self.style)   # same decision, another move
            self.stats["retries"] += 1
            return self._guard(frame, moves.move_for(self.state, frame))
        return self._decide(view)

    def on_result(self, view: TurnView, direction: Direction, result: MoveResult) -> None:
        if result == MoveResult.INVALID:
            return
        if self.recorder is not None:
            self.recorder.accept(direction)
            if self._last_turn:
                self.recorder.finish(complete=True)
        snap = self._snap
        if result in (MoveResult.DIED, MoveResult.CAPTURED):
            self.hints.outward = self.hints.across = None
        elif snap is not None and not snap.exposed and not bitboard.has_cell(snap.my_land_bb, step(snap.my_pos, direction)):
            self.hints.outward = Direction(direction)                      # just left my land: remember which way
        self.stats[f"result_{result.name}"] += 1

    def debug_state(self) -> str:
        return self.state.name + (f" !{self._record.forced}" if self._record and self._record.forced else "")

    def close(self) -> None:
        """Save a game still being recorded (as incomplete if it did not reach its last turn)."""
        if self.recorder is not None and self.recorder.active:
            self.recorder.finish(complete=False)

    # ---- one frame -----------------------------------------------------------------------------------------------------

    def _decide(self, view: TurnView) -> Direction:
        behavior = self.game_behavior
        mark = time.perf_counter()
        snap = build_snapshot(view, behavior)
        frame = Frame(view, snap, self.hints, behavior, self.style)
        self._tick("snapshot", mark)

        mark = time.perf_counter()
        inputs = build_inputs(view, snap, self.state, self.cfg) if self.build_inputs else None
        self._tick("inputs", mark)

        mark = time.perf_counter()
        values = np.asarray(self.q_model.values(frame, inputs), dtype=np.float64)
        valid = moves.valid_states(frame)
        died = Event.SELF_DIED in view.events or Event.KILLED_BY_OPPONENT in view.events
        choice = chooser.choose(values, valid, self.state, died=died, trail_len=snap.trail_len, margin=snap.safety_margin,
                                cfg=behavior)
        self._tick("q", mark)

        mark = time.perf_counter()
        previous, self.state = self.state, choice.state
        forced = choice.forced
        explored = self._explore(valid, forced)
        if not snap.exposed:
            self.hints.outward = self.hints.across = None
        move = self._guard(frame, moves.move_for(self.state, frame))
        if behavior.safety_lookahead and forced != "death" and (
                self.state != State.RETURN or safety.kills(snap, step(frame.pos, move))):
            move, forced = self._look_ahead(frame, move, forced)
        self._tick("move", mark)

        self._turn, self._snap = view.obs.turn, snap
        self._record = self.last = FrameRecord(view.obs.turn, inputs, snap, values, valid, previous, self.state, forced, move)
        self.last_explored = explored
        if self.recorder is not None:
            if not self.recorder.active:
                self.recorder.start(view, getattr(self.q_model, "name", type(self.q_model).__name__), self.explore_rate)
            self.recorder.add(self._record, view, explored)
        self._last_turn = view.obs.turns_left <= 1
        if explored:
            self.stats["explored"] += 1
        self.stats[f"state_{self.state.name}"] += 1
        if forced:
            self.stats[f"forced_{forced}"] += 1
        if previous != self.state:
            self.stats["switches"] += 1
        return move

    def _explore(self, valid: np.ndarray, forced: str | None) -> bool:
        """Exploration: replace the chosen state by a random valid one now and then, and hold it for a few frames.
        A hard rule ends it. Changes self.state; returns whether this frame's state came from exploration."""
        if self.explore_rate <= 0.0:
            return False
        if forced is not None:
            self._explore_left = 0
            return False
        if self._explore_left > 0 and self._explore_state is not None and valid[self._explore_state]:
            self._explore_left -= 1
            self.state = self._explore_state
            return True
        self._explore_left = 0
        if self._rng.random() >= self.explore_rate:
            return False
        options = [s for s in State if valid[s]]
        if not options:
            return False
        self._explore_state = self._rng.choice(options)
        self._explore_left = self._rng.randint(1, self.explore_hold) - 1
        self.state = self._explore_state
        return True

    def _look_ahead(self, frame: Frame, move: Direction, forced: str | None) -> tuple[Direction, str | None]:
        """Refuse a move that would leave the safety margin below the floor: go home instead if there is a trail to go
        home along, else stay inside my land. If even that step would (RETURN's step home can cut the enemy's trail and
        bring it back right beside mine), take the safe move with the best margin, the nearer home among equals.
        Nothing Q or the switching cost says can outvote this."""
        snap = frame.snap
        behavior = self.game_behavior
        margin = lambda d: safety.margin_after(snap, step(frame.pos, d), behavior)  # noqa: E731
        if margin(move) >= behavior.safety_floor:
            return move, forced
        self.state = State.RETURN if snap.exposed else State.PATROL
        fallback = self._guard(frame, moves.move_for(self.state, frame))
        if margin(fallback) < behavior.safety_floor and frame.safe:
            fallback = max(frame.safe, key=lambda d: (margin(d), -int(snap.home_dist[step(frame.pos, d)]), -int(d)))
        return fallback, "lookahead"

    def _guard(self, frame: Frame, move: Direction | None) -> Direction:
        """The last word on a move: it must be one the engine accepts that does not step onto my own trail. If the
        state gave none (or an unsafe one), fall back on the safe move that gets home soonest."""
        if move is not None and move in frame.safe:
            return move
        self.stats["guarded"] += 1
        if frame.safe:
            snap = frame.snap

            def key(direction: Direction):
                dest = step(frame.pos, direction)
                return (int(snap.home_dist[dest]), int(direction))

            return min(frame.safe, key=key)
        legal = frame.view.legal_moves
        return legal[0] if legal else Direction.NORTH

    def _tick(self, name: str, mark: float) -> None:
        self.timing_ms[name].append((time.perf_counter() - mark) * 1000)
