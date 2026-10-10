"""QStrategy: the Strategy (agent.controller.Strategy) that runs the Q model and the five action-states.

Per frame, in the order the brief gives (the belief has already been updated by the runtime):

    snapshot -> Q's three inputs -> Q's five values -> masks -> switching cost -> hard rules -> the chosen
    state's own logic -> one move (guarded) -> a record of the frame

choose() may be called again in the same frame after the engine refused a move; then the state is kept, only the
move is recomputed with the refused cell excluded. The plan is advanced in on_result(), never in choose().

The Q model is anything with values(frame, inputs) -> (5,): HeuristicQ now, the NumPy network later.
"""

import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from candidate.agent.data.types import Direction, MoveResult
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import step
from candidate.agent.q import chooser, movers, safety
from candidate.agent.q.config import Q, QConfig
from candidate.agent.q.frame import Frame, Hints
from candidate.agent.q.heuristic import HeuristicQ
from candidate.agent.q.inputs import QInputs, build_inputs
from candidate.agent.q.snapshot import Snapshot, build_snapshot
from candidate.agent.q.types import State
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
    def __init__(self, q_model=None, cfg: QConfig = Q, build_inputs: bool = True) -> None:
        self.cfg = cfg
        self.q_model = q_model if q_model is not None else HeuristicQ(cfg)
        self.build_inputs = build_inputs
        self.reset()

    def reset(self) -> None:
        """A new match."""
        self.state = State.PATROL
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
            frame = Frame(view, self._snap, self.hints, self.cfg)         # same decision, a different move
            self.stats["retries"] += 1
            return self._guard(frame, movers.move_for(self.state, frame))
        return self._decide(view)

    def on_result(self, view: TurnView, direction: Direction, result: MoveResult) -> None:
        if result == MoveResult.INVALID:
            return
        snap = self._snap
        if result in (MoveResult.DIED, MoveResult.CAPTURED):
            self.hints.outward = None
        elif snap is not None and not snap.exposed and not bitboard.has_cell(snap.my_land_bb, step(snap.my_pos, direction)):
            self.hints.outward = Direction(direction)                      # just left my land: remember which way
        self.stats[f"result_{result.name}"] += 1

    def debug_state(self) -> str:
        return self.state.name + (f" !{self._record.forced}" if self._record and self._record.forced else "")

    # ---- one frame -----------------------------------------------------------------------------------------------------

    def _decide(self, view: TurnView) -> Direction:
        cfg = self.cfg
        mark = time.perf_counter()
        snap = build_snapshot(view, cfg)
        frame = Frame(view, snap, self.hints, cfg)
        self._tick("snapshot", mark)

        mark = time.perf_counter()
        inputs = build_inputs(view, snap, self.state, cfg) if self.build_inputs else None
        self._tick("inputs", mark)

        mark = time.perf_counter()
        values = np.asarray(self.q_model.values(frame, inputs), dtype=np.float64)
        valid = movers.valid_states(frame)
        died = Event.SELF_DIED in view.events or Event.KILLED_BY_OPPONENT in view.events
        choice = chooser.choose(values, valid, self.state, died=died, trail_len=snap.trail_len, margin=snap.d7, cfg=cfg)
        self._tick("q", mark)

        mark = time.perf_counter()
        previous, self.state = self.state, choice.state
        forced = choice.forced
        if not snap.exposed:
            self.hints.outward = None
        move = self._guard(frame, movers.move_for(self.state, frame))
        if cfg.safety_lookahead and self.state != State.RETURN and forced != "death":
            move, forced = self._look_ahead(frame, move, forced)
        self._tick("move", mark)

        self._turn, self._snap = view.obs.turn, snap
        self._record = self.last = FrameRecord(view.obs.turn, inputs, snap, values, valid, previous, self.state, forced, move)
        self.stats[f"state_{self.state.name}"] += 1
        if forced:
            self.stats[f"forced_{forced}"] += 1
        if previous != self.state:
            self.stats["switches"] += 1
        return move

    def _look_ahead(self, frame: Frame, move: Direction, forced: str | None) -> tuple[Direction, str | None]:
        """Refuse a move that would leave the safety margin below the floor: go home instead if there is a trail to go
        home along, else stay inside my land. Nothing Q or the switching cost says can outvote this."""
        snap = frame.snap
        if safety.margin_after(snap, step(frame.pos, move), self.cfg) >= self.cfg.safety_floor:
            return move, forced
        self.state = State.RETURN if snap.exposed else State.PATROL
        return self._guard(frame, movers.move_for(self.state, frame)), "lookahead"

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
