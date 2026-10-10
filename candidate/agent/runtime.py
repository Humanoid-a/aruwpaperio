"""The run-time API between the framework and the agent's decision code.

A strategy (a state machine, a probe, anything) never touches framework types.
Each turn it gets a TurnView, decides a Direction, and hands that to
AgentRuntime.move(). The runtime does the bookkeeping the strategy should not
have to:

  * translates the framework's observation (agent.adapter) and keeps the
    MemoryMap up to date, including what deaths and captures do to cells
    that are out of view;
  * copes with the engine's calling pattern: take_turn is called again after
    an INVALID move within the same turn, and a new match can start on the
    same controller object (the GUI's Reset);
  * infers what happened between this turn and the last (my own death, an
    opponent kill, a capture, a kill of the opponent, land lost to the
    opponent) and says so in TurnView.events;
  * predicts every move before making it and compares the prediction with
    what the engine actually does, and reports any disagreement to
    Diagnostics;
  * runs the belief over the hidden opponent in the order of a frame: at
    begin_turn() the observation goes into memory and then into the belief
    (generate, cut, weigh, settle; see agent.belief.filter), and at the end
    of move(), once the engine has accepted the move, the move is applied to
    every hypothesis. At the start of a match both the memory and the belief
    are set to the known starting state.

Typical use (agent.controller.Controller does exactly this):

    view = runtime.begin_turn(game)
    direction = strategy.choose(view)
    result = runtime.move(game, direction)

begin_turn() and move() raise on a genuine bug. The controller turns that
into a fallback move, because an exception in take_turn costs the turn.
"""

import time
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property

import numpy as np

from candidate.agent import adapter
from candidate.agent.belief.after_move import MyMove
from candidate.agent.belief.filter import BeliefFilter
from candidate.agent.belief.state import is_fresh_start
from candidate.agent.data.cells import MY_TERRITORY, MY_TRAIL
from candidate.agent.data.belief_view import BeliefEvent, BeliefView
from candidate.agent.data.memory import MemoryMap
from candidate.agent.data.observation import Observation
from candidate.agent.data.types import Direction, MoveResult
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.distance import shortest_path
from candidate.agent.geometry.fill import BOARD
from candidate.agent.geometry.grid import direction_to, step
from candidate.agent.prediction import Prediction, predict_move
from candidate.agent.rules.match import MatchRules


class Event(Enum):
    """Things inferred to have happened since the previous turn."""

    MATCH_STARTED = auto()
    SELF_DIED = auto()            # my last move stepped on my own trail
    KILLED_BY_OPPONENT = auto()   # the opponent stepped on my trail
    CAPTURED = auto()             # my last move closed my trail
    KILLED_OPPONENT = auto()      # my last move stepped on the opponent's trail
    TERRITORY_LOST = auto()       # my territory shrank without a death: the opponent captured some
    FORFEITED_TURN = auto()       # no move was accepted last turn


class TurnView:
    """Everything a strategy may use this turn. Cheap facts are plain
    attributes; derived ones are computed on first use and then cached."""

    def __init__(
        self,
        obs: Observation,
        memory: MemoryMap,
        rules: MatchRules,
        events: tuple[Event, ...],
        opp_last_seen: tuple[int, int] | None,
        territory_drift: int,
        belief: BeliefView | None = None,
    ) -> None:
        self.obs = obs
        self.belief = belief                  # what I currently believe about the hidden opponent
        self.memory = memory
        self.rules = rules
        self.events = events
        self.opp_last_seen = opp_last_seen    # (cell, turn) of the latest sighting, or None
        self.territory_drift = territory_drift  # cells remembered as mine minus the engine's count
        self.retry = False                    # True when this is a repeat call after an INVALID move
        self.blocked: frozenset[int] = frozenset()  # destinations the engine already refused this turn

    # ---- simple facts ----------------------------------------------------

    @property
    def exposed(self) -> bool:
        """Is my trail out in the open?"""
        return self.obs.my_trail.size > 0

    @property
    def lead(self) -> int:
        """My territory count minus the opponent's."""
        return self.obs.my_count - self.obs.opp_count

    @cached_property
    def my_trail_bb(self) -> np.ndarray:
        return bitboard.from_cells(int(c) for c in self.obs.my_trail)

    @cached_property
    def my_territory_bb(self) -> np.ndarray:
        """Cells remembered as mine (may be stale outside the window)."""
        return bitboard.from_mask(self.memory.territory == MY_TERRITORY)

    # ---- moves -----------------------------------------------------------

    @cached_property
    def legal_moves(self) -> tuple[Direction, ...]:
        """Moves the engine will not refuse: on the board and not onto the
        opponent's agent. Includes moves onto my own trail (death)."""
        out = []
        for direction in Direction:
            dest = step(self.obs.my_pos, direction)
            if dest is not None and dest != self.obs.opp_seen_at and dest not in self.blocked:
                out.append(direction)
        return tuple(out)

    @cached_property
    def safe_moves(self) -> tuple[Direction, ...]:
        """Legal moves that do not step onto my own trail."""
        return tuple(
            d for d in self.legal_moves if self.obs.window_cell(step(self.obs.my_pos, d))[1] != MY_TRAIL
        )

    def predict(self, direction: Direction) -> Prediction:
        """What the engine will do with this move, as far as I can tell."""
        return predict_move(self.obs, self.my_territory_bb, self.rules, direction)

    # ---- getting home ----------------------------------------------------

    @cached_property
    def home_path(self) -> list[int] | None:
        """A shortest route from my head onto my own territory, as the cells to
        step onto, never crossing my trail or the opponent. [] if I am already
        on my land; None if no route is known."""
        if not self.my_territory_bb.any():
            return None
        passable = BOARD & ~self.my_trail_bb
        if self.obs.opp_seen_at is not None:
            passable = passable & ~bitboard.from_cells([self.obs.opp_seen_at])
        return shortest_path(self.obs.my_pos, self.my_territory_bb, passable)

    @property
    def home_distance(self) -> int | None:
        path = self.home_path
        return None if path is None else len(path)

    def direction_of(self, cell: int) -> Direction | None:
        """The move from my agent onto an adjacent cell."""
        return direction_to(self.obs.my_pos, cell)


@dataclass(eq=False)
class _Turn:
    """What happened on one turn, filled in as it goes."""

    obs: Observation
    direction: Direction | None = None
    prediction: Prediction | None = None
    result: MoveResult | None = None   # None until a move is accepted


class AgentRuntime:
    def __init__(self, diagnostics: Diagnostics | None = None, belief: BeliefFilter | None = None) -> None:
        self.diag = diagnostics if diagnostics is not None else Diagnostics()
        self.belief = belief if belief is not None else BeliefFilter()
        self.memory = MemoryMap.new()
        self.rules: MatchRules | None = None
        self._view: TurnView | None = None
        self._turn: _Turn | None = None
        self._turn_done = True
        self._blocked: set[int] = set()
        self._t0 = 0
        self._last_turns_left = 0
        self._opp_seen: tuple[int, int] | None = None

    # ---- start of turn ---------------------------------------------------

    def begin_turn(self, game) -> TurnView:
        started = time.perf_counter()
        turns_left = game.get_remaining_turns()

        same_turn = self._view is not None and not self._turn_done and turns_left == self._last_turns_left
        if same_turn and self._unchanged_since_last_call(game):
            return self._repeat_call()

        # A new match shows as more turns left than before; so does a reset that lands
        # on the same turn count with a different board (the state would not match).
        new_match = self._view is None or turns_left > self._last_turns_left or same_turn
        skipped = 1 if new_match else self._last_turns_left - turns_left
        if new_match:
            self._start_match(game, turns_left)
        turn = self._t0 - turns_left

        mark = time.perf_counter()
        obs = adapter.observation_from_game(game, turn, self._last_result(new_match))
        self.diag.time("adapter", (time.perf_counter() - mark) * 1000)

        if new_match:
            self.rules = MatchRules.mirrored(obs.respawn)
            self.belief.begin_match(self.rules, obs)
            self._seed_memory(obs)
            events = [Event.MATCH_STARTED]
        else:
            events = self._reconcile(obs, skipped)

        # The belief has already had my last move applied (at the end of the last frame, see move()); this
        # observation brings in the opponent's reply.
        self.memory.update(obs)
        mark = time.perf_counter()
        belief_view = self.belief.observe(
            obs, self.memory, Event.KILLED_BY_OPPONENT in events, force_reseed=(skipped != 1 and not new_match)
        )
        self.diag.time("belief", (time.perf_counter() - mark) * 1000)
        self._note_belief(belief_view)

        if obs.opp_seen_at is not None:
            self._opp_seen = (obs.opp_seen_at, turn)
        drift = self.memory.my_territory_count() - obs.my_count
        self.diag.note_drift(drift)

        view = TurnView(obs, self.memory, self.rules, tuple(events), self._opp_seen, drift, belief_view)
        self._view, self._turn = view, _Turn(obs)
        self._turn_done = False
        self._blocked = set()
        self._last_turns_left = turns_left

        for event in events:
            self.diag.count(f"event_{event.name}")
        self.diag.count("turns")
        self.diag.note_turn(turn, self._t0)
        spent = (time.perf_counter() - started) * 1000
        self.diag.time("runtime", spent - self.diag.timings["adapter"][-1] - self.diag.timings["belief"][-1])
        return view

    def _note_belief(self, view: BeliefView) -> None:
        self.diag.sample("belief_hypotheses", view.n_hypotheses)
        self.diag.sample("belief_entropy", view.entropy)
        if BeliefEvent.RESEEDED in view.events:
            self.diag.count("belief_reseeds")
            self.diag.anomaly("BELIEF_RESEEDED", view.turn, "every hypothesis was contradicted; the belief was rebuilt", "info")

    def _last_result(self, new_match: bool) -> MoveResult | None:
        """The engine's answer to my previous move: None at the start of a match,
        INVALID if no move was accepted last turn (a forfeit)."""
        if new_match or self._turn is None:
            return None
        return self._turn.result if self._turn.result is not None else MoveResult.INVALID

    def _unchanged_since_last_call(self, game) -> bool:
        """Cheap check that a call with the same turns-left is really a repeat
        within one turn and not a fresh match."""
        obs = self._view.obs
        return (
            adapter.to_flat(game.get_agent_position()) == obs.my_pos
            and game.get_owned_territory_cell_count() == obs.my_count
            and len(game.get_active_trail()) == obs.my_trail.size
        )

    def _repeat_call(self) -> TurnView:
        """The engine calls take_turn again after an INVALID move in the same
        turn. Nothing has changed, so the translation is reused."""
        view = self._view
        view.retry = True
        view.blocked = frozenset(self._blocked)
        for name in ("legal_moves", "safe_moves"):
            view.__dict__.pop(name, None)  # depend on `blocked`
        self.diag.count("retries")
        return view

    def _start_match(self, game, turns_left: int) -> None:
        self.diag.start_match()
        adapter.check_board(game)
        self.memory = MemoryMap.new()
        self._t0 = turns_left
        self._turn = None
        self._opp_seen = None

    def _seed_memory(self, obs: Observation) -> None:
        """The world as it is at the start of a match: both start lands are known to be where they are, even
        the opponent's, which is out of view. Skipped when the first observation does not look like a start
        (a controller switched on mid-match), since then nothing of this is true."""
        if is_fresh_start(self.rules, obs):
            self.memory.reset_my_land(self.rules.me.start_cells, obs.turn)
            self.memory.reset_opponent_land(self.rules.opp.start_cells, obs.turn)

    # ---- what happened since last turn ----------------------------------

    def _reconcile(self, obs: Observation, skipped: int) -> list[Event]:
        """Compare this turn's observation with what the previous move should
        have produced, keep the memory in step, and name what happened."""
        events: list[Event] = []
        prev = self._turn
        if prev is None:
            return events
        if skipped != 1:
            self.diag.anomaly("SKIPPED_TURNS", obs.turn, f"turns left dropped by {skipped} since the last call")
            return events

        rules, before = self.rules, prev.obs
        start_land = rules.me.start_cells
        result, pred = prev.result, prev.prediction
        chain = (rules.me.respawn, rules.me.alt_respawn)
        actual_trail = tuple(int(c) for c in obs.my_trail)

        if result is None:
            events.append(Event.FORFEITED_TURN)
            self.diag.count("forfeits")
            expected_pos, expected_trail, expected_count = before.my_pos, tuple(int(c) for c in before.my_trail), before.my_count
        else:
            dest = pred.dest
            dest_terr, _ = before.window_cell(dest)
            if result == MoveResult.DIED:
                expected_pos, expected_trail, expected_count = None, (), len(start_land)
            elif result == MoveResult.CAPTURED:
                gain = pred.gain if pred.result == MoveResult.CAPTURED else None
                expected_pos, expected_trail = dest, ()
                expected_count = None if gain is None else before.my_count + gain
            else:
                trail = tuple(int(c) for c in before.my_trail)
                expected_pos = dest
                expected_trail = trail if dest_terr == MY_TERRITORY else trail + (dest,)
                expected_count = before.my_count

        died = False
        if result == MoveResult.DIED:
            died = True
            events.append(Event.SELF_DIED)
            if obs.my_pos in chain and actual_trail == ():
                self.diag.count("state_ok")
            else:
                self.diag.count("state_bad")
                self.diag.anomaly("STATE_MISMATCH", obs.turn, f"after dying I expected to be home with no trail, found {obs.my_pos}, trail {len(actual_trail)}")
        elif (obs.my_pos, actual_trail) == (expected_pos, expected_trail):
            self.diag.count("state_ok")
        else:
            was_killed = bool(expected_trail) and actual_trail == () and obs.my_pos in chain
            if was_killed:
                died = True
                events.append(Event.KILLED_BY_OPPONENT)
                self.diag.count("state_explained_by_kill")
            else:
                self.diag.count("state_bad")
                self.diag.anomaly(
                    "STATE_MISMATCH",
                    obs.turn,
                    f"expected position {expected_pos} trail length {len(expected_trail)}, "
                    f"found {obs.my_pos} trail length {len(actual_trail)}",
                )

        if died:
            self.memory.reset_my_land(start_land, obs.turn)
            expected_count = len(start_land)
        elif result == MoveResult.CAPTURED:
            events.append(Event.CAPTURED)
            if pred.region is not None:
                self.memory.apply_capture(before.my_trail.tolist(), pred.region, obs.turn)
        if result in (MoveResult.NORMAL, MoveResult.CAPTURED) and pred.killed:
            events.append(Event.KILLED_OPPONENT)
            self.memory.reset_opponent_land(rules.opp.start_cells, obs.turn)

        if expected_count is not None and obs.my_count != expected_count:
            delta = obs.my_count - expected_count
            if result == MoveResult.CAPTURED and not died:
                self.diag.count("gain_off")
            if delta < 0 and not died and result in (None, MoveResult.NORMAL):
                events.append(Event.TERRITORY_LOST)
            self.diag.anomaly(
                "COUNT_MISMATCH", obs.turn, f"territory count {obs.my_count}, expected {expected_count} ({delta:+d})", "info"
            )
        elif result == MoveResult.CAPTURED and expected_count is not None and not died:
            self.diag.count("gain_ok")
        return events

    # ---- making the move -------------------------------------------------

    def move(self, game, direction: Direction) -> MoveResult:
        """Predict, make the move, and compare. Returns the engine's answer in
        our own type. After INVALID, the destination is remembered in
        TurnView.blocked and the engine will call again."""
        view, turn = self._view, self._turn
        if view is None or turn is None:
            raise RuntimeError("move() called before begin_turn()")
        prediction = view.predict(direction)

        mark = time.perf_counter()
        result = adapter.send_move(game, direction)
        self.diag.time("move", (time.perf_counter() - mark) * 1000)

        if result == prediction.result:
            self.diag.count("pred_result_ok")
        else:
            self.diag.count("pred_result_bad")
            self.diag.anomaly(
                "RULE_MISMATCH",
                view.obs.turn,
                f"{Direction(direction).name} from {view.obs.my_pos}: predicted {prediction.result.name}, engine said {result.name}",
            )

        if result == MoveResult.INVALID:
            self.diag.count("invalid_moves")
            if prediction.dest is not None:
                self._blocked.add(prediction.dest)
            return result

        turn.direction, turn.prediction, turn.result = Direction(direction), prediction, result
        self._turn_done = True
        self.diag.count(f"move_{result.name}")
        self._apply_to_belief(turn)
        return result

    def _apply_to_belief(self, turn: _Turn) -> None:
        """The end of the frame: the engine has accepted my move, so apply it to every hypothesis. It must not
        raise: the move is made, and an exception would make the controller try to move again."""
        mark = time.perf_counter()
        try:
            self.belief.after_my_move(
                MyMove(turn.result, turn.prediction, tuple(int(c) for c in turn.obs.my_trail), turn.obs.my_count)
            )
        except Exception as error:  # noqa: BLE001
            self.diag.anomaly("BELIEF_ERROR", turn.obs.turn, f"applying my move to the belief failed: {error!r}")
            self.belief.state.needs_reseed = True        # rebuild from the next observation
        self.diag.time("belief_move", (time.perf_counter() - mark) * 1000)
