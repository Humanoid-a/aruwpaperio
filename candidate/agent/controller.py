"""The controller that ties a strategy to the run-time layer.

Controller.take_turn(game) is what the framework's CandidateController
delegates to on every call. It never raises: an exception inside take_turn
makes the engine forfeit the turn, so any failure is recorded as an anomaly
and answered with a fallback move instead.

Strategy interface (anything with these methods works; agent.probe.ProbePolicy
is one):

    reset()                                   a new match has started
    choose(view: TurnView) -> Direction       pick a move; must not advance the plan
    on_result(view, direction, result)        the engine accepted (or refused) the move
    debug_state() -> str                      a short label for the GUI
"""

import time
import traceback
from typing import Protocol

from candidate.agent import adapter
from candidate.agent.config import BOARD_WIDTH
from candidate.agent.data.types import Direction, MoveResult
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.runtime import AgentRuntime, Event, TurnView


class Strategy(Protocol):
    def reset(self) -> None: ...
    def choose(self, view: TurnView) -> Direction: ...
    def on_result(self, view: TurnView, direction: Direction, result: MoveResult) -> None: ...
    def debug_state(self) -> str: ...


def _percent(p: float) -> str:
    return f"{p:.1%}" if p >= 0.001 else f"{p:.3%}"


def belief_legend(k: int) -> str:
    return f"[belief] each turn of mine: the {k} cells the opponent is most likely on, as (x,y) probability"


def belief_line(view: TurnView, k: int) -> str | None:
    """One line saying where the opponent most probably is: the k likeliest cells with their
    probabilities, as (x,y) like the framework's GridPosition. None if there is no belief."""
    belief = view.belief
    if belief is None:
        return None
    cells = "  ".join(f"({c % BOARD_WIDTH},{c // BOARD_WIDTH}) {_percent(p)}" for c, p in belief.top_positions(k))
    where = "in view" if belief.sighted_at is not None else "out of view"
    return f"[belief] turn {view.obs.turn:4d} | {where:11s} | {cells}"


def belief_description(view: TurnView) -> list[str] | None:
    """Four lines describing the whole belief this turn: what it holds, what it says about each side's
    territory, and what this frame's evidence did to it. No cell lists (belief_line has those).
    None if there is no belief."""
    belief = view.belief
    summary = None if belief is None else belief.summary
    if summary is None:
        return None
    s = summary
    pct = lambda p: f"{100 * max(p, 0.0):.1f}%"  # noqa: E731
    tag = f"[belief state] turn {view.obs.turn:4d}"
    counts = ", ".join(map(str, s.opp_counts)) if s.opp_counts else "-"
    exposed = "n/a" if s.exposed_share is None else pct(s.exposed_share)
    events = ", ".join(e.name for e in belief.events) or "none"
    return [
        f"{tag} | {belief.n_hypotheses} hypotheses hold {pct(belief.hypothesis_mass)} of the belief, the residual "
        f"{pct(belief.residual_mass)} | {int(belief.support.sum())} possible cells, entropy {belief.entropy:.2f} nats",
        f"{tag} |   opponent: territory count {counts} on every hypothesis (engine says {view.obs.opp_count}); "
        f"trail out on {exposed} of the mass (mean {s.trail_mean:.1f} cells, longest {s.trail_longest}); "
        f"land cells fully known for {pct(s.opp_land_known)} of the mass",
        f"{tag} |   me: territory count {s.my_count}; {s.my_views} view(s) of my land across the hypotheses, "
        f"{pct(s.my_views_exact)} of the mass agrees with that count",
        f"{tag} |   this frame: {s.successors} successors considered; cut by its count {s.cut_by_count}, by what I can see "
        f"{s.cut_by_window}, by my count {s.cut_by_my_count}; merged {s.merged}; {pct(s.pruned_mass)} pruned into the "
        f"residual; rehydrated {s.rehydrated}; events: {events}",
    ]


class Controller:
    def __init__(
        self,
        strategy: Strategy,
        diagnostics: Diagnostics | None = None,
        report_every: int = 250,
        show_belief: int = 0,
        describe_belief: bool = False,
    ) -> None:
        """`show_belief` is how many of the opponent's most probable cells to print (through the
        diagnostics' echo) on every one of my turns; 0 prints none. `describe_belief` also prints
        belief_description() on every turn."""
        self.diag = diagnostics if diagnostics is not None else Diagnostics()
        self.runtime = AgentRuntime(self.diag)
        self.strategy = strategy
        self.report_every = report_every
        self.show_belief = show_belief
        self.describe_belief = describe_belief

    def take_turn(self, game) -> None:
        started = time.perf_counter()
        view: TurnView | None = None
        direction: Direction | None = None
        result: MoveResult | None = None
        try:
            view = self.runtime.begin_turn(game)
            if Event.MATCH_STARTED in view.events:
                self.strategy.reset()
            mark = time.perf_counter()
            direction = self.strategy.choose(view)
            self.diag.time("policy", (time.perf_counter() - mark) * 1000)
            result = self.runtime.move(game, direction)
            self.strategy.on_result(view, direction, result)
        except Exception:  # noqa: BLE001 - a raised exception would forfeit the turn
            self._recover(game, view, traceback.format_exc())
        finally:
            try:
                self._after_turn(view, direction, result, started)
            except Exception:  # noqa: BLE001 - reporting must never cost the turn either
                pass

    def debug_state(self) -> str:
        label = self.strategy.debug_state()
        errors = self.diag.counts["anomaly_error"]
        return f"{label} | err {errors}" if errors else label

    # ---- failure handling and reporting -----------------------------------

    def _recover(self, game, view: TurnView | None, details: str) -> None:
        turn = view.obs.turn if view is not None else -1
        try:
            self.diag.anomaly("CONTROLLER_EXCEPTION", turn, details.strip().splitlines()[-1])
            self.diag.count("exceptions")
        except Exception:  # noqa: BLE001
            pass
        try:
            if view is not None:
                for direction in self._recovery_moves(view):
                    if self.runtime.move(game, direction) != MoveResult.INVALID:
                        return
        except Exception:  # noqa: BLE001 - fall through to the framework-only fallback
            pass
        try:
            adapter.fallback_move(game)
        except Exception:  # noqa: BLE001 - nothing else can be done
            pass

    @staticmethod
    def _recovery_moves(view: TurnView) -> list[Direction]:
        """The way home first (it closes a loop), then any safe move, then any legal one."""
        moves: list[Direction] = []
        try:
            path = view.home_path
            if path:
                home = view.direction_of(path[0])
                if home in view.legal_moves:
                    moves.append(home)
        except Exception:  # noqa: BLE001 - a broken helper must not stop the simpler fallbacks
            pass
        moves += [d for d in (*view.safe_moves, *view.legal_moves) if d not in moves]
        return moves

    def _after_turn(self, view, direction, result, started: float) -> None:
        elapsed = (time.perf_counter() - started) * 1000
        self.diag.time("total", elapsed)
        if view is None:
            return
        obs = view.obs
        if elapsed > self.diag.slow_turn_ms:
            self.diag.anomaly("SLOW_TURN", obs.turn, f"{elapsed:.1f} ms", "info")
        self.diag.record(
            {
                "turn": obs.turn,
                "retry": view.retry,
                "pos": obs.my_pos,
                "trail": int(obs.my_trail.size),
                "mine": obs.my_count,
                "theirs": obs.opp_count,
                "seen": obs.opp_seen_at,
                "dir": None if direction is None else getattr(direction, "name", repr(direction)),
                "result": None if result is None else result.name,
                "events": [e.name for e in view.events],
                "belief": None
                if view.belief is None
                else [view.belief.n_hypotheses, round(view.belief.entropy, 3), round(view.belief.most_likely()[1], 4)],
                "state": self.strategy.debug_state(),
                "ms": round(elapsed, 3),
            }
        )
        if view.retry:
            return
        if self.show_belief > 0 and self.diag.echo is not None:
            line = belief_line(view, self.show_belief)
            if line is not None:
                if Event.MATCH_STARTED in view.events:
                    self.diag.echo(belief_legend(self.show_belief))
                self.diag.echo(line)
        if self.describe_belief and self.diag.echo is not None:
            for text in belief_description(view) or ():
                self.diag.echo(text)
        if self.diag.echo is not None and obs.turn > 0 and obs.turn % self.report_every == 0:
            self.diag.echo(self.diag.summary_line(obs.turn))
        if result is not None and result != MoveResult.INVALID and obs.turns_left == 1:
            self.diag.finish_match()
