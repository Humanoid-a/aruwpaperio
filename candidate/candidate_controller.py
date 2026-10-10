"""The framework's entry point for the candidate's own controller (shown as
"Candidate Controller" in the GUI).

This file stays thin. The real work lives in candidate/agent:

    adapter.py      framework types <-> our types (the only place that translates)
    runtime.py      per-match memory, event inference, prediction checks; the API
    controller.py   one turn: begin_turn -> strategy.choose -> move, never raising
    q/              the strategy plugged in: the five action-states, the safety
                    rules and (for now) a hand-set stand-in for the Q model
                    (see docs/Q_BUILD_PLAN.md)
    probe.py        a loop-maker for testing the run-time layer in a live
                    match, not a game strategy (CandidateController(ProbePolicy()))
    diagnostics.py  counters, timings, anomalies, the per-turn log and reports

While a match runs, the console that launched the GUI shows, on each of the
candidate's turns, the five cells the opponent is most likely on with their
probabilities (show_belief=0 turns that off) and four "[belief state]" lines
describing the whole belief (describe_belief=False turns those off), plus progress
lines and a final report; a per-turn log is written to candidate/runtime_logs/. Replace the strategy by passing another object with
the same four methods (see agent.controller.Strategy).
"""

from pathlib import Path

from candidate.agent.controller import Controller, Strategy
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.q.strategy import QStrategy
from territorygame.api.agent_controller import AgentController
from territorygame.api.game_api import GameApi

_LOG_DIR = Path(__file__).resolve().parent / "runtime_logs"


def _echo(message: str) -> None:
    print(message, flush=True)


class CandidateController(AgentController):
    def __init__(
        self,
        strategy: Strategy | None = None,
        diagnostics: Diagnostics | None = None,
        show_belief: int = 5,
        describe_belief: bool = True,
    ) -> None:
        if diagnostics is None:
            diagnostics = Diagnostics(echo=_echo, log_dir=_LOG_DIR)
        self._agent = Controller(
            strategy if strategy is not None else QStrategy(),
            diagnostics,
            show_belief=show_belief,
            describe_belief=describe_belief,
        )

    def take_turn(self, game: GameApi) -> None:
        self._agent.take_turn(game)

    def get_debug_state(self) -> str | None:
        return self._agent.debug_state()
