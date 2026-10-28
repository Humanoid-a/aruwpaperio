"""The framework's entry point for the candidate's own controller (shown as
"Candidate Controller" in the GUI).

This file stays thin. The real work lives in candidate/agent:

    adapter.py      framework types <-> our types (the only place that translates)
    runtime.py      per-match memory, event inference, prediction checks; the API
    controller.py   one turn: begin_turn -> strategy.choose -> move, never raising
    behavior/       the five action-states, the chooser and the safety rules
                    (knows nothing of Q)
    q/              the Q model (the hand-set stand-in or a trained network,
                    q/network.py), the game recorder (q/record.py) and QStrategy,
                    the strategy plugged in here, which joins the two; see
                    docs/Q_BUILD_PLAN.md
    probe.py        a loop-maker for testing the run-time layer in a live
                    match, not a game strategy (CandidateController(ProbePolicy()))
    diagnostics.py  counters, timings, anomalies, the per-turn log and reports

While a match runs, the console that launched the GUI shows, on each of the
candidate's turns, the five cells the opponent is most likely on with their
probabilities (show_belief=0 turns that off) and four "[belief state]" lines
describing the whole belief (describe_belief=False turns those off), plus progress
lines and a final report; a per-turn log is written to candidate/runtime_logs/.

Every game played this way is also recorded for training, one file per game in
candidate/q_training/raw/gui/ (about 0.5 MB each; record=False turns it off). Code that
brings its own Diagnostics (tests, runtime_check) does not record unless asked.

weights chooses the Q model: None is the hand-set stand-in; a path is a trained
network (a .pt file, e.g. online15's epoch 25, the default, or its latest.pt),
run on the GPU if PyTorch can use it, else the CPU. The GUI uses
DEFAULT_WEIGHTS, or the file named by the environment variable CANDIDATE_Q_WEIGHTS
if that is set (to try a network in the GUI without editing this file); both are
read when the controller is built. PyTorch is imported only when a network is
asked for, and if it is missing or the file cannot be loaded the controller falls
back to the stand-in (and says so), so the GUI never fails to start. A network
made with different input code than the current one (its inputs version differs)
still loads, with a warning: its inputs may no longer mean what it was trained on.

Replace the strategy by passing another object with the same four methods (see
agent.controller.Strategy).
"""

import os
from pathlib import Path

from candidate.agent.controller import Controller, Strategy
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.q.heuristic import HeuristicQ
from candidate.agent.q.record import GameRecorder
from candidate.agent.q.strategy import QStrategy
from territorygame.api.agent_controller import AgentController
from territorygame.api.game_api import GameApi

_HERE = Path(__file__).resolve().parent
_LOG_DIR = _HERE / "runtime_logs"
RECORD_DIR = _HERE / "q_training" / "raw" / "gui"          # see candidate/offline/locations.py
# The submitted networks: online15's epoch 25, Q and K (both tracked in git; see q_training/.gitignore for the other
# networks kept, e.g. online15/latest.pt, its newest). None is the hand-set stand-in (Q) or the hand-set K. To try
# another network in the GUI, set $CANDIDATE_Q_WEIGHTS / $CANDIDATE_K_WEIGHTS instead of editing these.
DEFAULT_WEIGHTS: Path | None = Path(__file__).parent / "q_training/online/online15/epoch_025/weights.pt"
DEFAULT_K_WEIGHTS: Path | None = Path(__file__).parent / "q_training/online/online15/epoch_025/k.npz"
WEIGHTS_VARIABLE = "CANDIDATE_Q_WEIGHTS"     # if set, the weights file the GUI's controller uses instead
K_WEIGHTS_VARIABLE = "CANDIDATE_K_WEIGHTS"   # if set, the K file the GUI's controller uses instead
_DEFAULT = object()                          # "weights not given": DEFAULT_WEIGHTS / the variable, read at build time


def default_k_weights() -> str | Path | None:
    """The motion model the GUI's controller uses: $CANDIDATE_K_WEIGHTS if set and not empty, else DEFAULT_K_WEIGHTS."""
    return os.environ.get(K_WEIGHTS_VARIABLE) or DEFAULT_K_WEIGHTS


def make_belief(k_weights, echo=None):
    """The belief filter with the K in `k_weights` (None: the runtime's default filter, with the hand-set K)."""
    if k_weights is None:
        return None
    from candidate.agent.belief.filter import BeliefFilter
    from candidate.agent.belief.neural import NeuralMotionModel, load_motion

    motion = load_motion(k_weights, echo)
    if echo is not None and isinstance(motion, NeuralMotionModel):
        echo(f"[candidate] belief: the motion model {k_weights}")
    return BeliefFilter(motion=motion)


def default_weights() -> str | Path | None:
    """The weights the GUI's controller uses: $CANDIDATE_Q_WEIGHTS if set and not empty, else DEFAULT_WEIGHTS."""
    return os.environ.get(WEIGHTS_VARIABLE) or DEFAULT_WEIGHTS


def _echo(message: str) -> None:
    print(message, flush=True)


def load_q(weights, echo=_echo):
    """The Q model for a weights setting: the stand-in for None, else the network in that file (or the stand-in if
    the file cannot be loaded)."""
    if weights is None:
        return HeuristicQ()
    try:
        from candidate.agent.q.network import NetworkQ      # imports PyTorch, so only when a network is asked for

        q = NetworkQ.load(weights)
    except Exception as error:  # noqa: BLE001 - a bad weights file must not stop the controller
        if echo is not None:
            echo(f"[candidate] could not load Q weights {weights}: {error!r}; playing with the hand-set stand-in")
        return HeuristicQ()
    if echo is not None:
        echo(f"[candidate] Q: the network {weights} ({'+'.join(q.model.branches)}) on {q.device}")
        if not q.inputs_match:
            echo(f"[candidate] warning: {weights} was trained on inputs of another code version "
                 f"({q.inputs_version} vs {q.current_version}); retrain it if the inputs have changed")
    return q


class CandidateController(AgentController):
    def __init__(
        self,
        strategy: Strategy | None = None,
        diagnostics: Diagnostics | None = None,
        show_belief: int = 5,
        describe_belief: bool = True,
        record: bool | None = None,
        weights=_DEFAULT,
        k_weights=_DEFAULT,
    ) -> None:
        if weights is _DEFAULT:
            weights = default_weights()
        if k_weights is _DEFAULT:
            k_weights = default_k_weights()
        if record is None:
            record = diagnostics is None and strategy is None          # the GUI's own construction
        if diagnostics is None:
            diagnostics = Diagnostics(echo=_echo, log_dir=_LOG_DIR)
        if strategy is None:
            recorder = GameRecorder(RECORD_DIR, label="gui") if record else None
            strategy = QStrategy(load_q(weights, diagnostics.echo), recorder=recorder)
        self._agent = Controller(strategy, diagnostics, show_belief=show_belief, describe_belief=describe_belief,
                                 belief=make_belief(k_weights, diagnostics.echo))

    def take_turn(self, game: GameApi) -> None:
        self._agent.take_turn(game)

    def get_debug_state(self) -> str | None:
        return self._agent.debug_state()
