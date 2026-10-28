"""Replaying a recorded game to regenerate Q's inputs with the current code.

A recorded game stores the inputs as they were during play, which is what training must use. If the belief, the
memory or an input builder has changed since (the record's inputs_version differs from version.inputs_version()), the
stored inputs are stale. The sidecar's move tapes fix that: replaying both players' moves through the engine passes
through exactly the same boards, and running the current pipeline (adapter, runtime, belief, snapshot, input builders)
on my side regenerates the inputs as the current code would have seen them in that very game. The states and moves
are the recorded ones; nothing is decided again.
"""

import json
from pathlib import Path

import numpy as np

from candidate.agent.controller import Controller
from candidate.agent.data.types import Direction
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.behavior.snapshot import build_snapshot
from candidate.agent.behavior.states import N_STATES, State
from candidate.agent.q.inputs import build_inputs
from candidate.agent.q.record import GameRecorder, load_record
from candidate.agent.q.strategy import FrameRecord, QStrategy
from candidate.offline import match

_LETTER_TO_DIRECTION = {"N": Direction.NORTH, "E": Direction.EAST, "S": Direction.SOUTH, "W": Direction.WEST}


class _ReplayStrategy(QStrategy):
    """Rebuilds the snapshot and inputs every frame exactly as QStrategy does, with the recorded previous state, and
    plays the recorded move. Records into its own GameRecorder."""

    def __init__(self, tape: str, previous: dict[int, int], states: dict[int, int], recorder: GameRecorder) -> None:
        super().__init__(recorder=recorder)
        self._tape, self._previous, self._states = tape, previous, states

    def choose(self, view):
        turn = view.obs.turn
        letter = self._tape[turn]
        if letter == "-":
            raise match.ReplayDiverged(f"turn {turn}: my recorded turn had no accepted move")
        if view.retry:
            raise match.ReplayDiverged(f"turn {turn}: the engine refused a recorded move")
        snap = build_snapshot(view, self.behavior)
        current = State(self._previous.get(turn, int(self.state)))
        inputs = build_inputs(view, snap, current, self.cfg)
        state = State(self._states.get(turn, int(current)))
        direction = _LETTER_TO_DIRECTION[letter]
        record = FrameRecord(turn, inputs, snap, np.zeros(N_STATES), np.zeros(N_STATES, dtype=bool), current, state, None,
                             direction)
        self._record = self.last = record
        self._snap = snap
        if not self.recorder.active:
            self.recorder.start(view, "replay", 0.0)
        self.recorder.add(record, view, False)
        self.state = state
        self._last_turn = view.obs.turns_left <= 1
        return direction


def replay(record_path: str | Path, out_dir: str | Path) -> Path:
    """Replay the game of `record_path` (its sidecar must exist) and write a regenerated record into `out_dir`.
    Returns the new record's path. Raises ReplayDiverged if the replay does not follow the recording."""
    record_path = Path(record_path)
    sidecar = json.loads(record_path.with_suffix(".json").read_text(encoding="utf-8"))
    meta, arrays = load_record(record_path)
    seat = sidecar["config"]["seat"]
    tapes = sidecar["tapes"]
    previous = {int(t): int(p) for t, p in zip(arrays["turn"], arrays["previous"])}
    states = {int(t): int(s) for t, s in zip(arrays["turn"], arrays["state"])}

    recorder = GameRecorder(out_dir, label=meta.get("label", "") + "-replay",
                            meta={k: meta[k] for k in ("opponent", "seat", "close", "seed", "model", "oracle", "k_model",
                                                       "spawn", "warmup_turns", "warmup_agent") if k in meta})
    strategy = _ReplayStrategy(tapes[seat], previous, states, recorder)
    truth = belief = None
    if meta.get("oracle"):                       # it was played with the true world: regenerate its inputs the same way
        from candidate.offline.oracle import EngineTruth, OracleBelief

        truth = EngineTruth()
        belief = OracleBelief(truth, seat)
    elif meta.get("k_model") and Path(meta["k_model"]).exists():    # played with a trained K: the same K again
        from candidate.agent.belief.filter import BeliefFilter
        from candidate.agent.belief.neural import load_motion

        belief = BeliefFilter(motion=load_motion(meta["k_model"]))
    controllers = [match.Replayer(tapes[0]), match.Replayer(tapes[1])]
    controllers[seat] = match.AgentAdapter(Controller(strategy, Diagnostics(), belief=belief))
    respawns = sidecar["config"]["respawns"]
    result = match.play(controllers, sidecar["config"]["turns"], tuple(map(tuple, respawns)) if respawns else match.FAR,
                        on_engine=truth.attach if truth is not None else None)
    strategy.close()
    if recorder.last_path is None:
        raise match.ReplayDiverged(f"the replay wrote no record ({recorder.last_error})")
    if list(result.counts) != list(sidecar["final"]["counts"]):
        raise match.ReplayDiverged(f"final counts {result.counts} differ from the recording's {sidecar['final']['counts']}")
    new_path = recorder.last_path
    new_sidecar = dict(sidecar, record=str(new_path), replayed_from=str(record_path))
    new_path.with_suffix(".json").write_text(json.dumps(new_sidecar, indent=1), encoding="utf-8")
    return new_path
