"""Recording a game for training, from inside the controller (so it works in the GUI as well as headless).

One file per game, `<directory>/<game id>.npz`, with one row per frame (per turn of mine):

    turn            int32     my turn number
    previous        int8      the state I was in when the frame began (what scalars 18-23 one-hot say)
    state           int8      the state taken after the frame's decision (what the label is for)
    forced          int8      0 none, 1 death, 2 safety, 3 look-ahead (a hard rule decided)
    explored        bool      the state came from exploration
    values          float32   (6,) Q's values
    valid           bool      (6,) the validity mask
    direction       int8      the move the engine accepted, Direction 0-3; -1 if none was (a forfeit)
    my_count        int32     my territory count when the frame began
    opp_count       int32     the enemy's
    d5, d7          raw D5 (my gain if I return now) and D7 (safety margin): the label's open-loop credit
    trail_len       int16
    grid_bits       uint8     (386,) the 7 x 21 x 21 grid, bit-packed
    scalars         float32   (23,)
    probmap      float16   (10, 10, 10)

plus `meta`, a JSON string: game id, label, the inputs version (version.py), the Q model's name, the exploration
rate, both respawn cells, the number of turns, whether the game was recorded to its end, and the final counts when
known. The inputs are stored exactly as Q saw them during play (brief: never rebuilt from the true board).
About 2.1 KB a frame before compression (the grid's bits compress well).

The recorder never raises into the game: a failure to write is reported in `last_error` and the game goes on.
"""

import atexit
import json
import os
import time
from pathlib import Path

import numpy as np

from candidate.agent.data.types import Direction
from candidate.agent.behavior.config import BEHAVIOR, BehaviorConfig
from candidate.agent.behavior.states import N_STATES, State
from candidate.agent.q.config import Q, QConfig
from candidate.agent.q.features import NAMES as SCALAR_NAMES
from candidate.agent.q.version import inputs_version

FORCED = {None: 0, "death": 1, "safety": 2, "lookahead": 3}
GRID_SHAPE = (7, 21, 21)
GRID_BITS = int(np.prod(GRID_SHAPE))

_COUNTER = [0]


def pack_grid(grid: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(grid, dtype=np.uint8).reshape(-1))


def unpack_grid(bits: np.ndarray) -> np.ndarray:
    """(..., 386) packed rows -> (..., 7, 21, 21) uint8."""
    bits = np.asarray(bits, dtype=np.uint8)
    flat = np.unpackbits(bits, axis=-1, count=GRID_BITS)
    return flat.reshape(bits.shape[:-1] + GRID_SHAPE)


class GameRecorder:
    def __init__(self, directory: str | Path, label: str = "", meta: dict | None = None, cfg: QConfig = Q,
                 behavior: BehaviorConfig = BEHAVIOR) -> None:
        self.directory = Path(directory)
        self.label = label
        self.extra_meta = dict(meta or {})
        self.cfg, self.behavior = cfg, behavior
        self.last_path: Path | None = None
        self.last_error: str | None = None
        self.saved: list[Path] = []
        self._rows: dict[str, list] | None = None
        self._meta: dict = {}
        atexit.register(self._at_exit)

    @property
    def active(self) -> bool:
        return self._rows is not None

    # ---- a game --------------------------------------------------------------------------------------------------

    def start(self, view, q_name: str = "", explore_rate: float = 0.0) -> None:
        """A new game begins with this view. An unfinished previous game is saved as incomplete first."""
        if self.active:
            self.finish(complete=False)
        _COUNTER[0] += 1
        stamp = time.strftime("%Y%m%d-%H%M%S")
        game_id = f"{stamp}-{os.getpid()}-{_COUNTER[0]}" + (f"-{self.label}" if self.label else "")
        rules = view.rules
        self._meta = dict(
            self.extra_meta,
            game_id=game_id, label=self.label, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
            inputs_version=inputs_version(self.cfg, self.behavior), q_model=q_name, explore_rate=explore_rate,
            my_respawn=int(rules.me.respawn), opp_respawn=int(rules.opp.respawn),
            first_turn=int(view.obs.turn), total_turns=int(view.obs.turn + view.obs.turns_left),
            states=[s.name for s in State], forced_codes={str(k): v for k, v in FORCED.items()},
            scalar_names=list(SCALAR_NAMES), label_window=self.cfg.label_window,
        )
        self._rows = {k: [] for k in (
            "turn", "previous", "state", "forced", "explored", "values", "valid", "direction", "my_count",
            "opp_count", "d5", "d7", "trail_len", "grid_bits", "scalars", "probmap")}

    def add(self, record, view, explored: bool) -> None:
        """One frame, as decided. The move is filled in by accept() once the engine takes it."""
        if not self.active:
            return
        if record.inputs is None:
            raise ValueError("recording needs Q's inputs; build them (QStrategy(build_inputs=True))")
        rows, inputs, snap, obs = self._rows, record.inputs, record.snapshot, view.obs
        rows["turn"].append(obs.turn)
        rows["previous"].append(int(record.previous))
        rows["state"].append(int(record.state))
        rows["forced"].append(FORCED.get(record.forced, 0))
        rows["explored"].append(bool(explored))
        rows["values"].append(np.asarray(record.values, dtype=np.float32))
        rows["valid"].append(np.asarray(record.valid, dtype=bool))
        rows["direction"].append(-1)
        rows["my_count"].append(obs.my_count)
        rows["opp_count"].append(obs.opp_count)
        rows["d5"].append(snap.d5)
        rows["d7"].append(snap.safety_margin)                   # the worst case: the label's "was it safe"
        rows["trail_len"].append(snap.trail_len)
        rows["grid_bits"].append(pack_grid(inputs.grid))
        rows["scalars"].append(np.asarray(inputs.scalars, dtype=np.float32))
        rows["probmap"].append(np.asarray(inputs.probmap, dtype=np.float16))

    def accept(self, direction: Direction) -> None:
        if self.active and self._rows["direction"]:
            self._rows["direction"][-1] = int(direction)

    def finish(self, complete: bool, **extra) -> Path | None:
        """Write the game and stop recording. Returns the file, or None if there was nothing to write or it failed."""
        if not self.active:
            return None
        rows, meta = self._rows, dict(self._meta, complete=bool(complete), **extra)
        self._rows = None
        n = len(rows["turn"])
        if n == 0:
            return None
        meta["frames"] = n
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{meta['game_id']}.npz"
            arrays = dict(
                turn=np.array(rows["turn"], dtype=np.int32), previous=np.array(rows["previous"], dtype=np.int8),
                state=np.array(rows["state"], dtype=np.int8), forced=np.array(rows["forced"], dtype=np.int8),
                explored=np.array(rows["explored"], dtype=bool),
                values=np.stack(rows["values"]).reshape(n, N_STATES), valid=np.stack(rows["valid"]).reshape(n, N_STATES),
                direction=np.array(rows["direction"], dtype=np.int8), my_count=np.array(rows["my_count"], dtype=np.int32),
                opp_count=np.array(rows["opp_count"], dtype=np.int32), d5=np.array(rows["d5"], dtype=np.int32),
                d7=np.array(rows["d7"], dtype=np.float32), trail_len=np.array(rows["trail_len"], dtype=np.int16),
                grid_bits=np.stack(rows["grid_bits"]), scalars=np.stack(rows["scalars"]),
                probmap=np.stack(rows["probmap"]),
            )
            np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
        except Exception as error:  # noqa: BLE001 - recording must never cost the game
            self.last_error = repr(error)
            return None
        self.last_path = path
        self.saved.append(path)
        return path

    def _at_exit(self) -> None:
        if self.active:
            self.finish(complete=False)


def load_record(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    """(meta, arrays) of one recorded game."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        arrays = {k: data[k] for k in data.files if k != "meta"}
    return meta, arrays
