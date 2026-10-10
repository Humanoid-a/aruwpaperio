"""The opponent motion model K.

K(S_t, S_t+1) says how attractive one move is to the opponent, given where it
is and what the move would do. The belief resolves every legal move with the
game rules first (so it knows if the move kills the opponent, closes a loop,
or cuts my trail) and then asks K to score the outcomes; a softmax over the
legal moves, mixed with a small floor, turns the scores into probabilities:

    P(a | s) = (1 - eps) * exp(beta * K[a]) / sum_legal exp(beta * K[a'])  +  eps / |legal|

The floor guarantees that no legal move has probability zero, which the
elimination step depends on: evidence may only rule a hypothesis out because
the rules or the observation contradict it, never because K thought it unlikely.

Subclass MotionModel to plug in something else. Two are provided:

  * FeatureMotionModel (the default): a weighted sum of seven labelled features
    of each move's outcome (F1 .. F7, see below), with the weights in MotionWeights.
  * ConstantMotionModel: every move scores the same, so every legal move is
    equally likely. A baseline; it prints a description of each batch it is given.
"""

from abc import ABC, abstractmethod
import dataclasses
from dataclasses import dataclass, fields

import numpy as np

from candidate.agent.data.observation import Observation
from candidate.agent.geometry.distance import taxicab_distance


@dataclass(frozen=True, eq=False)
class SuccessorBatch:
    """Everything K may look at: n hypotheses and, for each, the outcome of
    each of the four moves (columns north, east, south, west). All arrays are
    read-only views; entries for illegal moves are meaningless."""

    n: int
    # the parents
    pos: np.ndarray          # (n,) flat cell of the opponent
    trail_bb: np.ndarray     # (n, 50) its trail
    trail_len: np.ndarray    # (n,)
    last_dir: np.ndarray     # (n,) -1 if not exposed
    opp_count: np.ndarray    # (n,) its territory size
    weight: np.ndarray       # (n,) the parent's probability mass
    # the outcome of each move
    dest: np.ndarray         # (n, 4) cell stepped onto (may be off the board where not legal)
    legal: np.ndarray        # (n, 4) on the board and not onto my agent
    self_trail: np.ndarray   # (n, 4) steps on its own trail: it dies and respawns
    kills_me: np.ndarray     # (n, 4) steps on my trail: I die
    enters_own: np.ndarray   # (n, 4) steps onto its own territory (and does not die)
    captures: np.ndarray     # (n, 4) closes a loop (enters_own with a trail)
    capture_gain: np.ndarray  # (n, 4) cells the capture would add (0 where there is none)
    # the situation
    my_pos: int              # where my agent is while the opponent moves
    my_trail_bb: np.ndarray  # (50,) my trail at that moment
    opp_land_bb: np.ndarray  # (50,) the cells memory believes the opponent owns (shared by all hypotheses)
    obs: Observation         # this turn's observation

    def select(self, rows: np.ndarray) -> "SuccessorBatch":
        """The batch for only the hypotheses at `rows`. Used to score just the parents that still have a
        child after the evidence has been applied; the situation (my agent, the maps) is shared."""
        rows = np.asarray(rows)
        return dataclasses.replace(
            self, n=int(rows.size), **{name: getattr(self, name)[rows] for name in _PER_HYPOTHESIS}
        )

    def describe(self) -> str:
        """A short description of everything in the batch: each field's name, type, shape and
        dtype, and what it holds. No values, so it is safe to print every turn."""
        lines = [f"SuccessorBatch: {self.n} hypotheses x 4 moves (columns N, E, S, W)"]
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                kind = f"ndarray{value.shape} {value.dtype}"
            elif isinstance(value, Observation):
                kind = f"Observation (turn {value.turn}, window {value.window_territory.shape[0]}x{value.window_territory.shape[1]})"
            else:
                kind = type(value).__name__
            lines.append(f"  {f.name:13s} {kind:28s} {_FIELD_NOTES[f.name]}")
        return "\n".join(lines)


# the fields that have one entry (or four) per hypothesis; the rest describe the shared situation
_PER_HYPOTHESIS = (
    "pos", "trail_bb", "trail_len", "last_dir", "opp_count", "weight",
    "dest", "legal", "self_trail", "kills_me", "enters_own", "captures", "capture_gain",
)

_FIELD_NOTES = {
    "n": "number of hypotheses",
    "pos": "opponent's cell, per hypothesis (flat index y*50+x)",
    "trail_bb": "its trail as 50 row bitboards",
    "trail_len": "cells in its trail",
    "last_dir": "direction of its last move, -1 if not exposed",
    "opp_count": "its territory size",
    "weight": "the hypothesis' probability mass",
    "dest": "cell each move steps onto",
    "legal": "move is on the board and not onto my agent",
    "self_trail": "move steps on its own trail (it dies)",
    "kills_me": "move steps on my trail (I die)",
    "enters_own": "move steps onto its own land",
    "captures": "move closes a loop",
    "capture_gain": "cells that capture would add (0 if none)",
    "my_pos": "my agent's cell during its move",
    "my_trail_bb": "my trail as 50 row bitboards",
    "opp_land_bb": "opponent land per memory, as 50 row bitboards (shared)",
    "obs": "this turn's observation",
}


class MotionModel(ABC):
    @abstractmethod
    def score(self, batch: SuccessorBatch) -> np.ndarray:
        """K for every (hypothesis, move): a float array of shape (batch.n, 4),
        higher meaning the opponent is more likely to choose that move. Only
        the scores of legal moves are used. Must be deterministic for a given
        batch and must not modify it."""


class ConstantMotionModel(MotionModel):
    """The placeholder: every move gets the same score, so after the softmax
    every legal move is equally likely. Replace with a model that looks at the
    consequence flags and the opponent's state."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def score(self, batch: SuccessorBatch) -> np.ndarray:
        print(batch.describe())
        return np.full((batch.n, 4), self.value, dtype=np.float64)


# =============================================================================
# The feature-based motion model
#
#     K(s, a) = sum over features f of  weight[f] * feature[f](s, a)
#
# Seven features, each one a labelled section below. Every feature is an (n, 4)
# array (n hypotheses x moves N, E, S, W). Positive pushes the opponent towards
# that move, negative away from it. The weights are the only numbers to tune.
# =============================================================================


@dataclass(frozen=True)
class MotionWeights:
    """The weight of each feature, and the two constants the features use.
    A weight of 0 switches a feature off.

    The defaults were fitted by hand to the one-step log-likelihood of the real
    opponent's move (each hypothesis being the opponent's true state), against the
    enemy, basic and random state machines. Measured over several hundred moves each:
    uniform -1.35 / -1.38 / -1.38, these defaults -0.87 / -1.23 / -1.42."""

    # F1 -- stepping on its own trail kills it. Large but finite: K never rules a move out.
    self_trail: float = -6.0
    # F2 -- stepping on my trail kills me. An opponent that can see a kill takes it.
    kills_my_trail: float = 2.0
    # F3 -- closing a loop. Multiplies log(1 + cells gained), so a big capture pulls harder
    #       than a small one without dwarfing everything else.
    capture: float = 0.25
    # F4 -- going home. Per step closer to its own land, at full trail pressure.
    home_pull: float = 1.0
    home_pressure_half: float = 6.0   # trail length at which the pressure is half of full
    # F5 -- chasing my trail. Per step closer to it, only while it is inside the opponent's view.
    chase: float = 1.0
    view_radius: int = 5              # the opponent sees 11 x 11, i.e. 5 cells each way
    # F6 -- momentum. Keeping the direction of its last move (while it has a trail).
    momentum: float = 1.2
    # F7 -- exploring. Leaving its own land when it is standing on it.
    explore: float = 0.4

    def __post_init__(self) -> None:
        if self.home_pressure_half <= 0:
            raise ValueError("home_pressure_half must be positive")
        if self.view_radius < 1:
            raise ValueError("view_radius must be at least 1")


FEATURE_NAMES = ("self_trail", "kills_my_trail", "capture", "home_pull", "chase", "momentum", "explore")


class FeatureMotionModel(MotionModel):
    """K as a weighted sum of hand-written features of each move's outcome.

    `features(batch)` returns every feature by name (useful for looking at why a
    move scored what it did); `score(batch)` is the weighted sum of them.

    Two distance maps are built once per call and shared by all hypotheses. They
    are taxicab distances (obstacles ignored), which for the opponent is a safe
    lower bound on its travel time, since only its own trail blocks it."""

    def __init__(self, weights: MotionWeights | None = None) -> None:
        self.weights = weights if weights is not None else MotionWeights()

    def score(self, batch: SuccessorBatch) -> np.ndarray:
        w = self.weights
        parts = self.features(batch)
        total = np.zeros((batch.n, 4), dtype=np.float64)
        for name in FEATURE_NAMES:
            total += getattr(w, name) * parts[name]
        return total

    def features(self, batch: SuccessorBatch) -> dict[str, np.ndarray]:
        w = self.weights
        pos = batch.pos.astype(np.int64)                   # (n,) where each opponent stands
        dest = batch.dest.astype(np.int64)                 # (n, 4) where each move lands (= pos if illegal)
        has_trail = (batch.trail_len > 0)[:, None]         # (n, 1) standing outside its own land

        # ---- F1: steps on its own trail ------------------------------------
        # It dies and goes back to its respawn. 1 where the move does that.
        f_self_trail = batch.self_trail.astype(np.float64)

        # ---- F2: steps on my trail -----------------------------------------
        # I die. 1 where the move does that.
        f_kills_my_trail = batch.kills_me.astype(np.float64)

        # ---- F3: captures ----------------------------------------------------
        # log(1 + cells the loop would add); 0 where the move closes no loop.
        f_capture = np.log1p(batch.capture_gain.astype(np.float64))

        # ---- F4: going home, stronger the longer the trail -------------------
        # change = +1 if the move gets closer to its own land, -1 if further, 0 if level.
        # pressure = trail / (trail + half), from 0 (no trail) towards 1 (a long trail).
        home = taxicab_distance(batch.opp_land_bb).reshape(-1).astype(np.int64)
        change = np.clip(home[pos][:, None] - home[dest], -1, 1)
        pressure = batch.trail_len / (batch.trail_len + w.home_pressure_half)
        f_home_pull = change * pressure[:, None]

        # ---- F5: chasing my trail, only if it is in view ---------------------
        # The same +1 / -1 / 0 change, for the distance to my trail. In view means my
        # trail is within view_radius steps of where it stands now.
        mine = taxicab_distance(batch.my_trail_bb).reshape(-1).astype(np.int64)
        in_view = (mine[pos] <= w.view_radius)[:, None]
        closer = np.clip(mine[pos][:, None] - mine[dest], -1, 1)
        f_chase = closer * in_view

        # ---- F6: momentum ------------------------------------------------------
        # 1 for the move that repeats its last direction. last_dir is -1 when it is not
        # exposed (no trail), so it only applies mid-expedition.
        f_momentum = (batch.last_dir[:, None] == np.arange(4)[None, :]).astype(np.float64)

        # ---- F7: exploring ---------------------------------------------------
        # 1 for a move that leaves its land, when it has no trail (so it stands on its land).
        f_explore = (~has_trail & ~batch.enters_own).astype(np.float64)

        return {
            "self_trail": f_self_trail, "kills_my_trail": f_kills_my_trail, "capture": f_capture,
            "home_pull": f_home_pull, "chase": f_chase, "momentum": f_momentum, "explore": f_explore,
        }


def move_probabilities(scores: np.ndarray, legal: np.ndarray, beta: float, epsilon: float) -> np.ndarray:
    """Turn K scores into move probabilities: zero for illegal moves, a softmax
    mixed with the floor `epsilon` over the legal ones. Each row sums to 1, or to
    0 if the hypothesis has no legal move at all."""
    scores = np.asarray(scores, dtype=np.float64)
    masked = np.where(legal, beta * scores, -np.inf)
    top = np.max(masked, axis=1, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    exp = np.where(legal, np.exp(masked - top), 0.0)
    total = exp.sum(axis=1, keepdims=True)
    n_legal = legal.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        soft = np.where(total > 0, exp / total, 0.0)
        floor = np.where(n_legal > 0, epsilon / n_legal, 0.0)
    return np.where(legal, (1.0 - epsilon) * soft + floor, 0.0) * (n_legal > 0)
