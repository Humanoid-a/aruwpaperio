"""Choosing the state: masks, a switching cost, and two hard rules that Q cannot outvote.

In order (rules 1 and 2 win over 3 and 4):

    1  after a death, go to PATROL at the respawn
    2  if the worst-case safety margin is at or below the threshold (and there is a trail to protect), go to RETURN
    3  remove the states that are not valid right now
    4  switch to the best remaining state only if its value exceeds the current state's by the switching cost h;
       switching from RETURN back to EXPAND or EXPAND2 costs more, because it lengthens the trail

If the current state has become invalid (RETURN after the loop closed, say) the controller moves to the best valid
state at once, with no cost. A masked state is never chosen. Pure: no state is kept here.

The values come from Q (candidate/agent/q), but nothing here depends on where they come from: any five numbers in
cells of score difference, in the order of State, will do.
"""

from dataclasses import dataclass

import numpy as np

from candidate.agent.behavior.config import BEHAVIOR, BehaviorConfig
from candidate.agent.behavior.states import N_STATES, State


@dataclass(frozen=True)
class Choice:
    state: State
    forced: str | None = None      # "death" or "safety" when a hard rule decided, else None


def switching_cost(current: State, wanted: State, cfg: BehaviorConfig = BEHAVIOR) -> float:
    if current == State.RETURN and wanted in (State.EXPAND, State.EXPAND2):
        return cfg.switch_cost_return_to_expand
    return cfg.switch_cost


def choose(
    values: np.ndarray,
    valid: np.ndarray,
    current: State,
    *,
    died: bool,
    trail_len: int,
    margin: float,
    cfg: BehaviorConfig = BEHAVIOR,
) -> Choice:
    """`values` and `valid` are (6,) in the order of State. `margin` is the worst-case safety margin
    (Snapshot.safety_margin: from every cell the enemy could be on, not only the likely ones)."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != (N_STATES,) or valid.shape != (N_STATES,):
        raise ValueError("values and valid must each have one entry per state")

    if died:
        return Choice(State.PATROL, "death")
    if trail_len > 0 and margin <= cfg.safety_threshold:
        return Choice(State.RETURN, "safety")

    if not valid.any():
        return Choice(current)                                   # nothing to choose from; the caller falls back
    masked = np.where(valid, values, -np.inf)
    best = State(int(masked.argmax()))                           # ties go to the lower index
    if not valid[current]:
        return Choice(best)
    if best != current and values[best] > values[current] + switching_cost(current, best, cfg):
        return Choice(best)
    return Choice(current)
