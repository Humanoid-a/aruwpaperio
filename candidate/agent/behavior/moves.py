"""The table over the six states: which are valid right now, and the move of the one chosen.

valid_states() is the mask the chooser applies to Q's values (a masked state is never chosen); move_for() runs the
chosen state's own move. Each state's module has the same two functions, valid(frame) and move(frame), so adding or
changing a state touches its own module and the one line of the table below.
"""

import numpy as np

from candidate.agent.behavior import expand, expand2, hunt, patrol, return_home, track
from candidate.agent.behavior.frame import Frame
from candidate.agent.behavior.states import N_STATES, State
from candidate.agent.data.types import Direction

_STATES = {State.EXPAND: expand, State.EXPAND2: expand2, State.RETURN: return_home, State.HUNT: hunt,
           State.TRACK: track, State.PATROL: patrol}
assert set(_STATES) == set(State), "every state needs its module"


def valid_states(frame: Frame) -> np.ndarray:
    """(6,) boolean in the order of Q's outputs. With no move available at all, nothing is valid."""
    out = np.zeros(N_STATES, dtype=bool)
    if not frame.safe:
        return out
    for state, module in _STATES.items():
        out[state] = module.valid(frame)
    return out


def move_for(state: State, frame: Frame) -> Direction | None:
    """The chosen state's move, or None when it has none (the strategy's guard then picks a safe one)."""
    return _STATES[State(state)].move(frame)
