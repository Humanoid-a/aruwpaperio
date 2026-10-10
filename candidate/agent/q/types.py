"""The five action-states, in the order of Q's five outputs."""

from enum import IntEnum


class State(IntEnum):
    EXPAND = 0
    RETURN = 1
    HUNT = 2
    TRACK = 3     # TRACK TARGET
    PATROL = 4


N_STATES = len(State)
