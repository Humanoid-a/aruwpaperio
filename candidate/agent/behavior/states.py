"""The six action-states. Their order is the order of Q's six outputs: Q's output i is the value of State(i)."""

from enum import IntEnum


class State(IntEnum):
    EXPAND = 0    # go out from the nearest edge of my land, and straight on
    EXPAND2 = 1   # the turn: sideways, perpendicular to the way out
    RETURN = 2    # back to my nearest land by the shortest way
    HUNT = 3      # to the nearest cell that probably holds the enemy's trail
    TRACK = 4     # TRACK TARGET: go where a look would tell me the most
    PATROL = 5    # reposition inside my land


N_STATES = len(State)
