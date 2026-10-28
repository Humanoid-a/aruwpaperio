"""Q's scalar input: 23 numbers, in the brief's order and scaling (the brief's 22, with six states instead of five).

     1 trail length / 100            12 D8 kill race, clipped +-25, / 25
     2 my count / 500                13 turns left / 2000
     3 enemy count / 500             14 enemy visible (0/1)
     4 score difference / 500        15 enemy trail known (0/1)
     5 D1 / 50                       16 turns since sighting / 100
     6 D2 / 50                       17 log(1 + cells the enemy is likely on) / log 2500
     7 D3 / 50                       18-23 current state, one-hot (six)
     8 D4 / 50
     9 D5 gain / 100
    10 D6 gain / 100
    11 D7 margin (D1 - D4), clipped +-25, / 25

D1..D8 are read from the world map, above the BehaviorConfig thresholds (behavior/snapshot.py). All of them are
counts and distances, so they do not change if the board is rotated or flipped.
"""

import math

import numpy as np

from candidate.agent.q.config import Q, QConfig
from candidate.agent.behavior.snapshot import Snapshot
from candidate.agent.behavior.states import N_STATES, State
from candidate.agent.runtime import TurnView

N_SCALARS = 17 + N_STATES

NAMES = (
    "trail_len", "my_count", "enemy_count", "score_diff", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8",
    "turns_left", "enemy_visible", "enemy_trail_known", "turns_since_sighting", "belief_spread",
) + tuple(f"state_{s.name.lower()}" for s in State)


def build_scalars(view: TurnView, snap: Snapshot, current: State, cfg: QConfig = Q) -> np.ndarray:
    obs, cap = view.obs, cfg.scale_distance
    clip = cfg.margin_clip
    out = np.zeros(N_SCALARS, dtype=np.float32)
    out[0] = snap.trail_len / cfg.scale_trail
    out[1] = obs.my_count / cfg.scale_count
    out[2] = obs.opp_count / cfg.scale_count
    out[3] = (obs.my_count - obs.opp_count) / cfg.scale_count
    out[4] = min(max(snap.d1, 0), cap) / cap
    out[5] = min(max(snap.d2, 0), cap) / cap
    out[6] = min(max(snap.d3, 0.0), cap) / cap
    out[7] = min(max(snap.d4, 0), cap) / cap
    out[8] = snap.d5 / cfg.scale_gain
    out[9] = snap.d6 / cfg.scale_gain
    out[10] = min(max(snap.d7, -clip), clip) / clip
    out[11] = min(max(snap.d8, -clip), clip) / clip
    out[12] = obs.turns_left / cfg.scale_turns
    out[13] = 1.0 if snap.enemy_seen else 0.0
    out[14] = 1.0 if snap.enemy_trail_known else 0.0
    out[15] = snap.turns_since_sighting / cfg.scale_since_sighting
    out[16] = math.log1p(int(snap.likely.sum())) / math.log(cfg.belief_spread_cells)
    out[17 + int(current)] = 1.0
    return out
