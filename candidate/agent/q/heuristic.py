"""A hand-set stand-in for Q, so the controller can play before any network exists.

It returns five numbers in the unit Q will use (cells of score difference) from the snapshot alone. It is the
baseline every trained model is compared with, and the policy that collects the first round of training data. It
is deliberately simple:

    EXPAND  a base value that shrinks with the trail already out and grows with the safety margin
    RETURN  the gain of closing the loop now, less a penalty when the margin is thin
    HUNT    a fixed prize when I can reach the enemy's trail before it gets home
    TRACK   small; a little more when the enemy is out of view
    PATROL  small and constant

Any object with values(frame, inputs) -> (5,) can take its place; the NumPy network will.
"""

import numpy as np

from candidate.agent.q.config import Q, QConfig
from candidate.agent.q.frame import Frame
from candidate.agent.q.types import N_STATES, State


class HeuristicQ:
    def __init__(self, cfg: QConfig = Q) -> None:
        self.cfg = cfg

    def values(self, frame: Frame, inputs=None) -> np.ndarray:
        cfg, snap = self.cfg, frame.snap
        margin = min(max(snap.d7, 0.0), cfg.margin_clip)
        out = np.zeros(N_STATES, dtype=np.float64)
        out[State.EXPAND] = cfg.h_expand_base + cfg.h_expand_per_margin * margin - cfg.h_expand_per_trail * snap.trail_len
        out[State.RETURN] = snap.d5 - (cfg.h_return_unsafe_penalty if snap.d7 <= cfg.return_buffer else 0.0)
        out[State.HUNT] = cfg.h_hunt_value if snap.d8 <= -cfg.h_hunt_min_race and snap.enemy_trail_known else 0.0
        out[State.TRACK] = cfg.h_track_base + (0.0 if snap.enemy_seen else cfg.h_track_unseen_bonus)
        out[State.PATROL] = cfg.h_patrol_value
        return out
