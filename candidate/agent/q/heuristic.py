"""A hand-set stand-in for Q, so the controller can play before any network exists.

It returns six numbers in the unit Q will use (cells of score difference) from the snapshot and the expedition's
hints. It is the baseline every trained model is compared with, and the policy that collects the first round of
training data. It is deliberately simple, and plays a rectangle: out, across, home.

    EXPAND   a base value that grows with the safety margin and shrinks with the trail already out; plus a bonus
             at home and for the first h_out_leg cells of an expedition (the out leg)
    EXPAND2  the same base; plus the bonus once the out leg is done (or the turn has been taken): the across leg
    RETURN   the gain of closing the loop now, less a penalty when the margin is thin; it grows with the trail, so it
             wins once the loop is worth closing
    HUNT     a fixed prize when I can reach the enemy's trail before it gets home
    TRACK    small; a little more when the enemy is out of view
    PATROL   small and constant

Any object with values(frame, inputs) -> (6,) can take its place; the trained network (network.NetworkQ) does.
"""

import numpy as np

from candidate.agent.behavior.frame import Frame
from candidate.agent.behavior.states import N_STATES, State
from candidate.agent.q.config import Q, QConfig


class HeuristicQ:
    def __init__(self, cfg: QConfig = Q) -> None:
        self.cfg = cfg

    def values(self, frame: Frame, inputs=None) -> np.ndarray:
        cfg, snap = self.cfg, frame.snap
        margin = min(max(snap.safety_margin, 0.0), cfg.margin_clip)
        going = cfg.h_expand_base + cfg.h_expand_per_margin * margin - cfg.h_expand_per_trail * snap.trail_len
        out_leg = not snap.exposed or (frame.hints.across is None and snap.trail_len < cfg.h_out_leg)
        out = np.zeros(N_STATES, dtype=np.float64)
        out[State.EXPAND] = going + (cfg.h_leg_bonus if out_leg else 0.0)
        out[State.EXPAND2] = going + (0.0 if out_leg else cfg.h_leg_bonus)
        out[State.RETURN] = snap.d5 - (cfg.h_return_unsafe_penalty if snap.safety_margin <= cfg.h_return_safe_margin else 0.0)
        out[State.HUNT] = cfg.h_hunt_value if snap.d8 <= -cfg.h_hunt_min_race and snap.enemy_trail_known else 0.0
        out[State.TRACK] = cfg.h_track_base + (0.0 if snap.enemy_seen else cfg.h_track_unseen_bonus)
        out[State.PATROL] = cfg.h_patrol_value
        return out
