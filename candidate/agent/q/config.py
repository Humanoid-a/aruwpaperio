"""Every constant of the Q model, in one place.

The action-states have their own (candidate/agent/behavior/config.py; see there for why there are two). Nothing under
agent/q hard-codes a number that belongs here. The values are starting points to tune. Section references are to the
brief (docs/Q_Model.pdf) and docs/Q_BUILD_PLAN.md.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QConfig:
    # ---- labels and data collection -------------------------------------------------------------------------
    label_window: int = 100                # frames ahead that a label looks at (and further, with bootstrapping)
    label_horizons: tuple = (20, 50)       # the label's segments: frames 1-20, 21-50, 51-label_window ...
    label_weights: tuple = (1.0, 0.6, 0.3)  # ... weighted so (nearer counts more), then all scaled so that a steady gain
                                           # per frame gives the same label as unweighted (offline/dataset.py)
    exploration_rate: float = 0.03         # share of frames on which a random valid state is held ...
    exploration_hold: int = 8              # ... for up to this many frames
    belief_share: float = 0.2              # share of training games played with the real belief filter; in the others
                                           # the agent is given the true world (offline/oracle.py), much faster

    # ---- inputs (features.py, grid.py, probmap.py) ------------------------------------------------------------
    grid_crop: int = 21                    # egocentric grid side
    probmap_block: int = 5              # pooling block: 50 x 50 -> 10 x 10
    scale_distance: float = 50.0           # D1..D4 are clipped to this and divided by it
    margin_clip: int = 25                  # D7 and D8 are clipped to +-this
    scale_trail: float = 100.0
    scale_count: float = 500.0
    scale_gain: float = 100.0
    scale_turns: float = 2000.0
    scale_since_sighting: float = 100.0    # turns since the enemy was last seen / this
    belief_spread_cells: int = 2500        # log(1 + cells) / log(this)

    # ---- the stand-in for Q (heuristic.py), until a network exists --------------------------------------------
    h_expand_base: float = 5.0             # EXPAND and EXPAND2 ...
    h_expand_per_margin: float = 0.15      # ... plus this per turn of safety margin (up to margin_clip)
    h_expand_per_trail: float = 0.3        # ... minus this per cell of trail already out
    h_leg_bonus: float = 3.0               # ... plus this for the one whose leg it is: going out for the first
    h_out_leg: int = 5                     #     h_out_leg cells (EXPAND), then turning (EXPAND2), so that the loop
                                           #     encloses something; RETURN wins once its gain outgrows them
    h_return_unsafe_penalty: float = 10.0  # RETURN is worth this less ...
    h_return_safe_margin: int = 2          # ... while the margin is at most this
    h_hunt_value: float = 12.0
    h_hunt_min_race: int = 2               # D3 - own distance must be at least this
    h_track_base: float = 0.5
    h_track_unseen_bonus: float = 1.5
    h_patrol_value: float = 1.0


Q = QConfig()
