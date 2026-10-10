"""Every constant of the Q model and the action-states, in one place (as the brief requires).

The values are starting points to tune. Nothing else under agent/q hard-codes a number that belongs here.
Section references are to the brief (docs/Q_Model.pdf) and docs/Q_BUILD_PLAN.md.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QConfig:
    # ---- labels and data collection -------------------------------------------------------------------------
    label_window: int = 50                 # frames ahead that a label looks at
    exploration_rate: float = 0.03         # share of frames on which a random valid state is held ...
    exploration_hold: int = 8              # ... for up to this many frames

    # ---- the chooser ------------------------------------------------------------------------------------------
    switch_cost: float = 2.0               # h: a new state must beat the current one by this many cells
    switch_cost_return_to_expand: float = 4.0   # larger, because it lengthens the trail
    safety_threshold: int = 1              # turns of worst-case margin at or below which RETURN is forced
    margin_offset: int = 0                 # shifts every safety margin by a fixed number of turns (turn-order question)
    safety_lookahead: bool = True          # also refuse any move that would leave the margin below safety_floor (safety.py)
    safety_floor: float = 0.0              # the margin a move must leave; 0 = the enemy arrives no earlier than I close

    # ---- RETURN -----------------------------------------------------------------------------------------------------
    return_buffer: int = 2                 # turns of margin a path needs to be preferred for its gain
    return_max_sideways: int = 8           # W

    # ---- TRACK TARGET ---------------------------------------------------------------------------------------------
    track_min_distance: int = 3
    track_max_distance: int = 5
    track_view_radius: int = 5             # half of the 11 x 11 view
    track_reach_cap: int = 30              # farthest cell TRACK TARGET considers heading for
    track_distance_cost: float = 0.01      # belief mass given up per step of walking, so near beats far when equal
    track_outside_margin: float = 0.05     # a cell outside my territory must cover this much more belief than the best inside one

    # ---- what counts as known / possible -----------------------------------------------------------------------
    known_trail_probability: float = 0.5   # an enemy trail cell is known when this likely (or visible)
    possible_min_probability: float = 0.0  # a cell is in the possibility mask when its probability exceeds this

    # ---- inputs ----------------------------------------------------------------------------------------------------
    grid_crop: int = 21                    # egocentric grid side
    confidence_block: int = 5              # pooling block: 50 x 50 -> 10 x 10
    since_sighting_cap: int = 100
    distance_cap: int = 50                 # every distance is clipped to this (and 50 means "none")
    margin_clip: int = 25                  # D7 and D8 are clipped to +-this
    scale_trail: float = 100.0
    scale_count: float = 500.0
    scale_gain: float = 100.0
    scale_turns: float = 2000.0
    belief_spread_cells: int = 2500        # log(1 + cells) / log(this)

    # ---- PATROL's exit score (terms are each scaled to 0..1) ---------------------------------------------------
    exit_room_radius: int = 3              # free-cell density is measured in a (2r+1) square just outside the exit
    exit_weight_room: float = 1.0
    exit_weight_safety: float = 1.0
    exit_weight_info: float = 0.5
    exit_weight_walk: float = 0.3          # penalty for walking far to the exit (steps / 50)
    info_age_cap: int = 100

    # ---- the stand-in for Q (heuristic.py), until a network exists --------------------------------------------
    h_expand_base: float = 5.0
    h_expand_per_margin: float = 0.15      # per turn of safety margin (up to margin_clip)
    h_expand_per_trail: float = 0.3        # lost per cell of trail already out
    h_return_unsafe_penalty: float = 10.0
    h_hunt_value: float = 12.0
    h_hunt_min_race: int = 2               # D3 - own distance must be at least this
    h_track_base: float = 0.5
    h_track_unseen_bonus: float = 1.5
    h_patrol_value: float = 1.0


Q = QConfig()
