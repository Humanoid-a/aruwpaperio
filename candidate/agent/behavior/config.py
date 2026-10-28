"""Every constant of the action-state layer, in one place.

The brief asks for one configuration file. There are two, one per layer, so that each layer owns its numbers and
neither has to know the other's: this one for how the agent sees the board, moves and switches state, and
candidate/agent/q/config.py for the Q model (the scaling of its inputs, labels, exploration, the stand-in's weights).
Nothing under agent/behavior hard-codes a number that belongs here. The values are starting points to tune. Section
references are to the brief (docs/Q_Model.pdf) and docs/Q_BUILD_PLAN.md.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorConfig:
    # ---- choosing a state (chooser.py) ------------------------------------------------------------------------------
    switch_cost: float = 2.0               # h: a new state must beat the current one by this many cells
    switch_cost_return_to_expand: float = 4.0   # larger, because it lengthens the trail
    safety_threshold: int = 1              # turns of worst-case margin at or below which RETURN is forced

    # ---- the safety margin (snapshot.py, safety.py) -----------------------------------------------------------------
    margin_offset: int = 0                 # shifts every safety margin by a fixed number of turns (turn-order question)
    safety_lookahead: bool = True          # also refuse any move that would leave the margin below safety_floor (safety.py)
    safety_floor: float = 0.0              # the margin a move must leave; 0 = the enemy arrives no earlier than I close

    # ---- D1..D8, read from the world map (snapshot.py): only the cells at least this likely ----------------------
    position_threshold: float = 1e-4       # the enemy's position (D1, D3)
    trail_threshold: float = 0.5           # the enemy's trail (D2, D6)
    land_threshold: float = 0.5            # mine and the enemy's land (D3, D4, D5, D6)

    # ---- the worst case, for the hard safety rules (snapshot.py, safety.py) --------------------------------------
    known_trail_probability: float = 0.5   # a step onto an enemy trail cell kills it when the cell is this likely (or seen)
    possible_min_probability: float = 0.0  # the enemy could be on a cell whose probability exceeds this (0: any)
    distance_cap: int = 50                 # every distance is clipped to this (and 50 means "none")
    since_sighting_cap: int = 100          # turns since the enemy was last seen are counted up to this

    # ---- RETURN (return_home.py) -----------------------------------------------------------------------------------
    return_max_extra: int = 8              # candidate places to close the loop: up to this many steps beyond the nearest
    return_max_candidates: int = 64        # ... and at most this many of them
    return_margin: int = 1                 # turns ahead of the enemy a path must be to be chosen for its gain

    # ---- HUNT (hunt.py) -------------------------------------------------------------------------------------------
    hunt_min_chance: float = 0.4           # HUNT is worth it when its best move cuts the enemy's trail before it gets home
                                           # in at least this share of the whole belief (hunt.py). Measured in real-belief
                                           # games: below 0.4 the true trail could be cut on 0 of 567 frames, above it on
                                           # 23 of 26

    # ---- TRACK TARGET (track.py) ------------------------------------------------------------------------------------
    track_view_radius: int = 5             # half of the 11 x 11 view
    track_discount: float = 0.97           # a place's worth is multiplied by this per step of walking: near beats far,
                                           # but a likely place far away still beats an empty one nearby

    # ---- PATROL's exit score (exits.py; terms are each scaled to 0..1) ----------------------------------------------
    exit_room_radius: int = 3              # free-cell density is measured in a (2r+1) square just outside the exit
    exit_weight_room: float = 1.0
    exit_weight_safety: float = 1.0
    exit_weight_info: float = 0.5
    exit_weight_walk: float = 0.3          # penalty for walking far to the exit (steps / 50)
    info_age_cap: int = 100

    # ---- EXPAND's choice of exit and direction (expand.py) --------------------------------------------------------
    expand_max_walk: int = 2               # exits farther than this (steps inside my land) are not considered: EXPAND picks
                                           # a spot near me to go out from, it does not reposition (with none in reach: the
                                           # nearest exit)
    expand_border_radius: int = 3          # the border's line at an exit is fitted to my land's edge cells this close to it

    # ---- randomness for training games (style.py) -------------------------------------------------------------------
    randomness: float = 0.0                # 0: every game played exactly as written (a real match). In (0, 1]: each game
                                           # draws its own variation of the constants above and of PATROL's and EXPAND's
                                           # choices, larger at larger values; only for training games


BEHAVIOR = BehaviorConfig()
