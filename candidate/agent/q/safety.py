"""The safety rule with a look-ahead.

The brief's hard rule 2 sends the controller to RETURN when the worst-case safety margin (D7 = D1 - D4) is at or below
a threshold. That reads the margin of *now*. But a step can cost much more than one turn of margin: stepping straight
toward an approaching enemy shortens D1 by two (it closes one, and my new trail cell is one nearer to it) and
lengthens D4 by one, so the margin falls by up to three in a single move. A margin of 4 then becomes 1 and then -2
before the rule can act, and the enemy reaches my trail on the way home. (Seen in play: the hunter at the close
respawn killed the stand-in 19 times in 400 turns, always this way.)

margin_after() is the margin the *next* frame would show if I stepped onto `dest` now, assuming the worst: the
enemy's arrival times fall by one (it moves), my trail gains `dest`, and I am `home_dist[dest]` steps from home.
The strategy refuses any move other than RETURN's that would leave this below `safety_floor`, and does RETURN
instead (or PATROL if there is no trail to return along). RETURN's own steps keep the margin unchanged: each step
home shortens D4 by one and the enemy's approach shortens D1 by one.
"""

import math

from candidate.agent.geometry import bitboard
from candidate.agent.q.config import QConfig
from candidate.agent.q.snapshot import Snapshot


def margin_after(snap: Snapshot, dest: int, cfg: QConfig) -> float:
    """The worst-case safety margin at the next frame if I step onto `dest` now. Infinite when the step is onto my own
    land (it closes the loop or stays home) or onto an enemy trail cell (it kills the enemy and sends it home)."""
    if bitboard.has_cell(snap.my_land_bb, dest) or bitboard.has_cell(snap.known_trail_bb, dest):
        return math.inf
    cap = cfg.distance_cap
    nearest = min(snap.d1 if snap.exposed else cap, int(snap.enemy_arrival[dest]))
    way_home = min(int(snap.home_dist[dest]), cap)
    return float((nearest - 1) - way_home + cfg.margin_offset)
