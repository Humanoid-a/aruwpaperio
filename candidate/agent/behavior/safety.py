"""The safety rule with a look-ahead.

The brief's hard rule 2 sends the controller to RETURN when the worst-case safety margin (D7 = D1 - D4) is at or below
a threshold. That reads the margin of *now*. But a step can cost much more than one turn of margin: stepping straight
toward an approaching enemy shortens D1 by two (it closes one, and my new trail cell is one nearer to it) and
lengthens D4 by one, so the margin falls by up to three in a single move. A margin of 4 then becomes 1 and then -2
before the rule can act, and the enemy reaches my trail on the way home. (Seen in play: the hunter at the close
respawn killed the stand-in 19 times in 400 turns, always this way.)

margin_after() is the margin the *next* frame would show if I stepped onto `dest` now, assuming the worst: the
enemy's arrival times fall by one (it moves), my trail gains `dest`, and I am `home_dist[dest]` steps from home.
The cells of that way home, the land cell it enters by included, count like trail cells: if the enemy can stand on
one before I pass, it blocks me and my way home grows. (Seen in play with the brief's margin alone: the hunter stepped
onto the one land cell my way home went through, the way home grew by two, and it reached my trail next turn.)
The strategy refuses any move other than RETURN's that would leave this below `safety_floor`, and does RETURN
instead (or PATROL if there is no trail to return along). RETURN's own steps keep the margin unchanged: each step
home shortens D4 by one and the enemy's approach shortens D1 by one.

A step onto the enemy's trail kills it, but it does not send it far away: it reappears at once on its respawn cell
and moves next. Where that cell is next to my trail, the kill is fatal for me the turn after (seen in play: hunting
next to the enemy's land, my step home cut its trail, it reappeared beside my trail and stepped on it). So a kill is
scored with the enemy at its respawn, and the strategy checks every move that would kill, RETURN's included.
"""

import math

from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import manhattan
from candidate.agent.behavior import paths
from candidate.agent.behavior.config import BehaviorConfig
from candidate.agent.behavior.snapshot import Snapshot


def kills(snap: Snapshot, dest: int) -> bool:
    """Would stepping onto `dest` cut the enemy's trail (as far as I know it)?"""
    return bitboard.has_cell(snap.known_trail_bb, dest)


def margin_after(snap: Snapshot, dest: int, cfg: BehaviorConfig) -> float:
    """The worst-case safety margin at the next frame if I step onto `dest` now. Infinite when the step is onto my own
    land (it closes the loop or stays home). A step that kills the enemy is scored with the enemy at its respawn."""
    if bitboard.has_cell(snap.my_land_bb, dest):
        return math.inf
    cap = cfg.distance_cap
    way_home = int(snap.home_dist[dest])
    if way_home >= cap:
        return float(-cap)                                  # no way home from there
    path = paths.path_from_map(snap.home_dist, dest) or []
    cells = [dest] + list(path)
    if kills(snap, dest):
        respawn = snap.enemy_alt_respawn if dest == snap.enemy_respawn else snap.enemy_respawn
        exposed = bitboard.to_cells(snap.my_trail_bb) + cells
        nearest = min(manhattan(respawn, c) for c in exposed)
    else:
        nearest = min(snap.safety_d1 if snap.exposed else cap, int(snap.enemy_arrival[cells].min()))
    return float((nearest - 1) - way_home + cfg.margin_offset)
