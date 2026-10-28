"""The six states' own logic and their validity, on small hand-built boards.

The default board: I start on the 3x3 around (25, 18) (rows 24-26, cols 17-19), the enemy on the 3x3 around (25, 31).
"""

import numpy as np
import pytest

from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import direction_to, flat, step
from candidate.agent.behavior import expand, expand2, hunt, moves, patrol, return_home, track
from candidate.agent.behavior.config import BehaviorConfig
from candidate.agent.behavior.frame import Frame, Hints
from candidate.agent.behavior.snapshot import build_snapshot
from candidate.agent.behavior.states import State
from candidate.agent.rules.match import MatchRules
from candidate.tests.q_support import CLOSE, belief_of, scene, world_with

N, E, S, W = Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST
BIG_LAND = [(y, x) for y in range(20, 31) for x in range(14, 23)]          # rows 20-30, cols 14-22


def frame_of(view, hints=None, cfg=None):
    snap = build_snapshot(view) if cfg is None else build_snapshot(view, cfg)
    return Frame(view, snap, hints, cfg) if cfg is not None else Frame(view, snap, hints)


def big_world(me=(25, 18), **kw):
    return world_with(CLOSE, me=me, my_land=BIG_LAND, **kw)


def in_land(frame, direction):
    return bitboard.has_cell(frame.snap.my_land_bb, step(frame.pos, direction))


# ---- validity -------------------------------------------------------------------------------------------------------


def test_with_no_trail_on_my_land_return_is_invalid_and_patrol_and_track_are_valid():
    mask = moves.valid_states(frame_of(scene(world_with(CLOSE), CLOSE)))
    assert not mask[State.RETURN] and not mask[State.HUNT]
    assert mask[State.PATROL] and mask[State.TRACK]


def test_with_a_trail_return_is_valid_and_patrol_never_is():
    mask = moves.valid_states(frame_of(scene(world_with(CLOSE, my_trail=[(25, 20), (25, 21)]), CLOSE)))
    assert mask[State.RETURN] and not mask[State.PATROL] and mask[State.TRACK]


def test_hunt_needs_an_enemy_trail_i_can_cut():
    assert not moves.valid_states(frame_of(scene(world_with(CLOSE), CLOSE)))[State.HUNT]
    out = world_with(CLOSE, opp_trail=north_trip(), opp_land=ENEMY_LAND)                   # see the HUNT section
    assert moves.valid_states(frame_of(scene(out, CLOSE)))[State.HUNT]


def test_hunt_is_invalid_when_the_only_known_trail_cell_is_under_the_enemy():
    world = world_with(CLOSE, opp_trail=[(25, 22)])                        # a one-cell trail: the enemy stands on it
    mask = moves.valid_states(frame_of(scene(world, CLOSE)))
    assert not mask[State.HUNT]


def test_nothing_is_valid_when_no_move_is_available():
    view = scene(world_with(CLOSE), CLOSE)
    view.__dict__["safe_moves"] = ()
    assert not moves.valid_states(frame_of(view)).any()


# ---- PATROL -------------------------------------------------------------------------------------------------------------


def walk(world_at, start, turns, state):
    """Play `state` for some frames on a board where I only move and nothing else changes; returns my cells."""
    pos, seen = start, []
    for turn in range(turns):
        view = scene(world_at(pos), CLOSE, turn=turn + 40)
        frame = frame_of(view)
        direction = moves.move_for(state, frame)
        assert direction is not None
        assert direction in frame.safe
        y, x = divmod(step(frame.pos, direction), 50)
        pos = (y, x)
        seen.append(pos)
    return seen


def test_patrol_never_leaves_my_territory_and_ends_up_pacing_at_its_exit():
    land = set(BIG_LAND)
    seen = walk(lambda pos: big_world(me=pos), (22, 20), 60, State.PATROL)
    assert all(p in land for p in seen)
    tail = seen[-8:]
    assert len(set(tail)) == 2                                            # it paces between two cells


def test_patrol_goes_to_the_exit_away_from_the_enemy():
    belief = belief_of([dict(pos=flat(25, 10), weight=1.0)])              # certainly west of my land, out of view
    frame = frame_of(scene(big_world(), CLOSE, belief))
    assert frame.exits.best % 50 == 22                                    # the east edge: 12 away, not 4


def test_the_room_term_prefers_an_edge_with_free_space_beyond_it():
    cfg = BehaviorConfig(exit_weight_room=1.0, exit_weight_safety=0.0, exit_weight_info=0.0, exit_weight_walk=0.0)
    world = big_world(opp_land=[(y, x) for y in range(15, 36) for x in range(6, 14)])      # a wall of enemy land on the west
    frame = frame_of(scene(world, CLOSE), cfg=cfg)
    assert frame.exits.best % 50 != 14


def test_the_information_term_prefers_an_edge_not_looked_at_for_a_long_time():
    cfg = BehaviorConfig(exit_weight_room=0.0, exit_weight_safety=0.0, exit_weight_info=1.0, exit_weight_walk=0.0)
    view = scene(big_world(), CLOSE, turn=100)
    view.memory.last_seen[:, 19:] = 100                                   # the east was seen just now
    frame = frame_of(view, cfg=cfg)
    assert frame.exits.best % 50 == 14


def test_the_walk_term_prefers_a_near_exit_when_the_rest_is_equal():
    cfg = BehaviorConfig(exit_weight_room=0.0, exit_weight_safety=0.0, exit_weight_info=0.0, exit_weight_walk=1.0)
    frame = frame_of(scene(big_world(me=(25, 21)), CLOSE), cfg=cfg)
    assert frame.exits.best % 50 == 22 and abs(frame.exits.best // 50 - 25) <= 1


def test_patrol_off_my_land_without_a_trail_walks_home():
    world = world_with(CLOSE, me=(25, 23))                                # standing outside my land, no trail
    frame = frame_of(scene(world, CLOSE))
    assert not frame.snap.on_land
    assert patrol.move(frame) == W


# ---- EXPAND ---------------------------------------------------------------------------------------------------------------


def test_expand_inside_walks_to_a_near_exit_and_steps_out_on_the_chosen_side():
    frame = frame_of(scene(big_world(me=(22, 16)), CLOSE))
    cell, side = expand.plan(frame)
    assert np.isfinite(frame.exits.score.reshape(-1)[cell]) and frame.exits.walk[cell] <= frame.cfg.expand_max_walk
    y, x = divmod(cell, 50)
    from candidate.agent.geometry.grid import DX, DY
    assert not frame.snap.my_land_bb[y + DY[side]] >> (x + DX[side]) & 1             # the side leads out of my land
    direction = expand.move(frame)
    assert direction is not None and in_land(frame, direction)                      # it walks inside my land to it
    from candidate.agent.behavior import paths
    to_exit = paths.distance_map(bitboard.from_cells([cell]), frame.snap.my_land_bb & frame.snap.passable_bb)
    assert to_exit[step(frame.pos, direction)] == to_exit[frame.pos] - 1                 # one step closer to it


def test_expand_from_an_exit_steps_out_of_my_land():
    frame = frame_of(scene(big_world(me=(25, 14)), CLOSE))                 # on the west edge of my land
    direction = expand.move(frame)
    assert direction is not None and expand.valid(frame)
    cell, _ = expand.plan(frame)
    assert not in_land(frame, direction) or cell != frame.pos


def test_the_angle_is_wide_straight_out_of_a_border_and_narrow_along_it():
    land = np.zeros((50, 50), dtype=bool)
    land[20:31, 10:40] = True                                              # a wide block: its top edge is a row
    edge = expand.edge_mask(land)
    line = expand.border_line(edge, 20 * 50 + 25, 3)                       # the middle of the top edge
    assert line[1] > 0.9                                                    # a clear line ...
    assert abs(abs(line[0][1]) - 1.0) < 1e-6                                # ... running along x
    assert expand.angle(N, line) == pytest.approx(1.0, abs=1e-6)            # out of it: the widest angle
    assert expand.angle(E, line) < 0.1                                      # along it: almost none
    corner = expand.border_line(edge, 20 * 50 + 10, 3)                      # the top-left corner: a symmetric L
    at_corner = [expand.angle(d, corner) for d in (N, E, S, W)]
    assert max(at_corner) - min(at_corner) < 1e-6 and 0.4 < at_corner[0] < 1.0     # no side runs along it


def test_expand_leaves_straight_out_of_the_border_not_along_it():
    # me on my land's top edge, next to a tongue of land along row 19 to the east: going east from the corner at
    # (20, 22) would run along my own land; going north from the top edge leaves it behind
    tongue = [(19, x) for x in range(23, 35)]
    world = world_with(CLOSE, me=(20, 18), my_land=BIG_LAND + tongue)
    frame = frame_of(scene(world, CLOSE))
    cell, side = expand.plan(frame)
    edge = expand.edge_mask(bitboard.to_mask(frame.snap.my_land_bb))
    assert expand.angle(side, expand.border_line(edge, cell, frame.cfg.expand_border_radius)) > 0.9


def test_with_no_exit_within_reach_expand_heads_for_the_nearest_one():
    land = [(y, x) for y in range(10, 41) for x in range(5, 36)]             # a big block: its edge is 10+ steps away
    frame = frame_of(scene(world_with(CLOSE, me=(25, 20), my_land=land), CLOSE))
    assert expand.plan(frame) is None and expand.chosen_exit(frame) == expand.nearest_exit(frame) is not None
    direction = expand.move(frame)
    assert direction is not None and in_land(frame, direction) and expand.valid(frame)


def test_expand_outside_goes_straight_on_in_the_direction_it_set_out_in():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21)])
    frame = frame_of(scene(world, CLOSE))
    assert frame.heading() == E and expand.move(frame) == E
    hinted = frame_of(scene(world_with(CLOSE, my_trail=[(24, 20), (24, 21)]), CLOSE), Hints(outward=N))
    assert expand.move(hinted) == N                                    # remembers the expedition's direction


def test_expand_never_turns_back_onto_its_own_trail():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21)])
    frame = frame_of(scene(world, CLOSE), Hints(outward=W))               # a stale hint pointing back along the trail
    assert not expand.valid(frame) and expand.move(frame) is None


def test_expand_is_invalid_when_the_next_outward_cell_is_off_the_board_or_my_land():
    rules = MatchRules.from_respawns(flat(25, 46), flat(25, 4))
    world = world_with(rules, my_trail=[(25, 48), (25, 49)])
    frame = frame_of(scene(world, rules))
    assert frame.heading() == E and not expand.valid(frame)
    land_ahead = world_with(CLOSE, my_trail=[(25, 20), (25, 21)], my_land=[(25, 22)])
    assert not expand.valid(frame_of(scene(land_ahead, CLOSE)))          # going on would close the loop: not EXPAND's job


# ---- EXPAND2 ---------------------------------------------------------------------------------------------------------------


def test_expand2_turns_sideways_to_the_side_with_more_room():
    trail = [(25, 20), (25, 21), (25, 22)]                                # out to the east, head at (25, 22)
    north_blocked = world_with(CLOSE, my_trail=trail, my_land=[(23, 22)])
    frame = frame_of(scene(north_blocked, CLOSE))
    assert frame.heading() == E and expand2.side(frame) == S and expand2.move(frame) == S
    south_blocked = world_with(CLOSE, my_trail=trail, my_land=[(27, 22)])
    assert expand2.move(frame_of(scene(south_blocked, CLOSE))) == N


def test_expand2_remembers_its_side_until_the_trail_is_home():
    trail = [(25, 20), (25, 21), (25, 22)]
    hints = Hints()
    frame = frame_of(scene(world_with(CLOSE, my_trail=trail, my_land=[(23, 22)]), CLOSE), hints)
    assert expand2.move(frame) == S and hints.across == S
    again = frame_of(scene(world_with(CLOSE, my_trail=trail, my_land=[(27, 22)]), CLOSE), hints)
    assert expand2.move(again) == S                                       # the side is kept, though north has more room now


def test_expand2_needs_a_trail_and_a_free_side():
    assert not expand2.valid(frame_of(scene(world_with(CLOSE), CLOSE)))
    trail = [(25, 20), (25, 21), (25, 22), (26, 22), (26, 23), (25, 23)]  # the head is (25, 23); (26, 23) is my trail
    frame = frame_of(scene(world_with(CLOSE, my_trail=trail), CLOSE), Hints(outward=E, across=S))
    assert not expand2.valid(frame)                                       # its side (south) is its own trail


# ---- RETURN -------------------------------------------------------------------------------------------------------------------


def expedition(enemy_at, trail=((25, 20), (25, 21), (25, 22), (25, 23), (25, 24)), **kw):
    belief = belief_of([dict(pos=flat(*enemy_at), weight=1.0)])
    return frame_of(scene(world_with(CLOSE, my_trail=list(trail)), CLOSE, belief), **kw)


def test_every_candidate_path_gets_home_without_my_trail_and_enters_my_land_only_at_its_end():
    frame = expedition((25, 45))
    options = return_home.candidates(frame)
    assert len(options) > 3
    lengths = [len(path) for path in options]
    assert lengths == sorted(lengths) and lengths[0] == frame.snap.d4 and max(lengths) <= lengths[0] + 8
    for path in options:
        assert bitboard.has_cell(frame.snap.my_land_bb, path[-1])
        assert not any(bitboard.has_cell(frame.snap.my_land_bb, c) for c in path[:-1])
        assert not any(bitboard.has_cell(frame.snap.my_trail_bb, c) for c in path)
        steps = [frame.pos] + path
        assert all(direction_to(a, b) is not None for a, b in zip(steps, steps[1:]))
    assert len(return_home.candidates(Frame(frame.view, frame.snap, frame.hints, BehaviorConfig(return_max_candidates=2)))) == 2


def test_the_scores_are_what_the_definition_says():
    frame = expedition((25, 45))
    direct = [path for path in return_home.candidates(frame) if len(path) == 6][:1]
    gain, steps, margin = return_home.score_paths(frame, direct)
    assert steps[0] == 6                                                  # up, then five west to my land
    assert gain[0] == 10                                                  # five trail cells and five new ones
    assert margin[0] == 21 - 6                                            # the enemy needs 21 to reach (25, 24)


def test_with_the_enemy_far_away_return_takes_the_path_that_gains_the_most():
    frame = expedition((25, 45))
    path = return_home.choose_path(frame)
    options = return_home.candidates(frame)
    gains, _, margins = return_home.score_paths(frame, options)
    assert len(path) > frame.snap.d4                                      # a longer way round ...
    chosen = return_home.score_paths(frame, [path])[0][0]
    assert chosen == gains[margins >= 1].max()                            # ... for the most gain among the safe ones


def test_with_the_enemy_close_return_keeps_the_safest_path():
    frame = expedition((25, 27))
    options = return_home.candidates(frame)
    _, _, margins = return_home.score_paths(frame, options)
    assert (margins < 1).all()                                            # nothing is safe enough to choose for gain
    assert return_home.score_paths(frame, [return_home.choose_path(frame)])[2][0] == margins.max()


def test_return_steps_along_the_chosen_path_and_is_valid_only_with_a_trail():
    frame = expedition((25, 45))
    assert return_home.move(frame) == direction_to(frame.pos, return_home.choose_path(frame)[0])
    assert not return_home.valid(frame_of(scene(world_with(CLOSE), CLOSE)))
    assert return_home.valid(frame_of(scene(world_with(CLOSE, my_trail=[(25, 20)]), CLOSE)))


# ---- HUNT ------------------------------------------------------------------------------------------------------------------------

ENEMY_LAND = [(y, x) for y in range(24, 27) for x in range(30, 33)]


def north_trip():
    """The enemy went north from its land and west along row 10: its head is at (10, 22), 22 steps from home."""
    return [(y, 31) for y in range(23, 9, -1)] + [(10, x) for x in range(30, 21, -1)]


def test_hunt_needs_a_trail_it_can_cut_before_the_enemy_is_home():
    assert not hunt.valid(frame_of(scene(world_with(CLOSE), CLOSE)))                         # no trail at all
    far_out = frame_of(scene(world_with(CLOSE, opp_land=ENEMY_LAND, opp_trail=north_trip()), CLOSE))
    assert hunt.valid(far_out) and hunt.plan(far_out).best_chance == 1.0
    assert hunt.move(far_out) in (N, E) and hunt.move(far_out) in far_out.safe
    nearly_home = frame_of(scene(world_with(CLOSE, opp_land=ENEMY_LAND, opp_trail=[(23, 31), (22, 31)]), CLOSE))
    assert not hunt.valid(nearly_home)                     # 2 steps from home, 15 from me: it is home long before


def hunt_belief(share, cfg=None):
    """`share` of the belief is the north trip above; the rest has no trail."""
    land = bitboard.from_cells([flat(y, x) for y, x in ENEMY_LAND])
    trail = [flat(y, x) for y, x in north_trip()]
    belief = belief_of([dict(pos=flat(10, 22), weight=share, trail=trail, opp_land=land),
                        dict(pos=flat(25, 40), weight=1.0 - share, opp_land=land)])
    return frame_of(scene(world_with(CLOSE), CLOSE, belief), cfg=cfg)


def test_hunts_chance_is_the_share_of_the_belief_it_can_cut():
    likely = hunt_belief(0.5)
    assert abs(hunt.plan(likely).best_chance - 0.5) < 1e-9
    assert hunt.valid(likely)
    assert abs(hunt.plan(hunt_belief(0.3)).best_chance - 0.3) < 1e-9
    assert not hunt.valid(hunt_belief(0.3))                                   # below hunt_min_chance (0.4)
    assert not hunt.valid(hunt_belief(0.5, cfg=BehaviorConfig(hunt_min_chance=0.7)))


def test_hunt_takes_the_move_that_cuts_the_likelier_trail():
    def two_trails(north):
        # each trail can be cut by exactly one first move: its enemy is 10 steps from its own land, I arrive in 10
        # steps going straight at it and in 12 any other way
        a = dict(pos=flat(15, 19), weight=north, trail=[flat(15, 18), flat(15, 19)], opp_land=bitboard.from_cells([flat(15, 29)]))
        b = dict(pos=flat(35, 19), weight=1 - north, trail=[flat(35, 18), flat(35, 19)],
                 opp_land=bitboard.from_cells([flat(35, 29)]))
        return frame_of(scene(world_with(CLOSE), CLOSE, belief_of([a, b])))

    north = two_trails(0.6)
    assert hunt.move(north) == N and abs(hunt.plan(north).chance[N] - 0.6) < 1e-9
    assert abs(hunt.plan(north).chance[S] - 0.4) < 1e-9 and hunt.plan(north).chance[E] == 0.0
    assert hunt.move(two_trails(0.45)) == S                                   # 0.55 to the south


def test_hunt_never_steps_onto_my_trail_or_the_enemy():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21)], opp_trail=[(25, 23), (25, 24)], opp=(25, 24))
    frame = frame_of(scene(world, CLOSE))
    direction = hunt.move(frame)
    assert direction is None or direction in frame.safe
    assert direction is None or step(frame.pos, direction) not in (flat(25, 24), flat(25, 20))


# ---- TRACK TARGET ------------------------------------------------------------------------------------------------------------


def test_tracking_is_for_when_the_enemy_is_out_of_view():
    assert not track.valid(frame_of(scene(world_with(CLOSE, opp=(25, 22)), CLOSE)))         # seen: nothing to find
    assert track.valid(frame_of(scene(world_with(CLOSE), CLOSE)))


def test_tracking_heads_for_where_the_enemy_most_likely_is():
    residual = np.zeros((50, 50))
    residual[10, 40] = 0.3                                                 # a single likely cell ...
    residual[38:43, 6:11] = 0.7 / 25                                       # ... and a likelier region, spread out
    frame = frame_of(scene(big_world(), CLOSE, belief_of(residual=residual)))
    cell = track.target(frame)
    y, x = divmod(cell, 50)
    assert abs(y - 40) <= 5 and abs(x - 8) <= 5                            # a view from there covers the region
    assert track.move(frame) in (S, W)                                     # and it sets off towards it


def test_of_two_equally_likely_places_tracking_goes_to_the_nearer():
    residual = np.zeros((50, 50))
    residual[25, 2] = 0.5                                                  # 12 columns west of my land's edge
    residual[25, 47] = 0.5                                                 # 25 columns east of it
    frame = frame_of(scene(big_world(), CLOSE, belief_of(residual=residual)))
    assert track.target(frame) % 50 < 25 and track.move(frame) == W


def test_tracking_goes_for_a_likely_place_however_far_it_is():
    residual = np.zeros((50, 50))
    residual[2:5, 45:48] = 1.0 / 9                                        # 50+ steps away: the old 30-step reach missed it
    frame = frame_of(scene(big_world(), CLOSE, belief_of(residual=residual)))
    y, x = divmod(track.target(frame), 50)
    assert abs(y - 3) <= 5 and abs(x - 46) <= 5
    assert track.move(frame) in (N, E)


def test_tracking_an_unseen_enemy_always_gives_a_safe_move():
    rng = np.random.default_rng(0)
    for _ in range(6):
        residual = rng.random((50, 50)) * (rng.random((50, 50)) < 0.05)
        residual[20:31, 13:24] = 0.0
        residual[0, 0] += 1.0
        frame = frame_of(scene(big_world(), CLOSE, belief_of(residual=residual)))
        assert track.move(frame) in frame.safe


# ---- the table -----------------------------------------------------------------------------------------------------------------------


def test_every_valid_state_has_a_move_that_the_engine_accepts():
    boards = [
        scene(world_with(CLOSE), CLOSE),
        scene(world_with(CLOSE, my_trail=[(25, 20), (25, 21), (24, 21)]), CLOSE),
        scene(big_world(me=(22, 20)), CLOSE),
        scene(world_with(CLOSE, opp_trail=[(25, 25), (25, 24), (25, 23)]), CLOSE),
        scene(world_with(CLOSE, my_trail=[(25, 20), (25, 21), (25, 22)]), CLOSE),
    ]
    for view in boards:
        frame = frame_of(view)
        for state, valid in zip(State, moves.valid_states(frame)):
            if valid:
                direction = moves.move_for(state, frame)
                assert direction in frame.safe, (state, direction)


def test_hunts_quick_bound_is_never_below_its_chance():
    rng = np.random.default_rng(4)
    for _ in range(30):
        hyps = []
        for _ in range(int(rng.integers(1, 6))):
            y, x = int(rng.integers(2, 48)), int(rng.integers(2, 48))
            trail = [flat(y, min(49, x + k)) for k in range(int(rng.integers(1, 8)))]
            land = bitboard.from_cells([flat(int(rng.integers(0, 50)), int(rng.integers(0, 50)))])
            hyps.append(dict(pos=trail[-1], weight=float(rng.random()), trail=trail, opp_land=land))
        frame = frame_of(scene(world_with(CLOSE), CLOSE, belief_of(hyps)))
        result = hunt.plan(frame)
        assert hunt.upper_bound(frame) >= (result.best_chance if result else 0.0) - 1e-12
