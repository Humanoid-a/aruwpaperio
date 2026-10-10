"""Step 4 of the Q build: the five states' own logic and their validity, on small hand-built boards.

The default board: I start on the 3x3 around (25, 18) (rows 24-26, cols 17-19), the enemy on the 3x3 around (25, 31).
"""

import numpy as np
import pytest

from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import direction_to, flat, step
from candidate.agent.q import movers
from candidate.agent.q.config import QConfig
from candidate.agent.q.frame import Frame, Hints
from candidate.agent.q.snapshot import build_snapshot
from candidate.agent.q.types import State
from candidate.agent.rules.match import MatchRules
from candidate.tests.q_support import CLOSE, belief_of, bb, cells, scene, world_with

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
    mask = movers.valid_states(frame_of(scene(world_with(CLOSE), CLOSE)))
    assert not mask[State.RETURN] and not mask[State.HUNT]
    assert mask[State.PATROL] and mask[State.TRACK]


def test_with_a_trail_return_is_valid_and_patrol_never_is():
    mask = movers.valid_states(frame_of(scene(world_with(CLOSE, my_trail=[(25, 20), (25, 21)]), CLOSE)))
    assert mask[State.RETURN] and not mask[State.PATROL] and mask[State.TRACK]


def test_hunt_needs_a_known_enemy_trail_cell_i_can_reach():
    assert not movers.valid_states(frame_of(scene(world_with(CLOSE), CLOSE)))[State.HUNT]
    seen = world_with(CLOSE, opp_trail=[(25, 25), (25, 24), (25, 23)], opp_land=[(y, x) for y in range(24, 27) for x in range(30, 33)])
    assert movers.valid_states(frame_of(scene(seen, CLOSE)))[State.HUNT]


def test_hunt_is_invalid_when_the_only_known_trail_cell_is_under_the_enemy():
    world = world_with(CLOSE, opp_trail=[(25, 22)])                        # a one-cell trail: the enemy stands on it
    mask = movers.valid_states(frame_of(scene(world, CLOSE)))
    assert not mask[State.HUNT]


def test_nothing_is_valid_when_no_move_is_available():
    view = scene(world_with(CLOSE), CLOSE)
    view.__dict__["safe_moves"] = ()
    assert not movers.valid_states(frame_of(view)).any()


# ---- PATROL -------------------------------------------------------------------------------------------------------------


def walk(world_at, start, turns, state):
    """Play `state` for some frames on a board where I only move and nothing else changes; returns my cells."""
    pos, seen = start, []
    for turn in range(turns):
        view = scene(world_at(pos), CLOSE, turn=turn + 40)
        frame = frame_of(view)
        direction = movers.move_for(state, frame)
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
    cfg = QConfig(exit_weight_room=1.0, exit_weight_safety=0.0, exit_weight_info=0.0, exit_weight_walk=0.0)
    world = big_world(opp_land=[(y, x) for y in range(15, 36) for x in range(6, 14)])      # a wall of enemy land on the west
    frame = frame_of(scene(world, CLOSE), cfg=cfg)
    assert frame.exits.best % 50 != 14


def test_the_information_term_prefers_an_edge_not_looked_at_for_a_long_time():
    cfg = QConfig(exit_weight_room=0.0, exit_weight_safety=0.0, exit_weight_info=1.0, exit_weight_walk=0.0)
    view = scene(big_world(), CLOSE, turn=100)
    view.memory.last_seen[:, 19:] = 100                                   # the east was seen just now
    frame = frame_of(view, cfg=cfg)
    assert frame.exits.best % 50 == 14


def test_the_walk_term_prefers_a_near_exit_when_the_rest_is_equal():
    cfg = QConfig(exit_weight_room=0.0, exit_weight_safety=0.0, exit_weight_info=0.0, exit_weight_walk=1.0)
    frame = frame_of(scene(big_world(me=(25, 21)), CLOSE), cfg=cfg)
    assert frame.exits.best % 50 == 22 and abs(frame.exits.best // 50 - 25) <= 1


def test_patrol_off_my_land_without_a_trail_walks_home():
    world = world_with(CLOSE, me=(25, 23))                                # standing outside my land, no trail
    frame = frame_of(scene(world, CLOSE))
    assert not frame.snap.on_land
    assert movers.patrol(frame) == W


# ---- EXPAND ---------------------------------------------------------------------------------------------------------------


def test_expand_from_an_exit_steps_out_of_my_land():
    exit_cell = frame_of(scene(big_world(), CLOSE)).exits.best
    frame = frame_of(scene(big_world(me=divmod(exit_cell, 50)), CLOSE))
    direction = movers.expand(frame)
    assert direction is not None and not in_land(frame, direction)
    assert movers.expand_valid(frame)


def test_expand_inside_walks_to_the_exit_through_my_land():
    frame = frame_of(scene(big_world(me=(22, 16)), CLOSE))
    assert frame.pos != frame.exits.best
    direction = movers.expand(frame)
    assert direction is not None and in_land(frame, direction)
    before = frame.exits.walk[frame.pos]
    assert frame.exits.walk[step(frame.pos, direction)] == before - 1 or frame.exits.walk[frame.pos] == 0


def test_expand_outside_goes_straight_on_in_the_direction_it_set_out_in():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21)])
    frame = frame_of(scene(world, CLOSE))
    assert frame.heading() == E and movers.expand(frame) == E
    hinted = frame_of(scene(world_with(CLOSE, my_trail=[(24, 20), (24, 21)]), CLOSE), Hints(outward=N))
    assert movers.expand(hinted) == N                                    # remembers the expedition's direction


def test_expand_never_turns_back_onto_its_own_trail():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21)])
    frame = frame_of(scene(world, CLOSE), Hints(outward=W))               # a stale hint pointing back along the trail
    assert not movers.expand_valid(frame) and movers.expand(frame) is None


def test_expand_is_invalid_when_the_next_outward_cell_is_off_the_board():
    rules = MatchRules.from_respawns(flat(25, 46), flat(25, 4))
    world = world_with(rules, my_trail=[(25, 48), (25, 49)])
    frame = frame_of(scene(world, rules))
    assert frame.heading() == E and not movers.expand_valid(frame)


# ---- RETURN -------------------------------------------------------------------------------------------------------------------


def expedition(enemy_at, trail=((25, 20), (25, 21), (25, 22), (25, 23), (25, 24)), **kw):
    belief = belief_of([dict(pos=flat(*enemy_at), weight=1.0)])
    return frame_of(scene(world_with(CLOSE, my_trail=list(trail)), CLOSE, belief), **kw)


def test_there_is_a_direct_path_and_one_per_sideways_length_in_both_directions():
    frame = expedition((25, 45))
    assert len(movers.return_candidates(frame)) == 1 + 2 * 8
    narrow = Frame(frame.view, frame.snap, frame.hints, QConfig(return_max_sideways=3))
    assert len(movers.return_candidates(narrow)) == 1 + 2 * 3


def test_every_candidate_path_really_gets_home_without_touching_my_trail():
    frame = expedition((25, 45))
    for path in movers.return_candidates(frame):
        assert bitboard.has_cell(frame.snap.my_land_bb, path[-1])
        assert not any(bitboard.has_cell(frame.snap.my_trail_bb, c) for c in path)
        steps = [frame.pos] + path
        assert all(direction_to(a, b) is not None for a, b in zip(steps, steps[1:]))


def test_the_scores_of_the_direct_path_are_what_the_definition_says():
    frame = expedition((25, 45))
    direct = [list(frame.snap.home_path)]
    gain, time, margin = movers.score_paths(frame, direct)
    assert time[0] == 6                               # up, then five west to my land
    assert gain[0] == 10                              # five trail cells and five new ones, nothing enclosed
    assert margin[0] == 21 - 6                        # the enemy needs 21 to reach my head at (25, 24)


def test_with_the_enemy_far_away_return_takes_a_wider_loop_for_more_gain():
    frame = expedition((25, 45))
    path = movers.choose_return_path(frame)
    assert len(path) > len(frame.snap.home_path)
    gains, _, margins = movers.score_paths(frame, [path, list(frame.snap.home_path)])
    assert gains[0] > gains[1] and margins[0] >= frame.cfg.return_buffer


def test_with_the_enemy_close_return_picks_the_direct_path_home():
    frame = expedition((25, 27))
    assert movers.choose_return_path(frame) == list(frame.snap.home_path)


def test_return_steps_along_the_chosen_path():
    frame = expedition((25, 45))
    path = movers.choose_return_path(frame)
    assert movers.return_(frame) == direction_to(frame.pos, path[0])


def test_when_no_path_has_the_buffer_the_largest_margin_wins():
    frame = expedition((25, 27))
    candidates = movers.return_candidates(frame)
    _, _, margins = movers.score_paths(frame, candidates)
    assert (margins < frame.cfg.return_buffer).all()
    chosen = movers.choose_return_path(frame)
    assert margins[candidates.index(chosen)] == margins.max()


def test_return_counts_the_enemys_land_double_when_choosing():
    trail = ((25, 20), (25, 21), (25, 22), (25, 23), (25, 24))
    plain = expedition((25, 45), trail)
    rich_world = world_with(CLOSE, my_trail=list(trail), opp_land=[(23, 22), (23, 23), (23, 24), (22, 22), (22, 23), (22, 24)])
    rich = frame_of(scene(rich_world, CLOSE, belief_of([dict(pos=flat(25, 45), weight=1.0)])))
    assert max(movers.score_paths(rich, movers.return_candidates(rich))[0]) > max(movers.score_paths(plain, movers.return_candidates(plain))[0])


# ---- HUNT ------------------------------------------------------------------------------------------------------------------------


def hunting(trail, enemy=None):
    world = world_with(CLOSE, opp_trail=list(trail), opp_land=[(y, x) for y in range(24, 27) for x in range(30, 33)])
    return frame_of(scene(world, CLOSE))


def test_hunt_steps_toward_the_nearest_known_enemy_trail_cell_by_a_shortest_path():
    frame = hunting([(25, 25), (25, 24), (25, 23)])
    direction = movers.hunt(frame)
    assert direction in frame.safe
    assert frame.snap.hunt_dist[step(frame.pos, direction)] == frame.snap.d2 - 1


def test_hunt_chooses_the_nearer_of_two_trails():
    frame = hunting([(19, 18), (18, 18)])                                # a trail 5 rows north, within the view
    near = movers.hunt(frame)
    assert near == N


def test_hunt_avoids_my_own_trail():
    world = world_with(CLOSE, my_trail=[(25, 20), (25, 21)], opp_trail=[(25, 23), (25, 24)], opp=(25, 24))
    frame = frame_of(scene(world, CLOSE))
    direction = movers.hunt(frame)
    assert direction is None or direction in frame.safe                  # never onto my trail
    path = movers.paths.path_from_map(frame.snap.hunt_dist, frame.pos)
    assert path is None or not any(bitboard.has_cell(frame.snap.my_trail_bb, c) for c in path)


# ---- TRACK TARGET ------------------------------------------------------------------------------------------------------------


def test_tracking_a_visible_enemy_just_out_of_range_closes_in():
    frame = frame_of(scene(world_with(CLOSE, opp=(25, 24)), CLOSE))        # 6 away from (25, 18)
    assert movers.track(frame) == E


def test_tracking_a_visible_enemy_that_is_too_close_backs_off():
    frame = frame_of(scene(world_with(CLOSE, opp=(25, 20)), CLOSE))        # 2 away
    dest = step(frame.pos, movers.track(frame))
    assert abs(dest % 50 - 20) + abs(dest // 50 - 25) >= 3 and bitboard.has_cell(frame.snap.my_land_bb, dest)


def test_tracking_prefers_to_stay_in_range_and_on_my_land():
    frame = frame_of(scene(world_with(CLOSE, me=(25, 18), opp=(25, 22)), CLOSE))      # 4 away: already in range
    direction = movers.track(frame)
    dest = step(frame.pos, direction)
    assert 3 <= abs(dest % 50 - 22) + abs(dest // 50 - 25) <= 5 and in_land(frame, direction)


def test_tracking_an_unseen_enemy_heads_for_the_cell_whose_view_covers_the_most_belief():
    residual = np.zeros((50, 50))
    residual[25, 9] = 1.0                                                  # just beyond the west edge of my land
    frame = frame_of(scene(big_world(), CLOSE, belief_of(residual=residual)))
    direction = movers.track(frame)
    assert direction == W and in_land(frame, direction)


def test_tracking_does_not_walk_where_the_enemy_could_get_first():
    residual = np.zeros((50, 50))
    residual[25, 9] = 1.0
    frame = frame_of(scene(world_with(CLOSE, me=(25, 19)), CLOSE, belief_of(residual=residual)))
    direction = movers.track(frame)
    assert direction is not None and in_land(frame, direction)              # (25,14) would cover it but the enemy gets there first


def test_tracking_an_unseen_enemy_always_gives_a_safe_move():
    rng = np.random.default_rng(0)
    for _ in range(6):
        residual = rng.random((50, 50)) * (rng.random((50, 50)) < 0.05)
        residual[20:31, 13:24] = 0.0
        residual[0, 0] += 1.0
        frame = frame_of(scene(big_world(), CLOSE, belief_of(residual=residual)))
        assert movers.track(frame) in frame.safe


# ---- the table -----------------------------------------------------------------------------------------------------------------------


def test_every_valid_state_has_a_move_that_the_engine_accepts():
    boards = [
        scene(world_with(CLOSE), CLOSE),
        scene(world_with(CLOSE, my_trail=[(25, 20), (25, 21), (24, 21)]), CLOSE),
        scene(big_world(me=(22, 20)), CLOSE),
        scene(world_with(CLOSE, opp_trail=[(25, 25), (25, 24), (25, 23)]), CLOSE),
    ]
    for view in boards:
        frame = frame_of(view)
        for state, valid in zip(State, movers.valid_states(frame)):
            if valid:
                direction = movers.move_for(state, frame)
                assert direction in frame.safe, (state, direction)
