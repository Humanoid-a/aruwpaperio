"""Test-only bridge to the framework's real move rules.

Builds a real GameState from one of our WorldStates, applies a move with the
framework's MoveResolver, and converts the result back, so the reference and
fast transitions can be compared with the true rules on hand-built boards and
on long randomised games. Tests may import the framework; production code
may not.
"""

import random
from collections import Counter
from dataclasses import dataclass

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.cells import FREE, MY_TERRITORY, MY_TRAIL, OPP_TERRITORY, OPP_TRAIL
from candidate.agent.data.types import Direction, MoveResult, Side
from candidate.agent.data.world import WorldState
from candidate.agent.geometry.grid import DX, DY, flat, in_bounds, manhattan, to_yx
from candidate.agent.rules.match import MatchRules
from territorygame.api.direction import Direction as RealDirection
from territorygame.api.grid_position import GridPosition
from territorygame.api.move_result import MoveResult as RealResult
from territorygame.domain.agent import Agent
from territorygame.domain.board import Board
from territorygame.domain.game_state import GameState
from territorygame.domain.player import Player
from territorygame.domain.player_id import PlayerId
from territorygame.rules.move_resolver import MoveResolver
from territorygame.rules.respawn_service import RespawnService
from territorygame.rules.territory_resolver import TerritoryResolver

_REAL_DIRECTION = {
    Direction.NORTH: RealDirection.NORTH,
    Direction.EAST: RealDirection.EAST,
    Direction.SOUTH: RealDirection.SOUTH,
    Direction.WEST: RealDirection.WEST,
}
_RESULT = {
    RealResult.MOVED: MoveResult.NORMAL,
    RealResult.CAPTURED: MoveResult.CAPTURED,
    RealResult.DIED: MoveResult.DIED,
    RealResult.INVALID: MoveResult.INVALID,
}
_PLAYER = {Side.ME: PlayerId(0), Side.OPP: PlayerId(1)}


def _position(c: int) -> GridPosition:
    y, x = to_yx(c)
    return GridPosition(x, y)


def real_state_from_world(world: WorldState, rules: MatchRules) -> GameState:
    """A framework GameState holding `world`; ME is player 0, OPP player 1."""
    board = Board(BOARD_WIDTH, BOARD_HEIGHT)
    for code, side in ((MY_TERRITORY, Side.ME), (OPP_TERRITORY, Side.OPP)):
        for y, x in np.argwhere(world.territory == code):
            board.set_territory_owner(GridPosition(int(x), int(y)), _PLAYER[side])
    for code, side in ((MY_TRAIL, Side.ME), (OPP_TRAIL, Side.OPP)):
        for y, x in np.argwhere(world.trail == code):
            board.set_trail_owner(GridPosition(int(x), int(y)), _PLAYER[side])
    players = []
    for side, pos, trail in ((Side.ME, world.my_pos, world.my_trail), (Side.OPP, world.opp_pos, world.opp_trail)):
        role = rules.role(side)
        agent = Agent(_position(pos), _position(role.respawn))
        for c in trail:
            agent.append_trail(_position(c))
        players.append(Player(_PLAYER[side], agent, [_position(c) for c in role.start_cells]))
    return GameState(board, players, 10_000)


def world_from_real(state: GameState) -> WorldState:
    """Our WorldState (ME = player 0) from a framework GameState. Reads the
    board's cells directly because the public per-cell accessors are slow."""
    cells = state.get_board()._cells

    def layer(attribute: str, mine: int, theirs: int) -> np.ndarray:
        owners = np.array(
            [[-1 if getattr(c, attribute) is None else getattr(c, attribute).index for c in row] for row in cells],
            dtype=np.int8,
        )
        return np.where(owners == 0, mine, np.where(owners == 1, theirs, FREE)).astype(np.int8)

    territory = layer("_territory_owner", MY_TERRITORY, OPP_TERRITORY)
    trail = layer("_trail_owner", MY_TRAIL, OPP_TRAIL)
    me, opp = state.get_players()
    to_flat = lambda p: flat(p.y, p.x)  # noqa: E731
    return WorldState(
        territory,
        trail,
        to_flat(me.get_agent().get_position()),
        to_flat(opp.get_agent().get_position()),
        [to_flat(p) for p in me.get_agent().get_active_trail()],
        [to_flat(p) for p in opp.get_agent().get_active_trail()],
        int((territory == MY_TERRITORY).sum()),
        int((territory == OPP_TERRITORY).sum()),
    )


@dataclass(frozen=True, eq=False)
class RealOutcome:
    world: WorldState
    result: MoveResult
    killed: bool


def real_apply(world: WorldState, rules: MatchRules, mover: Side, direction: Direction) -> RealOutcome:
    """Apply one move with the framework's own rules."""
    state = real_state_from_world(world, rules)
    return resolve_on(state, mover, direction)


def resolve_on(state: GameState, mover: Side, direction: Direction) -> RealOutcome:
    """Apply one move to a live GameState (mutating it) with the real resolver."""
    resolver = MoveResolver(RespawnService(), TerritoryResolver())
    kills_before = state.get_kill_count(_PLAYER[mover])
    result = resolver.resolve(state, _PLAYER[mover], _REAL_DIRECTION[direction])
    killed = state.get_kill_count(_PLAYER[mover]) > kills_before
    return RealOutcome(world_from_real(state), _RESULT[result], killed)


# ---- randomised play ----------------------------------------------------

STYLES = ("wander", "loop", "hunter")


def random_rules(rng: random.Random) -> MatchRules:
    """Two respawns a fairly short distance apart (so the players meet), with
    non-overlapping starting territories, sometimes against a board edge."""
    while True:
        y1, x1 = rng.randrange(BOARD_HEIGHT), rng.randrange(BOARD_WIDTH)
        y2 = min(BOARD_HEIGHT - 1, max(0, y1 + rng.randint(-12, 12)))
        x2 = min(BOARD_WIDTH - 1, max(0, x1 + rng.randint(-12, 12)))
        if max(abs(y1 - y2), abs(x1 - x2)) >= 3:
            return MatchRules.from_respawns(flat(y1, x1), flat(y2, x2))


def choose_direction(rng: random.Random, world: WorldState, side: Side, style: str, loop_length: int) -> Direction:
    """A move for `side`. Deliberately unsafe and occasionally illegal, so the
    rules are exercised on deaths, kills and invalid moves as well as captures."""
    own_terr = MY_TERRITORY if side == Side.ME else OPP_TERRITORY
    other_trail_code = OPP_TRAIL if side == Side.ME else MY_TRAIL
    pos = world.my_pos if side == Side.ME else world.opp_pos
    other_pos = world.opp_pos if side == Side.ME else world.my_pos
    trail = world.my_trail if side == Side.ME else world.opp_trail
    y, x = to_yx(pos)

    target = None
    if style == "loop" and len(trail) >= loop_length:
        own = np.argwhere(world.territory == own_terr)
        if len(own):
            target = min(((int(a), int(b)) for a, b in own), key=lambda c: abs(c[0] - y) + abs(c[1] - x))
    elif style == "hunter":
        cuts = np.argwhere(world.trail == other_trail_code)
        if len(cuts):
            target = min(((int(a), int(b)) for a, b in cuts), key=lambda c: abs(c[0] - y) + abs(c[1] - x))
        else:
            target = to_yx(other_pos)

    if target is None or rng.random() < 0.15:
        return Direction(rng.randrange(4))
    best = []
    best_distance = None
    for d in Direction:
        ny, nx = y + DY[d], x + DX[d]
        distance = abs(ny - target[0]) + abs(nx - target[1])
        if best_distance is None or distance < best_distance:
            best, best_distance = [d], distance
        elif distance == best_distance:
            best.append(d)
    return rng.choice(best)


def play_random_game(seed: int, steps: int):
    """Yield (world_before, rules, side, direction, real_outcome) for each
    move of a randomly played game on the framework's real rules. The players
    alternate; the moves are only constrained by being aimed at something."""
    rng = random.Random(seed)
    rules = random_rules(rng)
    world = rules.initial_world()
    state = real_state_from_world(world, rules)
    styles = {Side.ME: rng.choice(STYLES), Side.OPP: rng.choice(STYLES)}
    lengths = {Side.ME: rng.randint(3, 14), Side.OPP: rng.randint(3, 14)}
    for t in range(steps):
        side = Side(t % 2)
        direction = choose_direction(rng, world, side, styles[side], lengths[side])
        outcome = resolve_on(state, side, direction)
        yield world, rules, side, direction, outcome
        world = outcome.world


def classify_event(before: WorldState, side: Side, outcome: RealOutcome, rules: MatchRules) -> list[str]:
    """Names for the interesting things a move did, for coverage counting."""
    events = [outcome.result.name]
    if outcome.killed:
        events.append("killed")
    me = side == Side.ME
    other_count_before = before.opp_count if me else before.my_count
    other_count_after = outcome.world.opp_count if me else outcome.world.my_count
    if outcome.result == MoveResult.CAPTURED and other_count_after < other_count_before and not outcome.killed:
        events.append("captured_enemy_land")
    if outcome.result == MoveResult.CAPTURED and outcome.killed:
        events.append("captured_and_killed")
    role = rules.role(side)
    pos_after = outcome.world.my_pos if me else outcome.world.opp_pos
    if outcome.result == MoveResult.DIED and pos_after == role.alt_respawn:
        events.append("respawn_displaced")
    other_role = rules.role(side.other)
    other_pos_after = outcome.world.opp_pos if me else outcome.world.my_pos
    if outcome.killed and other_pos_after == other_role.alt_respawn:
        events.append("victim_respawn_displaced")
    return events


def coverage(seeds, steps) -> Counter:
    counts: Counter = Counter()
    for seed in seeds:
        for before, rules, side, _, outcome in play_random_game(seed, steps):
            counts.update(classify_event(before, side, outcome, rules))
    return counts
