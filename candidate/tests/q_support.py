"""Hand-built scenes for testing the Q inputs and the action-states.

scene() turns a WorldState into the three things a TurnView holds, consistently: the Observation the framework
would give me (through the real adapter), a MemoryMap that knows the whole board, and a belief you describe
yourself with belief_of(). Nothing here runs the belief filter, so every number in a test is one the test chose.
"""

import numpy as np

from candidate.agent import adapter
from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.belief_view import BeliefDetail, BeliefView
from candidate.agent.data.memory import MemoryMap
from candidate.agent.data.world import WorldState
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.agent.rules.match import MatchRules
from candidate.agent.runtime import TurnView
from candidate.tests.builders import put_agent, put_territory, put_trail
from candidate.tests.live import LiveGame

from candidate.agent.data.types import Side

FAR = MatchRules.from_respawns(flat(25, 4), flat(25, 45))
CLOSE = MatchRules.from_respawns(flat(25, 18), flat(25, 31))


def cells(*yx):
    return [flat(y, x) for y, x in yx]


def bb(*yx):
    return bitboard.from_cells(cells(*yx))


def memory_of(world: WorldState) -> MemoryMap:
    """A memory that knows the whole board exactly as the world has it."""
    memory = MemoryMap.new()
    memory.territory[:] = world.territory
    memory.trail[:] = world.trail
    memory.last_seen[:] = 0
    return memory


def belief_of(hyps=(), residual=None, sighted_at=None, turn=0) -> BeliefView:
    """A belief you describe. `hyps` is a list of dicts: pos (flat), weight, and optionally trail (flat cells),
    opp_land and my_land (bitboards). `residual` is a (50, 50) array of mass. Weights and residual are
    normalised together to total 1."""
    residual = np.zeros((BOARD_HEIGHT, BOARD_WIDTH)) if residual is None else np.asarray(residual, dtype=np.float64)
    total = sum(h["weight"] for h in hyps) + residual.sum()
    assert total > 0, "a belief needs some mass"
    n = len(hyps)
    weight = np.array([h["weight"] for h in hyps], dtype=np.float64) / total
    residual = residual / total
    pos = np.array([h["pos"] for h in hyps], dtype=np.int32)
    trail = np.stack([bitboard.from_cells(h.get("trail", ())) for h in hyps]) if n else np.zeros((0, BOARD_HEIGHT), dtype=np.int64)
    opp_lands, opp_index = _distinct([h.get("opp_land", bitboard.empty()) for h in hyps])
    my_lands, my_index = _distinct([h.get("my_land", bitboard.empty()) for h in hyps])
    detail = BeliefDetail(weight, pos, trail, opp_index, opp_lands, my_index, my_lands, residual)
    position = residual.copy()
    for h, w in zip(hyps, weight):
        position.flat[h["pos"]] += w
    return BeliefView(
        turn=turn, position=position.astype(np.float32), support=position > 0, n_hypotheses=n,
        hypothesis_mass=float(weight.sum()), residual_mass=float(residual.sum()), entropy=0.0,
        sighted_at=sighted_at, events=(), _pos=pos, _weight=weight, _trail_bb=trail,
        _trail_len=bitboard.popcount(trail) if n else np.zeros(0, dtype=np.int64),
        _opp_count=np.zeros(n, dtype=np.int32), detail=detail,
    )


def _distinct(lands):
    if not lands:
        return np.zeros((0, BOARD_HEIGHT), dtype=np.int64), np.zeros(0, dtype=np.int64)
    stacked = np.stack(lands).astype(np.int64)
    unique, inverse = np.unique(stacked, axis=0, return_inverse=True)
    return unique, inverse.reshape(-1).astype(np.int64)


def scene(world: WorldState, rules: MatchRules = FAR, belief: BeliefView | None = None, turn: int = 40,
          events=(), opp_last_seen=None) -> TurnView:
    """A TurnView of `world` from my side. The memory is the whole true board; the belief is whatever you pass
    (by default: the enemy is exactly where the world has it, certainly)."""
    game = LiveGame(world, rules)
    obs = adapter.observation_from_game(game.me, turn, None)
    memory = memory_of(world)
    if belief is None:
        belief = belief_of(
            [dict(pos=world.opp_pos, weight=1.0, trail=world.opp_trail,
                  opp_land=bitboard.from_mask(world.territory == 3), my_land=bitboard.from_mask(world.territory == 1))],
            sighted_at=obs.opp_seen_at,
        )
    return TurnView(obs, memory, rules, tuple(events), opp_last_seen, 0, belief)


def open_world(rules: MatchRules = FAR) -> WorldState:
    """Both players on their start land, nothing else."""
    return rules.initial_world()


def world_with(rules: MatchRules = FAR, *, me=None, opp=None, my_land=(), opp_land=(), my_trail=(), opp_trail=()):
    world = rules.initial_world()
    if my_land:
        put_territory(world, Side.ME, my_land)
    if opp_land:
        put_territory(world, Side.OPP, opp_land)
    if my_trail:
        put_trail(world, Side.ME, my_trail)
    if opp_trail:
        put_trail(world, Side.OPP, opp_trail)
    if me is not None and not my_trail:
        put_agent(world, Side.ME, me)
    if opp is not None and not opp_trail:
        put_agent(world, Side.OPP, opp)
    world.my_count = int((world.territory == 1).sum())
    world.opp_count = int((world.territory == 3).sum())
    return world


def rotate_world(world: WorldState, rules: MatchRules, k: int, flip: bool = False):
    """The same game with the whole board mirrored left-right (if `flip`) and then rotated by k quarter turns
    (np.rot90), with the respawns moved along with it."""

    def cell(c):
        y, x = divmod(c, BOARD_WIDTH)
        if flip:
            x = BOARD_WIDTH - 1 - x
        for _ in range(k):
            y, x = BOARD_WIDTH - 1 - x, y             # np.rot90 once: element (y, x) goes to (49 - x, y)
        return flat(y, x)

    def array(a):
        return np.ascontiguousarray(np.rot90(a[:, ::-1] if flip else a, k=k))

    turned = WorldState(
        array(world.territory), array(world.trail), cell(world.my_pos), cell(world.opp_pos),
        [cell(c) for c in world.my_trail], [cell(c) for c in world.opp_trail], world.my_count, world.opp_count,
    )
    return turned, MatchRules.from_respawns(cell(rules.me.respawn), cell(rules.opp.respawn))
