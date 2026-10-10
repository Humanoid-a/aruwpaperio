"""A small rig for driving the belief filter the way the runtime does, with scenes built by hand."""

import numpy as np

from candidate.agent.belief.after_move import MyMove
from candidate.agent.belief.filter import BeliefFilter
from candidate.agent.data.hypotheses import HypothesisStore
from candidate.agent.data.memory import MemoryMap
from candidate.agent.data.territory import TerritoryDiffs
from candidate.agent.data.types import Direction, MoveResult
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.agent.geometry.zobrist import DEFAULT_KEYS
from candidate.agent.prediction import Prediction
from candidate.agent.rules.match import MatchRules

FAR = MatchRules.from_respawns(flat(25, 4), flat(25, 45))
NEAR = MatchRules.from_respawns(flat(25, 10), flat(25, 16))   # the opponent's start edge is inside my window


class Rig:
    """A filter plus the memory it works with, stepped the way the runtime steps them: seed the memory with
    the start lands, then for each frame update memory with the window and observe, and after the agent's move
    apply it."""

    def __init__(self, rules, config=None, motion=None, seed_memory=True):
        self.rules = rules
        self.memory = MemoryMap.new()
        if seed_memory:
            self.memory.reset_my_land(rules.me.start_cells, 0)
            self.memory.reset_opponent_land(rules.opp.start_cells, 0)
        self.belief = BeliefFilter(config, motion)
        self.belief.begin_match(rules)

    @property
    def state(self):
        return self.belief.state

    def step(self, obs, killed=False, force_reseed=False):
        """One observation: the window goes into memory, then the belief takes it in."""
        self.memory.update(obs)
        return self.belief.observe(obs, self.memory, killed, force_reseed)

    def move(self, move):
        """My move, applied at the end of the frame."""
        self.belief.after_my_move(move)

    def total_mass(self):
        return float(self.state.store.weight.sum()) + self.state.residual.total()


def prediction(result, *, killed=False, region=None, gain=None, dest=flat(25, 14), new_pos=flat(25, 14), new_trail=()):
    return Prediction(Direction.EAST, dest, result, killed, new_pos, tuple(new_trail), region, gain, False)


def plain_move(trail_after, pos_after=None, count=9):
    """An ordinary step that leaves me with `trail_after` (as (y, x) cells; the last is my head) and my land alone."""
    cells = tuple(flat(y, x) for y, x in trail_after)
    new_pos = pos_after if pos_after is not None else cells[-1]
    return MyMove(MoveResult.NORMAL, prediction(MoveResult.NORMAL, dest=new_pos, new_pos=new_pos, new_trail=cells), cells[:-1], count)


def prime(rig, entries, opp_count=9):
    """Replace the belief's hypotheses with `entries` (pos, trail cells as (y, x), weight, count), as if reached
    by play, standing on the start land of NEAR's opponent and with an empty residual."""
    state = rig.state
    n = len(entries)
    trails = [bitboard.from_cells(flat(y, x) for y, x in cells) for _, cells, _, _ in entries]
    state.opp_diffs = TerritoryDiffs()
    state.opp_base = bitboard.from_cells(NEAR.opp.start_cells)
    state.store = HypothesisStore.build(
        pos=[p for p, *_ in entries], last_dir=[0 if cells else -1 for _, cells, _, _ in entries],
        trail_len=[len(cells) for _, cells, _, _ in entries], trail_bb=np.stack(trails), opp_land_id=[0] * n,
        opp_count=[c for *_, c in entries], flags=[0] * n, weight=[w for _, _, w, _ in entries],
        hash=[DEFAULT_KEYS.hash_position("opp_pos", p) ^ DEFAULT_KEYS.hash_board("opp_trail", t) for (p, *_), t in zip(entries, trails)],
    )
    state.store.weight = state.store.weight / state.store.weight.sum()
    state.first = False
    state.opp_count = opp_count
    state.residual.mass[:] = 0
