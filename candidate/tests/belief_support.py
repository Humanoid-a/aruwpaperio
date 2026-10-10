"""Helpers for building belief objects from known true states, for tests."""

import numpy as np

from candidate.agent import adapter
from candidate.agent.belief.generate import Situation
from candidate.agent.belief.state import BeliefState, MyState, start_state
from candidate.agent.data.cells import MY_TERRITORY, OPP_TERRITORY
from candidate.agent.data.territory import TerritoryDiffs
from candidate.agent.data.world import WorldState
from candidate.agent.geometry import bitboard
from candidate.agent.rules.match import MatchRules
from candidate.tests.live import LiveGame


def stale_base(world: WorldState, rng: np.random.Generator, noise: int = 25) -> np.ndarray:
    """A made-up memory of the opponent's land: the true land with some cells wrongly
    missing and some wrongly present, as old memory would be."""
    true = bitboard.from_mask(world.territory == OPP_TERRITORY)
    mask = bitboard.to_mask(true)
    flips = rng.integers(0, mask.size, size=noise)
    mask.flat[flips] = ~mask.flat[flips]
    return bitboard.from_mask(mask)


def belief_state_of(
    world: WorldState,
    rules: MatchRules,
    opp_base: np.ndarray | None = None,
    my_base: np.ndarray | None = None,
    weight: float = 1.0,
) -> BeliefState:
    """A belief with one hypothesis holding exactly the true state of `world`: where the opponent is, its trail,
    its land and count, and my land. The lands are expressed against the given bases (by default the true
    lands, so the differences are empty), and I am where I am in the world."""
    state = start_state(rules)
    true_opp = bitboard.from_mask(world.territory == OPP_TERRITORY)
    true_mine = bitboard.from_mask(world.territory == MY_TERRITORY)
    state.opp_base = true_opp if opp_base is None else opp_base
    state.my_base = true_mine if my_base is None else my_base
    state.opp_diffs, state.my_diffs = TerritoryDiffs(), TerritoryDiffs()
    state.store = state.single(
        world.opp_pos, true_opp, bitboard.from_cells(world.opp_trail), world.opp_count,
        my_land_id=int(state.my_diffs.intern_territory(true_mine, state.my_base)[0]),
    )
    state.store.weight[:] = weight
    state.me = MyState(world.my_pos, bitboard.from_cells(world.my_trail), world.my_count, None)
    state.first = False
    return state


def situation_of(state: BeliefState, world: WorldState) -> Situation:
    """The situation while the opponent moves from `world` (I am where I am in it)."""
    obs = adapter.observation_from_game(LiveGame(world, state.rules).me, 0, None)
    return Situation.of(state, obs)
