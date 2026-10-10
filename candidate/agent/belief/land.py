"""Changing the lands of every hypothesis at once.

Both lands are stored as differences from a base (data.territory). Two things
change them for everybody: my own move (it takes cells from the opponent and
gives them to me, or wipes my land when I die), and the base itself moving, when
memory learns something from the window. These functions do both, and re-express
only the differences some hypothesis still uses, so the tables do not grow.
"""

import numpy as np

from candidate.agent.belief.state import BeliefState
from candidate.agent.data.territory import TerritoryDiffs
from candidate.agent.geometry import bitboard


def rewrite_opp_land(state: BeliefState, base_new: np.ndarray, remove: np.ndarray | None = None) -> None:
    """Express the opponent's land under every hypothesis against `base_new`, first taking the cells in
    `remove` (which became mine) out of it and out of the counts. The caller sets state.opp_base afterwards."""
    store = state.store
    used = np.unique(store.opp_land_id)
    if used.size == 0:
        state.opp_diffs = TerritoryDiffs(state.keys)
        return
    land = state.opp_land(used)
    if remove is not None:
        lost = np.zeros(len(state.opp_diffs), dtype=np.int64)
        lost[used] = bitboard.popcount(land & remove[None, :])
        land = land & ~remove[None, :]
        store.opp_count = (store.opp_count - lost[store.opp_land_id]).astype(np.int32)
    fresh = TerritoryDiffs(state.keys)
    lookup = np.zeros(len(state.opp_diffs), dtype=np.int64)
    lookup[used] = fresh.intern_territory(land, base_new)
    store.opp_land_id = lookup[store.opp_land_id]
    state.opp_diffs = fresh


def rewrite_my_land(
    state: BeliefState,
    base_new: np.ndarray,
    add: np.ndarray | None = None,
    remove: np.ndarray | None = None,
    replace: np.ndarray | None = None,
) -> None:
    """Change my land under every hypothesis (give it `replace` outright, then add `add` and take away
    `remove`) and express it against `base_new`. Nothing to do, and nothing copied, when the base has not
    moved and nothing changes. The caller sets state.my_base afterwards."""
    if add is None and remove is None and replace is None and np.array_equal(base_new, state.my_base):
        return
    store = state.store
    used = np.unique(store.my_land_id)
    if used.size == 0:
        state.my_diffs = TerritoryDiffs(state.keys)
        return
    land = state.my_land(used)
    if replace is not None:
        land = np.broadcast_to(replace, land.shape).copy()
    if add is not None:
        land = land | add[None, :]
    if remove is not None:
        land = land & ~remove[None, :]
    fresh = TerritoryDiffs(state.keys)
    lookup = np.zeros(len(state.my_diffs), dtype=np.int64)
    lookup[used] = fresh.intern_territory(land, base_new)
    store.my_land_id = lookup[store.my_land_id]
    state.my_diffs = fresh
