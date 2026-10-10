"""Step 4 of a frame: store the weighted children back as the belief.

    merge      identical hypotheses are folded into one (their weights add)
    normalise  everything, hypotheses and residual together, sums to 1; if nothing is left
               the observation contradicted every hypothesis and the belief is rebuilt
    prune      each position keeps its budget of best hypotheses; the rest goes to that
               position's residual, so the probability of any position is never changed
    rebase     (after memory has learned this frame's window) re-express both lands against
               the new memory, so differences for cells the window has since revealed drop out
    rehydrate  if the hypotheses hold too little of the belief, give part of the residual a
               guessed history again

Rehydration is done at the end of the frame so the next frame starts from final states.
"""

import numpy as np

from candidate.agent.belief.config import BeliefConfig
from candidate.agent.belief.generate import Children
from candidate.agent.belief.land import rewrite_my_land, rewrite_opp_land
from candidate.agent.belief.reseed import reseed
from candidate.agent.belief.state import BeliefState, BeliefStats
from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.data.belief_view import BeliefEvent
from candidate.agent.data.cells import MY_TERRITORY, OPP_TERRITORY
from candidate.agent.data.hypotheses import HypothesisStore
from candidate.agent.data.memory import MemoryMap
from candidate.agent.data.observation import Observation
from candidate.agent.geometry import bitboard


def merge(state: BeliefState, store: HypothesisStore, stats: BeliefStats) -> HypothesisStore:
    """Fold identical hypotheses together; their weights add."""
    if store.n <= 1:
        return store
    key = store.merge_key(state.opp_diffs, state.my_diffs, state.keys)
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    if first.size == store.n:
        return store
    stats.merged += store.n - first.size
    merged = store.take(first)
    merged.weight = np.bincount(inverse, weights=store.weight, minlength=first.size)
    return merged


def prune(state: BeliefState, store: HypothesisStore, cfg: BeliefConfig, stats: BeliefStats) -> HypothesisStore:
    """Keep each position's best hypotheses up to its budget; whatever is dropped goes into that
    position's residual, so the probability of each position is unchanged."""
    if store.n == 0:
        return store
    mass = store.mass_by_cell() + state.residual.mass
    budget = cfg.budget.per_cell(mass).reshape(-1)
    order = np.lexsort((-store.weight, store.pos))
    pos = store.pos[order]
    rank = np.arange(store.n) - np.searchsorted(pos, pos, side="left")
    keep = rank < budget[pos]
    if keep.sum() > cfg.capacity:
        best = np.argsort(-store.weight[order], kind="stable")[: cfg.capacity]
        keep = np.zeros(store.n, dtype=bool)
        keep[best] = True
    dropped = order[~keep]
    if dropped.size:
        moved = np.bincount(store.pos[dropped], weights=store.weight[dropped], minlength=CELL_COUNT)
        state.residual.mass = state.residual.mass + moved.reshape(BOARD_HEIGHT, BOARD_WIDTH)
        stats.pruned_mass += float(moved.sum())
    return store.take(np.sort(order[keep]))


def settle(state: BeliefState, kids: Children, cfg: BeliefConfig, stats: BeliefStats) -> bool:
    """Merge, normalise and prune the weighted children into the belief. False if the evidence left nothing:
    then the state is untouched and the caller must rebuild."""
    merged = merge(state, kids.store, stats)
    total = float(merged.weight.sum()) + state.residual.total()
    if not total > cfg.mass_floor:
        return False
    merged.weight = merged.weight / total
    state.residual.mass = state.residual.mass / total
    state.store = prune(state, merged, cfg, stats)
    return True


def rehydrate(state: BeliefState, cfg: BeliefConfig, stats: BeliefStats, count: int) -> None:
    """If almost all the belief has lost its history, give part of the residual a guessed one.

    A guess is the least the opponent could have been doing at that cell: standing on its own land with no
    trail, or one step out with a one-cell trail; its land is what memory believes, its count the one the
    engine reports, and my land as it is in the most likely hypothesis that is left. The guess is usually
    wrong, and then the next piece of evidence removes it. That costs nothing, because the rest of the
    residual keeps the position's probability."""
    if state.residual.total() <= 0 or state.store.weight.sum() >= cfg.rehydrate_below:
        return
    flat = state.residual.mass.reshape(-1)
    cells = np.flatnonzero(flat > 0)
    my_land_id = int(state.store.my_land_id[np.argmax(state.store.weight)]) if state.store.n else 0
    mass = flat[cells] * cfg.rehydrate_fraction
    state.residual.mass = state.residual.mass * (1.0 - cfg.rehydrate_fraction)
    at_home = bitboard.test_cells(state.opp_base, cells)
    out = ~at_home
    trail = np.zeros((cells.size, BOARD_HEIGHT), dtype=np.int64)
    rows, cols = np.divmod(cells[out], BOARD_WIDTH)
    trail[np.nonzero(out)[0], rows] = np.left_shift(np.int64(1), cols)
    keys = state.keys
    fresh = HypothesisStore.build(
        pos=cells,
        last_dir=np.where(out, 0, -1),
        trail_len=out.astype(np.int64),
        trail_bb=trail,
        opp_land_id=np.zeros(cells.size, dtype=np.int64),
        my_land_id=np.full(cells.size, my_land_id, dtype=np.int64),
        opp_count=np.full(cells.size, count),
        flags=np.zeros(cells.size),
        weight=mass,
        hash=keys.hash_position("opp_pos", cells) ^ np.where(out, keys.hash_position("opp_trail", cells), 0),
    )
    state.store = prune(state, merge(state, HypothesisStore.concatenate([state.store, fresh]), stats), cfg, stats)
    stats.rehydrated += int(cells.size)


def rebase(state: BeliefState, memory: MemoryMap, obs: Observation, cfg: BeliefConfig, stats: BeliefStats) -> None:
    """The end of the observation: express both lands against the memory as it is now (rebuilding the belief
    from this observation first if everything was contradicted), then rehydrate, then freeze the store."""
    opp_new = bitboard.from_mask(memory.territory == OPP_TERRITORY)
    my_new = bitboard.from_mask(memory.territory == MY_TERRITORY)
    rebuilt = state.needs_reseed
    if rebuilt:
        reseed(state, obs, opp_new)
        state.events.append(BeliefEvent.RESEEDED)
        stats.reseeds += 1
    else:
        rewrite_opp_land(state, opp_new)
        rewrite_my_land(state, my_new)
        state.store = merge(state, state.store, stats)
    state.opp_base, state.my_base = opp_new, my_new
    state.needs_reseed = False
    state.opp_count = obs.opp_count
    if not rebuilt:  # a rebuilt belief is deliberately plain until the next frame
        rehydrate(state, cfg, stats, obs.opp_count)
    state.store.freeze()
