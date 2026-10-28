"""Turning the state into the read-only BeliefView a strategy sees."""


import numpy as np

from candidate.agent.belief.state import BeliefState, BeliefStats
from candidate.agent.data.belief_view import BeliefDetail, BeliefEvent, BeliefSummary, BeliefView
from candidate.agent.data.observation import Observation
from candidate.agent.geometry import bitboard


def summarise(state: BeliefState, obs: Observation, stats: BeliefStats, mark: BeliefStats) -> BeliefSummary:
    """What the belief holds, and (from the statistics since `mark`) what the frame did to it."""
    s = state.store
    total = float(s.weight.sum())
    if s.n and total > 0:
        share = s.weight / total
        exposed = float(share[s.trail_len > 0].sum())
        counts = tuple(sorted(set(s.opp_count.tolist())))
        opp_cells = bitboard.popcount(state.opp_land(s.opp_land_id))
        known = float(share[opp_cells == s.opp_count].sum())
        my_cells = bitboard.popcount(state.my_land(s.my_land_id))
        exact = float(share[my_cells == obs.my_count].sum())
        mean_trail, longest = float((share * s.trail_len).sum()), int(s.trail_len.max())
        views = int(np.unique(s.my_land_id).size)
    else:
        exposed, counts, known, exact, mean_trail, longest, views = None, (), 0.0, 0.0, 0.0, 0, 0
    delta = lambda name: getattr(stats, name) - getattr(mark, name)  # noqa: E731
    return BeliefSummary(
        opp_counts=counts, exposed_share=exposed, trail_mean=mean_trail, trail_longest=longest,
        opp_land_known=known, my_views=views, my_views_exact=exact, my_count=obs.my_count,
        opp_land_diffs=len(state.opp_diffs), my_land_diffs=len(state.my_diffs),
        successors=delta("children"), cut_by_position=delta("cut_by_position"), cut_by_count=delta("cut_by_count"),
        cut_by_window=delta("cut_by_window"), cut_by_my_count=delta("cut_by_my_count"), merged=delta("merged"),
        pruned_mass=delta("pruned_mass"), rehydrated=delta("rehydrated"), resets=delta("resets"),
    )


def detail_of(state: BeliefState) -> BeliefDetail:
    """The hypotheses and residual as arrays that stay valid after the next update."""
    s = state.store
    opp_ids, opp_index = np.unique(s.opp_land_id, return_inverse=True)
    my_ids, my_index = np.unique(s.my_land_id, return_inverse=True)
    detail = BeliefDetail(
        weight=s.weight, pos=s.pos, trail_bb=s.trail_bb,
        opp_land_index=opp_index.reshape(-1).astype(np.int64), opp_lands=state.opp_land(opp_ids) if s.n else state.opp_land(np.zeros(0, dtype=np.int64)),
        my_land_index=my_index.reshape(-1).astype(np.int64), my_lands=state.my_land(my_ids) if s.n else state.my_land(np.zeros(0, dtype=np.int64)),
        residual=state.residual.mass.copy(),
    )
    for array in (detail.opp_land_index, detail.opp_lands, detail.my_land_index, detail.my_lands, detail.residual):
        array.flags.writeable = False
    return detail


def publish(
    state: BeliefState, obs: Observation, events: tuple[BeliefEvent, ...], stats: BeliefStats, mark: BeliefStats
) -> BeliefView:
    mass = state.store.mass_by_cell() + state.residual.mass
    total = float(mass.sum())
    position = mass / total if total > 0 else mass
    p = position[position > 0]
    entropy = float(-(p * np.log(p)).sum())
    hypothesis_mass = float(state.store.weight.sum()) / total if total > 0 else 0.0
    store = state.store
    return BeliefView(
        turn=obs.turn,
        position=position.astype(np.float32),
        support=position > 0,
        n_hypotheses=store.n,
        hypothesis_mass=hypothesis_mass,
        residual_mass=1.0 - hypothesis_mass if total > 0 else 0.0,
        entropy=entropy,
        sighted_at=obs.opp_seen_at,
        events=events,
        _pos=store.pos,
        _weight=store.weight,
        _trail_bb=store.trail_bb,
        _trail_len=store.trail_len,
        _opp_count=store.opp_count,
        summary=summarise(state, obs, stats, mark),
        detail=detail_of(state),
    )
