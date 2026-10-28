"""HUNT: step on the enemy's trail before the enemy gets home, choosing the move most likely to do it.

Each hypothesis of the belief (agent.data.belief_view.BeliefDetail) is one full guess at the enemy: its cell, its
trail and its land, with a probability. For a hypothesis i with a trail:

    h_i     the steps the enemy needs to reach its land (taxicab from its cell): its trail is safe once it is home
    kill    I can cut that trail after my move a if, for some cell c of the trail, 1 + d_a(c) <= h_i, where d_a is my
            shortest way from the cell a leads to, through cells off my own trail (passable_bb)

and for each of my safe moves

    chance(a) = sum of the probabilities of the hypotheses a lets me cut

over the whole belief: the residual (mass whose history is unknown, so no trail is known) adds nothing, so a belief
that knows little gives small chances. HUNT takes the move with the best chance; between equal chances, the one with
the shorter expected way to a cell it can cut. It is valid when the best chance is at least `hunt_min_chance`.

Approximations: the trail is taken as it is now (it grows while I walk, which only adds cells), and the enemy's way
home is taxicab (its real way can only be longer, so a cut judged possible may come with time to spare). The enemy's
own cell is never stepped on (a move onto it is refused; passable_bb leaves out the visible enemy).

Cost: one BFS per safe move (at most 4) and one pass over the (hypothesis, trail cell) pairs.
"""

from dataclasses import dataclass

import numpy as np

from candidate.agent.behavior import paths
from candidate.agent.behavior.frame import Frame
from candidate.agent.config import BOARD_WIDTH, CELL_COUNT
from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.distance import UNREACHABLE, taxicab_distance
from candidate.agent.geometry.grid import step

_BITS = np.arange(BOARD_WIDTH, dtype=np.int64)


@dataclass(frozen=True)
class HuntPlan:
    chance: dict            # Direction -> the probability of cutting a trail after that move
    steps: dict             # Direction -> expected steps to a cuttable cell, over the hypotheses it cuts
    best: Direction | None  # the move to take
    best_chance: float


def trail_cells(trail_bb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(hypothesis index, flat cell) for every cell of every trail, grouped by hypothesis (ascending)."""
    hyp, row = np.nonzero(trail_bb)
    if hyp.size == 0:
        return hyp, row
    bits = (trail_bb[hyp, row][:, None] >> _BITS) & 1
    pair_hyp, col = np.nonzero(bits)
    return hyp[pair_hyp], row[pair_hyp] * BOARD_WIDTH + col


def plan(frame: Frame) -> HuntPlan | None:
    """The chance of cutting a trail after each safe move, or None when the belief knows of no trail I could reach."""
    if "_hunt" in frame.__dict__:
        return frame.__dict__["_hunt"]
    frame.__dict__["_hunt"] = result = _plan(frame)
    return result


def _plan(frame: Frame) -> HuntPlan | None:
    detail = frame.view.belief.detail
    snap = frame.snap
    if detail is None or detail.n == 0 or not frame.safe:
        return None
    total = float(detail.weight.sum()) + float(detail.residual_mass)
    has = (detail.weight > 0) & detail.trail_bb.any(axis=1)
    if total <= 0 or not has.any():
        return None
    idx = np.flatnonzero(has)
    prob = detail.weight[idx] / total

    lands, which = np.unique(detail.opp_land_index[idx], return_inverse=True)       # its way home, per distinct land
    home_maps = taxicab_distance(detail.opp_lands[lands]).reshape(lands.size, CELL_COUNT)
    home = home_maps[which.reshape(-1), detail.pos[idx].astype(np.int64)].astype(np.int64)

    hyp, cell = trail_cells(detail.trail_bb[idx])
    keep = ~bitboard.to_mask(snap.enemy_cell_bb).reshape(-1)[cell]                 # never onto the enemy itself
    hyp, cell = hyp[keep], cell[keep]
    if hyp.size == 0:
        return None
    starts = np.flatnonzero(np.r_[True, hyp[1:] != hyp[:-1]])
    owners = hyp[starts]

    chance, steps = {}, {}
    for direction in frame.safe:
        nxt = step(frame.pos, direction)
        if nxt is None:
            continue
        mine = paths.distance_map(bitboard.from_cells([nxt]), snap.passable_bb).astype(np.int64)
        arrive = np.where(mine[cell] >= UNREACHABLE, UNREACHABLE, 1 + mine[cell])
        best = np.minimum.reduceat(arrive, starts)                                # my soonest arrival on each trail
        cut = best <= home[owners]
        p = float(prob[owners][cut].sum())
        chance[direction] = p
        steps[direction] = float((prob[owners][cut] * best[cut]).sum() / p) if p > 0 else float(UNREACHABLE)
    if not chance:
        return None
    best_move = max(chance, key=lambda d: (chance[d], -steps[d]))
    return HuntPlan(chance=chance, steps=steps, best=best_move if chance[best_move] > 0 else None,
                    best_chance=chance[best_move])


def upper_bound(frame: Frame) -> float:
    """At most the best chance, without a path search: a hypothesis can only be cut if the nearest cell of its trail
    is no farther from me as the crow flies (taxicab, a lower bound on any path) than the enemy is from home. Most
    frames end here, so the four searches run only when HUNT could be worth it."""
    if "_hunt_bound" in frame.__dict__:
        return frame.__dict__["_hunt_bound"]
    bound = 0.0
    detail = frame.view.belief.detail
    if detail is not None and detail.n:
        total = float(detail.weight.sum()) + float(detail.residual_mass)
        has = (detail.weight > 0) & detail.trail_bb.any(axis=1)
        if total > 0 and has.any() and float(detail.weight[has].sum()) / total >= frame.cfg.hunt_min_chance:
            idx = np.flatnonzero(has)
            lands, which = np.unique(detail.opp_land_index[idx], return_inverse=True)
            home_maps = taxicab_distance(detail.opp_lands[lands]).reshape(lands.size, CELL_COUNT)
            home = home_maps[which.reshape(-1), detail.pos[idx].astype(np.int64)].astype(np.int64)
            hyp, cell = trail_cells(detail.trail_bb[idx])
            if hyp.size:
                my_y, my_x = divmod(frame.pos, BOARD_WIDTH)
                cy, cx = np.divmod(cell, BOARD_WIDTH)
                crow = np.abs(cy - my_y) + np.abs(cx - my_x)
                starts = np.flatnonzero(np.r_[True, hyp[1:] != hyp[:-1]])
                owners = hyp[starts]
                near = np.minimum.reduceat(crow, starts) <= home[owners]
                bound = float(detail.weight[idx][owners][near].sum()) / total
    frame.__dict__["_hunt_bound"] = bound
    return bound


def valid(frame: Frame) -> bool:
    if upper_bound(frame) < frame.cfg.hunt_min_chance:
        return False
    result = plan(frame)
    return result is not None and result.best is not None and result.best_chance >= frame.cfg.hunt_min_chance


def move(frame: Frame) -> Direction | None:
    if not valid(frame):
        return None
    return plan(frame).best
