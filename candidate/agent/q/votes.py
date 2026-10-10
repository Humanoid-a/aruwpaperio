"""What the hypotheses say, cell by cell.

For three things the belief can be asked per cell, the probability that comes from hypotheses (the part
of the belief that still has a history) is the sum of the weights of the hypotheses that have the cell:

    enemy land    sum over distinct enemy lands of (weight of the hypotheses with that land) x [cell in land]
    enemy trail   sum over hypotheses of weight x [cell in its trail]
    my land       sum over distinct views of my land of (weight) x [cell in that view]

The residual (mass whose history was pruned) is not included here: how it votes depends on the reach region,
see confidence.py. The land sums are cheap, one term per distinct land; the trail sum touches every hypothesis
and cell, so it runs in chunks of hypotheses to keep memory small.
"""

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.belief_view import BeliefDetail
from candidate.agent.geometry import bitboard

CHUNK = 1024


def _land_votes(lands: np.ndarray, index: np.ndarray, weight: np.ndarray) -> np.ndarray:
    out = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float64)
    if lands.shape[0] == 0:
        return out
    per_land = np.bincount(index, weights=weight, minlength=lands.shape[0])
    for start in range(0, lands.shape[0], CHUNK):
        masks = bitboard.to_mask(lands[start:start + CHUNK])
        out += np.tensordot(per_land[start:start + CHUNK], masks, axes=(0, 0))
    return out


_SHIFTS = np.arange(BOARD_WIDTH, dtype=np.int64)


def trail_votes(trail_bb: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """(50, 50): the weight of the hypotheses whose trail covers each cell.

    A trail is a few cells, so almost every row-word of the (n, 50) bitboards is zero. Only the non-zero words are
    expanded into bits, in chunks of hypotheses so that memory stays small however many there are."""
    out = np.zeros(BOARD_HEIGHT * BOARD_WIDTH, dtype=np.float64)
    for start in range(0, trail_bb.shape[0], CHUNK):
        chunk, w = trail_bb[start:start + CHUNK], weight[start:start + CHUNK]
        hyp, row = np.nonzero(chunk)
        if hyp.size == 0:
            continue
        bits = (chunk[hyp, row][:, None] >> _SHIFTS) & 1
        cells = row[:, None] * BOARD_WIDTH + _SHIFTS[None, :]
        out += np.bincount(cells.ravel(), weights=(bits * w[hyp][:, None]).ravel(), minlength=out.size)
    return out.reshape(BOARD_HEIGHT, BOARD_WIDTH)


def enemy_land_votes(detail: BeliefDetail) -> np.ndarray:
    return _land_votes(detail.opp_lands, detail.opp_land_index, detail.weight)


def enemy_trail_votes(detail: BeliefDetail) -> np.ndarray:
    return trail_votes(detail.trail_bb, detail.weight)


def my_land_votes(detail: BeliefDetail) -> np.ndarray:
    return _land_votes(detail.my_lands, detail.my_land_index, detail.weight)
