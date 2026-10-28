"""Small array operations on (50, 50) maps, in plain NumPy (no SciPy)."""

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.geometry.grid import DX, DY


def box_sum(a: np.ndarray, radius: int) -> np.ndarray:
    """At each cell, the sum of `a` over the (2r+1) x (2r+1) square around it, clipped at the board edge (an
    integral image). With radius 5 and a probability map this is the mass an 11 x 11 view would cover."""
    integral = np.zeros((BOARD_HEIGHT + 1, BOARD_WIDTH + 1), dtype=np.float64)
    integral[1:, 1:] = a.astype(np.float64).cumsum(axis=0).cumsum(axis=1)
    ys, xs = np.arange(BOARD_HEIGHT), np.arange(BOARD_WIDTH)
    y0, y1 = np.maximum(ys - radius, 0), np.minimum(ys + radius + 1, BOARD_HEIGHT)
    x0, x1 = np.maximum(xs - radius, 0), np.minimum(xs + radius + 1, BOARD_WIDTH)
    return (integral[np.ix_(y1, x1)] - integral[np.ix_(y0, x1)] - integral[np.ix_(y1, x0)] + integral[np.ix_(y0, x0)])


def at_neighbor(a: np.ndarray, direction: int, fill) -> np.ndarray:
    """out[y, x] = a at the neighbour of (y, x) in `direction`; `fill` where that neighbour is off the board."""
    dy, dx = DY[direction], DX[direction]
    out = np.full(a.shape, fill, dtype=a.dtype)
    dst_y = slice(max(0, -dy), BOARD_HEIGHT - max(0, dy))
    dst_x = slice(max(0, -dx), BOARD_WIDTH - max(0, dx))
    src_y = slice(max(0, dy), BOARD_HEIGHT - max(0, -dy))
    src_x = slice(max(0, dx), BOARD_WIDTH - max(0, -dx))
    out[dst_y, dst_x] = a[src_y, src_x]
    return out
