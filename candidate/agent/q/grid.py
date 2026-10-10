"""Q's egocentric grid: 7 channels, 21 x 21, centred on me.

The crop is larger than the 11 x 11 view so that my own trail fits. Cells off the board are 0 in every channel.

    0 on the board                       4 enemy territory, visible cells
    1 visible now (the central window)   5 enemy trail, visible cells
    2 my territory, visible cells        6 enemy position, if visible
    3 my trail, the entire trail as far as the crop reaches (I always know it)

Stored as uint8 0/1 (it is logged as bytes). rotate() applies a rotation/flip to a grid or to any array whose
last two axes are the board, which is what augmentation needs: the same transform for the grid and the
confidence map.
"""

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.cells import MY_TERRITORY, OPP_TERRITORY, OPP_TRAIL
from candidate.agent.data.observation import Observation
from candidate.agent.q.config import Q, QConfig

N_CHANNELS = 7


def build_grid(obs: Observation, cfg: QConfig = Q) -> np.ndarray:
    size = cfg.grid_crop
    half = size // 2
    my_y, my_x = divmod(obs.my_pos, BOARD_WIDTH)
    ys = (my_y - half + np.arange(size))[:, None]
    xs = (my_x - half + np.arange(size))[None, :]
    on_board = (ys >= 0) & (ys < BOARD_HEIGHT) & (xs >= 0) & (xs < BOARD_WIDTH)

    grid = np.zeros((N_CHANNELS, size, size), dtype=np.uint8)
    grid[0] = on_board

    wy, wx = obs.window_origin
    h, w = obs.window_territory.shape
    visible = on_board & (ys >= wy) & (ys < wy + h) & (xs >= wx) & (xs < wx + w)
    grid[1] = visible
    iy = np.broadcast_to(ys - wy, visible.shape)[visible]
    ix = np.broadcast_to(xs - wx, visible.shape)[visible]
    territory = obs.window_territory[iy, ix]
    trail = obs.window_trail[iy, ix]
    for channel, mask in ((2, territory == MY_TERRITORY), (4, territory == OPP_TERRITORY), (5, trail == OPP_TRAIL)):
        layer = np.zeros((size, size), dtype=np.uint8)
        layer[visible] = mask
        grid[channel] = layer

    for cell in obs.my_trail:
        y, x = divmod(int(cell), BOARD_WIDTH)
        i, j = y - (my_y - half), x - (my_x - half)
        if 0 <= i < size and 0 <= j < size:
            grid[3, i, j] = 1
    if obs.opp_seen_at is not None:
        y, x = divmod(obs.opp_seen_at, BOARD_WIDTH)
        grid[6, y - (my_y - half), x - (my_x - half)] = 1
    return grid


def rotate(array: np.ndarray, quarter_turns: int = 0, flip: bool = False) -> np.ndarray:
    """Rotate by quarter turns (and optionally mirror left-right first) the last two axes of an array.
    Applying the same call to a grid and to a confidence board keeps them consistent."""
    out = array[..., ::-1] if flip else array
    return np.ascontiguousarray(np.rot90(out, k=quarter_turns, axes=(-2, -1)))
