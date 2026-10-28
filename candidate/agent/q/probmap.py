"""Q's probability map: the world map (behavior/worldmap.py, built once per frame as Snapshot.world), pooled.

The ten channels are pooled from 50 x 50 to 10 x 10 in 5 x 5 blocks while the features are built, not inside the
network (docs/Q_Probability_Map_Input.md):

    0-2  ownership: free, mine, enemy      mean
    3    ownership confidence              mean
    4-6  trail: none, my trail, enemy      mean
    7    trail confidence                  mean
    8    enemy position probability        sum
    9    my position                       sum
"""

import numpy as np

from candidate.agent.behavior.snapshot import Snapshot
from candidate.agent.behavior.worldmap import (
    ENEMY, ENEMY_POS, ENEMY_TRAIL, FREE, MINE, MY_POS, MY_TRAIL_CH, N_CHANNELS, NONE, OWN_CONF, TRAIL_CONF,
    confidence, probability_board, reach_region,
)
from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.q.config import Q, QConfig
from candidate.agent.runtime import TurnView

# The channel names and the map's pieces are re-exported, so Q's code and its tests can name them from here.
__all__ = ["ENEMY", "ENEMY_POS", "ENEMY_TRAIL", "FREE", "MINE", "MY_POS", "MY_TRAIL_CH", "N_CHANNELS", "NONE", "OWN_CONF",
           "TRAIL_CONF", "build_probability_board", "build_probmap", "confidence", "pool", "probability_board",
           "reach_region"]


def pool(board: np.ndarray, cfg: QConfig = Q) -> np.ndarray:
    """(10, 50, 50) -> (10, 10, 10): mean for the probability and confidence channels, sum for the two positions."""
    b = cfg.probmap_block
    blocks = board.reshape(N_CHANNELS, BOARD_HEIGHT // b, b, BOARD_WIDTH // b, b)
    pooled = np.empty((N_CHANNELS, BOARD_HEIGHT // b, BOARD_WIDTH // b), dtype=np.float32)
    pooled[:ENEMY_POS] = blocks[:ENEMY_POS].mean(axis=(2, 4))
    pooled[ENEMY_POS:] = blocks[ENEMY_POS:].sum(axis=(2, 4))
    return pooled


def build_probability_board(view: TurnView, snap: Snapshot, cfg: QConfig = Q) -> np.ndarray:
    """The full-board map for a frame: the snapshot's (it is built there, once)."""
    return snap.world


def build_probmap(view: TurnView, snap: Snapshot, cfg: QConfig = Q) -> np.ndarray:
    """Q's third input: (10, 10, 10) float32."""
    return pool(snap.world, cfg)
