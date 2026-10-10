"""MemoryMap: the best knowledge of the board from past observations.

Same two layers as WorldState (see data.cells), with one extra state:

    territory   (50, 50) int8, [y][x]: UNKNOWN / FREE / MY_TERRITORY / OPP_TERRITORY
    trail       (50, 50) int8, [y][x]: UNKNOWN / FREE / MY_TRAIL / OPP_TRAIL
    last_seen   (50, 50) int32: the turn each cell was last in the window, -1 if never

A cell is UNKNOWN in both layers exactly when last_seen is -1.

Only the current window is certain. Everything remembered from earlier turns
is a prior: the opponent may have changed it since, and so may deaths and
captures that happened out of view. update() never guesses; it only copies
what the latest window shows. Reasoning about stale cells is the caller's job,
and last_seen / age() say how stale each one is.
"""

from dataclasses import dataclass

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.cells import (
    FREE,
    MY_TERRITORY,
    MY_TRAIL,
    OPP_TERRITORY,
    OPP_TRAIL,
    UNKNOWN,
    LayerBitboards,
    combine,
    layer_bitboards,
    layer_problems,
)
from candidate.agent.data.observation import Observation
from candidate.agent.geometry import bitboard

_SHAPE = (BOARD_HEIGHT, BOARD_WIDTH)
NEVER_SEEN = -1


@dataclass(eq=False)
class MemoryMap:
    territory: np.ndarray
    trail: np.ndarray
    last_seen: np.ndarray

    @classmethod
    def new(cls) -> "MemoryMap":
        """Nothing observed yet."""
        return cls(
            np.full(_SHAPE, UNKNOWN, dtype=np.int8),
            np.full(_SHAPE, UNKNOWN, dtype=np.int8),
            np.full(_SHAPE, NEVER_SEEN, dtype=np.int32),
        )

    def copy(self) -> "MemoryMap":
        return MemoryMap(self.territory.copy(), self.trail.copy(), self.last_seen.copy())

    @property
    def cells(self) -> np.ndarray:
        """The single-code view (trail over territory). Lossy; for display."""
        return combine(self.territory, self.trail)

    def known_mask(self) -> np.ndarray:
        """Boolean (50, 50): cells observed at least once."""
        return self.last_seen != NEVER_SEEN

    def known_bitboard(self) -> np.ndarray:
        return bitboard.from_mask(self.known_mask())

    def bitboards(self) -> LayerBitboards:
        """The four layers as single-board bitboards. Unknown cells are in none."""
        return layer_bitboards(self.territory, self.trail)

    def age(self, turn: int) -> np.ndarray:
        """Turns since each cell was last seen, as int32 (50, 50); -1 for
        cells never seen."""
        return np.where(self.known_mask(), turn - self.last_seen, NEVER_SEEN).astype(np.int32)

    def update(self, obs: Observation) -> None:
        """Overwrite the cells in the observation's window and stamp them with
        its turn. Cells outside the window are left as they were."""
        rows, cols = obs.window_slices
        # The trail under an agent is hidden from view. Under my own agent nothing I did can
        # have changed it, so keep what was known (it may be the opponent's trail, after a
        # respawn). Under the opponent the observation's guess is better than old memory,
        # because it lays a trail wherever it steps off its own land.
        kept = [
            (c, self.trail.flat[c])
            for c in obs.trail_hidden
            if c == obs.my_pos and self.trail.flat[c] != UNKNOWN
        ]
        self.territory[rows, cols] = obs.window_territory
        self.trail[rows, cols] = obs.window_trail
        self.last_seen[rows, cols] = obs.turn
        for c, value in kept:
            self.trail.flat[c] = value

    def agrees_with(self, obs: Observation) -> bool:
        """Do the remembered cells inside the window equal the observation?
        True immediately after update(obs). The trail layer is not compared
        where the observation says it is hidden."""
        rows, cols = obs.window_slices
        visible_trail = ~obs.hidden_mask
        return bool(
            np.array_equal(self.territory[rows, cols], obs.window_territory)
            and np.array_equal(self.trail[rows, cols][visible_trail], obs.window_trail[visible_trail])
            and (self.last_seen[rows, cols] == obs.turn).all()
        )

    # ---- inferred updates ---------------------------------------------
    # update() copies only what the window shows. These apply what the rules
    # guarantee happened to cells that may be out of view. Inferred cells are
    # stamped with the turn of the inference.

    def my_territory_count(self) -> int:
        """How many cells remembered as mine."""
        return int((self.territory == MY_TERRITORY).sum())

    def _stamp(self, mask: np.ndarray, turn: int) -> None:
        self.trail[mask & (self.trail == UNKNOWN)] = FREE
        self.territory[mask & (self.territory == UNKNOWN)] = FREE
        self.last_seen[mask] = turn

    def _reset_land(self, territory_code: int, trail_code: int, start_cells, turn: int) -> None:
        touched = (self.territory == territory_code) | (self.trail == trail_code)
        self.territory[self.territory == territory_code] = FREE
        self.trail[self.trail == trail_code] = FREE
        start = np.zeros(_SHAPE, dtype=bool)
        for c in start_cells:
            start.flat[c] = True
        self._stamp(touched | start, turn)
        self.territory[start] = territory_code

    def reset_my_land(self, start_cells, turn: int) -> None:
        """I died: my trail and all my territory are gone and my starting land
        is mine again."""
        self._reset_land(MY_TERRITORY, MY_TRAIL, start_cells, turn)

    def reset_opponent_land(self, start_cells, turn: int) -> None:
        """The opponent died: their trail and territory are gone and their
        starting land is theirs again. `start_cells` is an assumption unless
        their respawn is known."""
        self._reset_land(OPP_TERRITORY, OPP_TRAIL, start_cells, turn)

    def apply_capture(self, trail_cells, region: np.ndarray, turn: int) -> None:
        """My trail closed: the trail cells and the newly enclosed region
        (a single bitboard) are now my territory, and my trail is gone."""
        mask = bitboard.to_mask(region)
        for c in trail_cells:
            mask.flat[c] = True
        self._stamp(mask, turn)
        self.territory[mask] = MY_TERRITORY
        self.trail[self.trail == MY_TRAIL] = FREE

    def problems(self) -> list[str]:
        out = layer_problems(self.territory, self.trail, _SHAPE, allow_unknown=True)
        if not isinstance(self.last_seen, np.ndarray) or self.last_seen.dtype != np.int32:
            out.append("last_seen must be an int32 array")
        elif self.last_seen.shape != _SHAPE:
            out.append(f"last_seen must have shape {_SHAPE}, got {self.last_seen.shape}")
        elif not out:
            never = self.last_seen == NEVER_SEEN
            if ((self.territory == UNKNOWN) != never).any() or ((self.trail == UNKNOWN) != never).any():
                out.append("a cell is UNKNOWN exactly when last_seen is -1, and this one is not")
            if (self.last_seen < NEVER_SEEN).any():
                out.append("last_seen has a value below -1")
        return out

    def validate(self) -> None:
        problems = self.problems()
        if problems:
            raise ValueError("invalid MemoryMap: " + "; ".join(problems))
