"""One turn's input, already converted from framework types by the adapter.

The visible window is the (up to) 11 x 11 block of cells centred on the
agent, clipped at the board edge. Because a cell can hold territory and a
trail at once, the window comes as two layers (see data.cells) rather than
one code per cell; the `window` property gives the single-code view.

Trails follow the framework's convention: the trail includes the cell the
agent is standing on, as its last element, whenever it is non-empty. So
my_trail[-1] == my_pos while exposed, and my_trail is empty only when the
agent has just closed a trail, respawned, or never left its land.

What the framework hides, and what the adapter must fill in
    * A cell holding an agent is reported as that agent, which hides any
      trail beneath it. Cells where the trail layer is therefore a guess are
      listed in trail_hidden, and MemoryMap.update() keeps what it already
      knew about the one under my own agent. This is not rare: an agent that
      respawns onto a cell the other player's trail crosses stands on that
      trail unseen.
      For my own cell, the trail layer comes from my_trail when the cell is
      on it (certain); otherwise it is hidden and guessed FREE.
      The opponent's cell cannot be read, so the adapter has to guess. One
      direction is certain: the engine never lays a trail on its own mover's
      land, so there is no OPP_TRAIL under the opponent if the territory
      beneath it is OPP_TERRITORY. The other direction is only usual: off its
      own land the opponent normally stands on its trail, except when it has
      an empty trail on enemy land (for example after my capture enclosed
      it). The belief layer should treat that cell as uncertain.
    * The territory layer is always fully visible, agents included.
"""

from dataclasses import dataclass

import numpy as np

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT, WINDOW_SIZE
from candidate.agent.data.cells import MY_TRAIL, combine, layer_problems
from candidate.agent.data.types import MoveResult


@dataclass(frozen=True, eq=False)
class Observation:
    turn: int                           # 0-based count of this controller's completed turns
    my_pos: int                         # flat index
    respawn: int                        # flat index
    my_trail: np.ndarray                # (L,) int16 flat indices, oldest first, head last
    turns_left: int
    my_count: int
    opp_count: int
    window_territory: np.ndarray        # (h, w) int8: FREE / MY_TERRITORY / OPP_TERRITORY
    window_trail: np.ndarray            # (h, w) int8: FREE / MY_TRAIL / OPP_TRAIL
    window_origin: tuple[int, int]      # (y, x) of window[0, 0] on the board
    opp_seen_at: int | None             # flat index if the opponent is visible
    last_result: MoveResult | None      # result of my previous move; None on the first call
    trail_hidden: tuple[int, ...] = ()  # flat cells whose trail layer is a guess (an agent stands there)

    def __post_init__(self) -> None:
        problems = self.problems()
        if problems:
            raise ValueError("invalid Observation: " + "; ".join(problems))
        for array in (self.my_trail, self.window_territory, self.window_trail):
            array.flags.writeable = False

    def problems(self) -> list[str]:
        out: list[str] = []
        if self.turn < 0:
            out.append("turn must be non-negative")
        if self.turns_left < 0:
            out.append("turns_left must be non-negative")
        if self.my_count < 0 or self.opp_count < 0:
            out.append("territory counts must be non-negative")
        for name, cell in (("my_pos", self.my_pos), ("respawn", self.respawn)):
            if not 0 <= cell < CELL_COUNT:
                out.append(f"{name} {cell} is off the board")
        if not isinstance(self.my_trail, np.ndarray) or self.my_trail.dtype != np.int16 or self.my_trail.ndim != 1:
            out.append("my_trail must be a 1-D int16 array")
        elif self.my_trail.size and (self.my_trail.min() < 0 or self.my_trail.max() >= CELL_COUNT):
            out.append("my_trail has a cell off the board")
        elif self.my_trail.size and int(self.my_trail[-1]) != self.my_pos:
            out.append("the trail's last cell must be the agent's own cell (my_pos)")

        shape = getattr(self.window_territory, "shape", None)
        out += layer_problems(self.window_territory, self.window_trail, shape or (0, 0), allow_unknown=False)
        if out or shape is None or len(shape) != 2:
            return out or ["window layers must be 2-D"]

        h, w = shape
        y0, x0 = self.window_origin
        if not (0 < h <= WINDOW_SIZE and 0 < w <= WINDOW_SIZE):
            out.append(f"window is {h}x{w}, expected at most {WINDOW_SIZE}x{WINDOW_SIZE}")
        elif not (0 <= y0 and y0 + h <= BOARD_HEIGHT and 0 <= x0 and x0 + w <= BOARD_WIDTH):
            out.append("window does not lie inside the board")
        else:
            if not self._covers(self.my_pos, y0, x0, h, w):
                out.append("my_pos is outside the window")
            if self.opp_seen_at is not None:
                if not 0 <= self.opp_seen_at < CELL_COUNT or not self._covers(self.opp_seen_at, y0, x0, h, w):
                    out.append("opp_seen_at is outside the window")
                elif self.opp_seen_at == self.my_pos:
                    out.append("opp_seen_at equals my_pos")
            if not out and self._visible_own_trail(y0, x0, h, w) != self._own_trail_in_window(y0, x0, h, w):
                out.append("MY_TRAIL cells in the window do not match my_trail")
            hidden = self.trail_hidden
            if len(set(hidden)) != len(hidden) or not all(
                0 <= c < CELL_COUNT and self._covers(c, y0, x0, h, w) for c in hidden
            ):
                out.append("trail_hidden must list distinct cells inside the window")
        return out

    def _visible_own_trail(self, y0: int, x0: int, h: int, w: int) -> set[int]:
        ys, xs = np.nonzero(self.window_trail == MY_TRAIL)
        return {(y0 + int(y)) * BOARD_WIDTH + x0 + int(x) for y, x in zip(ys, xs)}

    def _own_trail_in_window(self, y0: int, x0: int, h: int, w: int) -> set[int]:
        return {int(c) for c in self.my_trail if self._covers(int(c), y0, x0, h, w)}

    @staticmethod
    def _covers(c: int, y0: int, x0: int, h: int, w: int) -> bool:
        y, x = divmod(c, BOARD_WIDTH)
        return y0 <= y < y0 + h and x0 <= x < x0 + w

    @property
    def hidden_mask(self) -> np.ndarray:
        """Boolean (h, w) array: True where the trail layer is a guess."""
        mask = np.zeros(self.window_territory.shape, dtype=bool)
        y0, x0 = self.window_origin
        for c in self.trail_hidden:
            y, x = divmod(c, BOARD_WIDTH)
            mask[y - y0, x - x0] = True
        return mask

    @property
    def window(self) -> np.ndarray:
        """The single-code view of the window (trail over territory; lossy)."""
        return combine(self.window_territory, self.window_trail)

    @property
    def window_slices(self) -> tuple[slice, slice]:
        """Slices that select this window out of a (BOARD_HEIGHT, BOARD_WIDTH) array."""
        h, w = self.window_territory.shape
        y0, x0 = self.window_origin
        return slice(y0, y0 + h), slice(x0, x0 + w)

    def covers(self, c: int) -> bool:
        """Is board cell c inside the visible window?"""
        h, w = self.window_territory.shape
        y0, x0 = self.window_origin
        return 0 <= c < CELL_COUNT and self._covers(c, y0, x0, h, w)

    def window_cell(self, c: int) -> tuple[int, int]:
        """(territory, trail) codes of a visible board cell."""
        if not self.covers(c):
            raise ValueError(f"cell {c} is outside the visible window")
        y, x = divmod(c, BOARD_WIDTH)
        y0, x0 = self.window_origin
        return int(self.window_territory[y - y0, x - x0]), int(self.window_trail[y - y0, x - x0])
