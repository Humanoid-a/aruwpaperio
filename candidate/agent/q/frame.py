"""Everything a state needs to produce a move, for one frame."""

from dataclasses import dataclass
from functools import cached_property

from candidate.agent.data.types import Direction
from candidate.agent.geometry.grid import direction_to, neighbors
from candidate.agent.q.config import Q, QConfig
from candidate.agent.q.snapshot import Snapshot
from candidate.agent.runtime import TurnView


@dataclass
class Hints:
    """What the strategy remembers between frames, because the board cannot say it: the direction an
    expedition set out in. Cleared when the trail is empty."""

    outward: Direction | None = None


class Frame:
    def __init__(self, view: TurnView, snap: Snapshot, hints: Hints | None = None, cfg: QConfig = Q) -> None:
        self.view = view
        self.snap = snap
        self.hints = hints if hints is not None else Hints()
        self.cfg = cfg

    @property
    def pos(self) -> int:
        return self.view.obs.my_pos

    @property
    def safe(self) -> tuple[Direction, ...]:
        """Moves the engine accepts that do not step onto my own trail."""
        return self.view.safe_moves

    @cached_property
    def exits(self):
        from candidate.agent.q.exits import exit_scores

        return exit_scores(self)

    def heading(self) -> Direction | None:
        """The direction the current expedition is going in: the remembered one, else the last step of the trail
        (or, with a single trail cell, the way out of my land)."""
        if self.hints.outward is not None:
            return self.hints.outward
        trail = [int(c) for c in self.view.obs.my_trail]
        if len(trail) >= 2:
            return direction_to(trail[-2], trail[-1])
        if len(trail) == 1:
            from candidate.agent.geometry import bitboard

            for n in neighbors(trail[0]):
                if bitboard.has_cell(self.snap.my_land_bb, n):
                    return direction_to(n, trail[0])
        return None
