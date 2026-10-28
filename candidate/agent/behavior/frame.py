"""Everything a state needs to produce a move, for one frame: the view, its snapshot, the hints the strategy keeps
between frames, the behaviour's config, and the game's Style when it has one (style.py; only in training games). The
exit scores are computed on first use, since only PATROL and EXPAND (and EXPAND's validity) need them."""

from dataclasses import dataclass
from functools import cached_property

from candidate.agent.behavior.config import BEHAVIOR, BehaviorConfig
from candidate.agent.behavior.exits import Exits, exit_scores
from candidate.agent.behavior.snapshot import Snapshot
from candidate.agent.behavior.style import Style
from candidate.agent.data.types import Direction
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import direction_to, neighbors
from candidate.agent.runtime import TurnView


@dataclass
class Hints:
    """What the strategy remembers between frames, because the board cannot say it: the direction an expedition set
    out in, and the side EXPAND2 turned to. Both are cleared when the trail is empty."""

    outward: Direction | None = None
    across: Direction | None = None


class Frame:
    def __init__(self, view: TurnView, snap: Snapshot, hints: Hints | None = None, cfg: BehaviorConfig = BEHAVIOR,
                 style: Style | None = None) -> None:
        self.view = view
        self.snap = snap
        self.hints = hints if hints is not None else Hints()
        self.cfg = cfg
        self.style = style

    @property
    def pos(self) -> int:
        return self.view.obs.my_pos

    @property
    def safe(self) -> tuple[Direction, ...]:
        """Moves the engine accepts that do not step onto my own trail."""
        return self.view.safe_moves

    @cached_property
    def exits(self) -> Exits:
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
            for n in neighbors(trail[0]):
                if bitboard.has_cell(self.snap.my_land_bb, n):
                    return direction_to(n, trail[0])
        return None
