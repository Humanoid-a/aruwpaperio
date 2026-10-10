"""Tunable constants of the belief, in one place.

The central one is the budget: how many hypotheses to keep at each position.
After every update the belief prunes each position's bucket down to its
budget; what is dropped is not forgotten but moved into that position's
residual, so the probability of the opponent being at a position is never
changed by pruning (see data.hypotheses).

Why a budget per position, not one global top-N: a global cut would keep the
many likely hypotheses at the opponent's most probable cell and throw away the
rare ones elsewhere, so the belief would forget that the opponent might be
somewhere else. Each position keeps at least `min_keep` hypotheses and at most
`max_keep`, in proportion to its probability in between.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class BudgetPolicy(ABC):
    """How many hypotheses each position may keep."""

    @abstractmethod
    def per_cell(self, mass: np.ndarray) -> np.ndarray:
        """`mass` is the (50, 50) probability of each position (hypotheses plus
        residual). Returns an int (50, 50) array: the most hypotheses to keep
        at each position. Positions with no mass keep none."""


@dataclass(frozen=True)
class ProportionalBudget(BudgetPolicy):
    """keep = ceil(total * probability), at least `min_keep` and at most `max_keep`.

    With total=2000, a position holding 40% of the belief keeps up to 800
    hypotheses; one holding 0.01% keeps 1. When the opponent is seen, all the
    probability sits on one cell, so that cell gets min(total, max_keep)."""

    total: int = 2000
    min_keep: int = 1
    max_keep: int = 1500

    def __post_init__(self) -> None:
        if not 1 <= self.min_keep <= self.max_keep:
            raise ValueError("need 1 <= min_keep <= max_keep")
        if self.total < 1:
            raise ValueError("total must be positive")

    def per_cell(self, mass: np.ndarray) -> np.ndarray:
        keep = np.clip(np.ceil(self.total * mass), self.min_keep, self.max_keep).astype(np.int64)
        keep[mass <= 0] = 0
        return keep


@dataclass(frozen=True)
class BeliefConfig:
    budget: BudgetPolicy = field(default_factory=ProportionalBudget)
    capacity: int = 8000           # hard cap on stored hypotheses; the budget normally keeps well below it
    epsilon: float = 0.02          # floor mixed into every move probability: no legal move is ever impossible
    beta: float = 1.0              # sharpness of the opponent's preference among its legal moves
    mass_floor: float = 1e-300     # total mass at or below this after the evidence means every hypothesis was contradicted.
                                   # Tiny is not zero: an opponent moving purposefully can be a 1e-22 event for a random walk
                                   # and is still a valid posterior once normalised.
    # Who moves first decides how many moves the opponent has made when I first look: none if I move first,
    # one if it does. The API does not say which seat I have, so it is inferred from the respawn side (the default
    # config gives the left respawn to the first mover) and held as a belief, not a certainty.
    seat_confidence: float = 0.95   # how sure the inference is; 0.5 would mean no idea
    first_update_stay: float | None = None   # chance it has not yet moved at the first look; None = infer from the seat
    residual_respawn_rate: float = 0.02  # share of the residual that moves to the respawn cell each turn: it may have died
    rehydrate_below: float = 0.2   # when hypotheses hold less than this share of the belief ...
    rehydrate_fraction: float = 0.5  # ... this share of the residual is turned back into hypotheses with guessed histories

    def __post_init__(self) -> None:
        if not 0 <= self.epsilon < 1:
            raise ValueError("epsilon must be in [0, 1)")
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        if self.first_update_stay is not None and not 0 <= self.first_update_stay < 1:
            raise ValueError("first_update_stay must be in [0, 1)")
        if not 0.5 <= self.seat_confidence < 1:
            raise ValueError("seat_confidence must be in [0.5, 1)")
        for name in ("residual_respawn_rate", "rehydrate_below", "rehydrate_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
