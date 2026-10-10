"""Cell codes and the two-layer cell model.

Dense boards are int8 arrays indexed [y][x] and use these codes:

    -1 UNKNOWN        never observed (MemoryMap only)
     0 FREE           nothing there
     1 MY_TERRITORY
     2 MY_TRAIL
     3 OPP_TERRITORY
     4 OPP_TRAIL

A cell can hold a territory owner and a *different* player's trail at once
(the engine's BoardCell allows it: a trail crosses free space or enemy land
until it closes). One code per cell cannot say that, so every board-like
structure keeps two arrays instead:

    territory  holds FREE, MY_TERRITORY, or OPP_TERRITORY
    trail      holds FREE, MY_TRAIL, or OPP_TRAIL

The possible overlaps are my trail over opponent territory and opponent trail
over my territory (or over free land). The engine never puts a player's trail
on that same player's territory. combine() collapses the layers into the
single-code view with the trail on top, which is lossy: the territory under a
trail is not recoverable from it. Use it for display, not for state.
"""

from typing import NamedTuple

import numpy as np

from candidate.agent.geometry import bitboard

UNKNOWN = -1
FREE = 0
MY_TERRITORY = 1
MY_TRAIL = 2
OPP_TERRITORY = 3
OPP_TRAIL = 4

TERRITORY_CODES = frozenset({FREE, MY_TERRITORY, OPP_TERRITORY})
TRAIL_CODES = frozenset({FREE, MY_TRAIL, OPP_TRAIL})


def combine(territory: np.ndarray, trail: np.ndarray) -> np.ndarray:
    """The single-code view: a trail hides the territory beneath it, and a
    cell that is UNKNOWN in the trail layer stays UNKNOWN."""
    return np.where(trail != FREE, trail, territory).astype(np.int8)


class LayerBitboards(NamedTuple):
    """The four known-state layers as single-board bitboards. UNKNOWN cells
    appear in none of them."""

    my_territory: np.ndarray
    opp_territory: np.ndarray
    my_trail: np.ndarray
    opp_trail: np.ndarray


def layer_bitboards(territory: np.ndarray, trail: np.ndarray) -> LayerBitboards:
    return LayerBitboards(
        bitboard.from_mask(territory == MY_TERRITORY),
        bitboard.from_mask(territory == OPP_TERRITORY),
        bitboard.from_mask(trail == MY_TRAIL),
        bitboard.from_mask(trail == OPP_TRAIL),
    )


def layer_problems(
    territory: np.ndarray, trail: np.ndarray, shape: tuple[int, int], allow_unknown: bool
) -> list[str]:
    """Structural problems with a pair of layers: wrong type, wrong shape, or
    codes that do not belong in that layer. Empty when the pair is sound."""
    problems: list[str] = []
    for name, layer, codes in (("territory", territory, TERRITORY_CODES), ("trail", trail, TRAIL_CODES)):
        if not isinstance(layer, np.ndarray) or layer.dtype != np.int8:
            problems.append(f"{name} must be an int8 array")
            continue
        if layer.shape != shape:
            problems.append(f"{name} must have shape {shape}, got {layer.shape}")
            continue
        allowed = codes | {UNKNOWN} if allow_unknown else codes
        bad = sorted(set(np.unique(layer).tolist()) - allowed)
        if bad:
            problems.append(f"{name} has invalid codes {bad}")
    return problems
