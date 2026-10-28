"""Per-game variety in how the states behave: a small, seeded randomness for training games.

Two deterministic controllers play the same game every time they meet, so a batch of training games between them
would be one game many times over. BehaviorConfig.randomness gives each game its own variation instead. It is 0 by
default and in a real match, and then nothing here runs: the states behave exactly as written.

At randomness r in (0, 1], draw() gives the game:

  * its own constants, each moved a little from the configured one (more at larger r):
      switch_cost, switch_cost_return_to_expand   switches state more or less readily
      safety_threshold, safety_floor              the hard safety rules, but only ever MORE cautious than configured:
                                                  a less cautious rule would mean avoidable deaths, not variety
      hunt_min_chance                             how likely a cut must be before HUNT goes for the enemy's trail
      return_margin                               how safe RETURN wants a path before choosing it for its gain
      exit_weight_*                               what PATROL looks for in an exit (room, safety, news, a short walk)
    The constants the snapshot reads (what counts as possible or known, the distance caps, margin_offset) are never
    varied: they decide Q's inputs, which must mean the same thing in every game.
  * a preference over the board added to PATROL's exit score (smooth, fixed for the whole game, so PATROL goes out at
    different places in different games without dithering between exits within one)
  * a jog rate: the chance per frame that EXPAND, out in the open, takes one step sideways instead of straight on,
    which varies the shape and size of its loops

Every move still goes through the strategy's guard and the safety look-ahead, so a varied game is never an unsafe one.
The same seed always gives the same variation, so a game can be played again.
"""

import random
from dataclasses import dataclass, fields, replace

import numpy as np

from candidate.agent.behavior import maps
from candidate.agent.behavior.config import BehaviorConfig
from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH

EXIT_BIAS_RADIUS = 4            # the preference map is noise smoothed over a 9 x 9 square: regions, not single cells
EXIT_BIAS_SCALE = 0.3           # at r = 1 its standard deviation is about this (exit scores are sums of 0..1 terms)
JOG_RATE_MAX = 0.08             # at r = 1 a game's jog rate is drawn from 0 .. this


@dataclass(frozen=True, eq=False)
class Style:
    """One game's variation beyond its constants. `rng` is for the draws made during the game (EXPAND's jogs)."""

    exit_bias: np.ndarray         # (50, 50) float64, added to PATROL's exit score
    jog_rate: float
    rng: random.Random
    changed: dict                 # the constants that differ from the configured ones, for the game's record


def draw(cfg: BehaviorConfig, seed: int) -> tuple[BehaviorConfig, Style | None]:
    """The constants and the Style of one game. With cfg.randomness 0: cfg itself and no Style."""
    r = float(cfg.randomness)
    if r <= 0.0:
        return cfg, None
    rng = random.Random(f"style:{seed}")

    def scaled(value: float, spread: float) -> float:
        return value * rng.uniform(1.0 - spread * r, 1.0 + spread * r)

    def shifted(value: int, low: int, high: int) -> int:
        return value + rng.randint(round(low * r), round(high * r))

    varied = replace(
        cfg,
        switch_cost=scaled(cfg.switch_cost, 0.5),
        switch_cost_return_to_expand=scaled(cfg.switch_cost_return_to_expand, 0.5),
        safety_threshold=shifted(cfg.safety_threshold, 0, 2),
        safety_floor=cfg.safety_floor + rng.uniform(0.0, 2.0 * r),
        hunt_min_chance=min(0.8, max(0.2, scaled(cfg.hunt_min_chance, 0.5))),
        return_margin=shifted(cfg.return_margin, 0, 2),               # only ever more careful
        exit_weight_room=scaled(cfg.exit_weight_room, 0.5),
        exit_weight_safety=scaled(cfg.exit_weight_safety, 0.5),
        exit_weight_info=scaled(cfg.exit_weight_info, 0.5),
        exit_weight_walk=scaled(cfg.exit_weight_walk, 0.5),
    )
    noise = np.random.default_rng(rng.randrange(2**32)).normal(size=(BOARD_HEIGHT, BOARD_WIDTH))
    smooth = maps.box_sum(noise, EXIT_BIAS_RADIUS)
    smooth /= max(float(smooth.std()), 1e-9)
    changed = {f.name: getattr(varied, f.name) for f in fields(cfg) if getattr(varied, f.name) != getattr(cfg, f.name)}
    style = Style(exit_bias=EXIT_BIAS_SCALE * r * smooth, jog_rate=rng.uniform(0.0, JOG_RATE_MAX * r),
                  rng=random.Random(rng.randrange(2**32)), changed=changed)
    return varied, style
