"""Step 3 of a frame: apply the motion model K to what is left, and give every child its weight.

    weight(child) = weight(parent) x P(move | parent)

K scores the outcomes of the moves of a hypothesis, and a softmax over its legal
moves, mixed with a small floor, turns the scores into probabilities
(belief.motion.move_probabilities). The probabilities are those of the whole move
the opponent chose among *all* the moves the rules allow it, which is why K is
shown all four moves of a parent even though the evidence has already removed some
of the children: a parent whose other moves were all ruled out by the evidence is
exactly as unlikely as K says its remaining move is, not certain. Dividing by only
the survivors would erase that.

What the evidence does buy is that K is run only for the parents that still have a
child: when the opponent is in view, or the counts are very selective, that is a
handful of hypotheses instead of thousands.

The first look is special. At my first observation the opponent has made no move
if I move first and exactly one if it does. Which seat I have is not in the API; it
is inferred from the respawn side (the default config gives the first mover the
left respawn) and held as a belief, not a certainty (BeliefConfig.seat_confidence).
"""

import numpy as np

from candidate.agent.belief.config import BeliefConfig
from candidate.agent.belief.generate import Children, Expansion
from candidate.agent.belief.motion import MotionModel, move_probabilities
from candidate.agent.config import BOARD_WIDTH
from candidate.agent.rules.match import MatchRules


def first_look_stay(rules: MatchRules, cfg: BeliefConfig) -> float:
    """The chance that the opponent has not moved at all when I first look."""
    if cfg.first_update_stay is not None:
        return cfg.first_update_stay
    mine, theirs = rules.me.respawn % BOARD_WIDTH, rules.opp.respawn % BOARD_WIDTH
    if mine == theirs:
        return 0.5
    return cfg.seat_confidence if mine < theirs else 1.0 - cfg.seat_confidence


def assign_weights(
    kids: Children, exp: Expansion, motion: MotionModel, cfg: BeliefConfig, first_stay: float | None
) -> None:
    """Set the weight of every child in place. `first_stay` is the chance it has not moved yet when this is the
    first look (None on every later frame)."""
    parent_weight = exp.store.weight
    weight = kids.store.weight
    moved = np.nonzero(kids.move >= 0)[0]
    if moved.size:
        parents, inverse = np.unique(kids.parent[moved], return_inverse=True)
        batch = exp.batch.select(parents)
        probs = move_probabilities(motion.score(batch), batch.legal, cfg.beta, cfg.epsilon)
        p = probs[inverse.reshape(-1), kids.move[moved]]
        scale = 1.0 if first_stay is None else 1.0 - first_stay
        weight[moved] = parent_weight[kids.parent[moved]] * p * scale
    unmoved = np.nonzero(kids.move < 0)[0]
    if unmoved.size:
        weight[unmoved] = parent_weight[kids.parent[unmoved]] * (first_stay or 0.0)
