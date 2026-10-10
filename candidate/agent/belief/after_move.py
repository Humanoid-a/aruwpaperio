"""Step 5 of a frame: apply my own move, at the end of the frame, right after I made it.

The stored hypotheses describe the world as it stands after my latest move and
before the opponent's reply, so as soon as the engine has accepted my move it has
to be applied to every one of them. What my move can change under a hypothesis:

    I closed a loop     the cells I enclosed (my trail and the land inside) leave the opponent's land
                        and its count, and become mine
    I died              my land is gone but my start land, which my respawn takes back from the
                        opponent if it held any of it
    I cut its trail     (stepped onto it) it died: its state is now certain, so every hypothesis is
                        replaced by the one in which it stands on its respawn on its start land with no
                        trail; its respawn takes back its start land, out of mine if I held any of it
    an ordinary step    nothing: no hypothesis is touched or copied

It needs only what I did, not an observation, which is why it can be the last thing in the
frame. It leaves me (MyState) where my move put me, with the engine's count for me, which
is what the next observation's opponent move is made in and checked against.
"""

from dataclasses import dataclass

import numpy as np

from candidate.agent.belief.land import rewrite_my_land, rewrite_opp_land
from candidate.agent.belief.state import BeliefState, BeliefStats, MyState
from candidate.agent.data.belief_view import BeliefEvent
from candidate.agent.data.residual import ResidualGrid
from candidate.agent.data.types import MoveResult
from candidate.agent.geometry import bitboard
from candidate.agent.prediction import Prediction


@dataclass(frozen=True, eq=False)
class MyMove:
    """My move, once the engine has accepted it."""

    result: MoveResult               # what the engine did: NORMAL, CAPTURED or DIED (never INVALID)
    prediction: Prediction           # what I expected it to do, which also says where it leaves me
    trail_before: tuple[int, ...]    # my trail before the move
    count_before: int                # the engine's count of my cells before the move


def apply_my_move(state: BeliefState, move: MyMove, stats: BeliefStats) -> None:
    rules, pred, result = state.rules, move.prediction, move.result
    count_after: int | None = move.count_before
    if result == MoveResult.CAPTURED:
        flip = bitboard.from_cells(move.trail_before)
        if pred.region is not None:
            flip = flip | pred.region
        rewrite_opp_land(state, state.opp_base, remove=flip)
        rewrite_my_land(state, state.my_base, add=flip)
        count_after = move.count_before + pred.gain if pred.result == MoveResult.CAPTURED and pred.gain is not None else None
    elif result == MoveResult.DIED:
        rewrite_opp_land(state, state.opp_base, remove=rules.me.start_bb)
        rewrite_my_land(state, state.my_base, replace=rules.me.start_bb)
        count_after = len(rules.me.start_cells)
    if pred.killed and result in (MoveResult.NORMAL, MoveResult.CAPTURED):
        lost = collapse_after_kill(state, pred.new_pos)
        count_after = None if count_after is None else count_after - lost
        state.pending.append(BeliefEvent.OPPONENT_RESET)
        stats.resets += 1
    state.me = MyState(pred.new_pos, bitboard.from_cells(pred.new_trail), count_after, result)


def collapse_after_kill(state: BeliefState, my_pos_after: int) -> int:
    """I stepped on its trail, so it died: replace every hypothesis by the one certain state. Returns how many
    cells of mine its respawn took back as its start land.

    Which view of my land to carry on with is a guess (the most likely hypothesis'): the opponent's state is
    certain now, but how much of my land it took earlier may not be."""
    role = state.rules.opp
    respawn = role.alt_respawn if my_pos_after == role.respawn else role.respawn
    my_land_id = int(state.store.my_land_id[np.argmax(state.store.weight)]) if state.store.n else 0
    land = state.my_land([my_land_id])[0]
    lost = int(bitboard.popcount(land & role.start_bb))
    my_land_id = int(state.my_diffs.intern_territory(land & ~role.start_bb, state.my_base)[0])
    state.residual = ResidualGrid()
    state.store = state.single(respawn, role.start_bb, bitboard.empty(), len(role.start_cells), my_land_id)
    return lost
