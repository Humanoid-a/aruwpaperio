"""Training games with the true world instead of the belief filter (offline/oracle.py): the published belief is the
truth every turn, the plan mixes the two kinds, and an oracle game is recorded and replayed as one."""

from pathlib import Path

import numpy as np

from candidate.agent.controller import Controller
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.geometry import bitboard
from candidate.agent.q import probmap
from candidate.agent.q.record import load_record
from candidate.agent.q.strategy import QStrategy
from candidate.offline import collect, match, replay
from candidate.offline.oracle import EngineTruth, OracleBelief
from candidate.runtime_check import OPPONENTS
from candidate.tests.engine_support import world_from_snapshot


def test_the_oracle_publishes_the_true_world_every_turn_and_the_map_is_exactly_it():
    seen = []

    class Watching(QStrategy):
        def choose(self, view):
            move = super().choose(view)
            if not view.retry:
                seen.append((view.obs.turn, view.belief, probmap.build_probability_board(view, self.last.snapshot)))
            return move

    truth = EngineTruth()
    watch = match.Watch(keep=True)
    ours = match.AgentAdapter(Controller(Watching(), Diagnostics(), belief=OracleBelief(truth, 1)))
    match.play([OPPONENTS["hunter"](), ours], 150, match.CLOSE, watch=watch, on_engine=truth.attach)
    assert len(seen) == 150
    for turn, belief, board in seen:
        world = world_from_snapshot(watch.snapshots[2 * turn + 1], 1)
        assert belief.n_hypotheses == 1 and belief.residual_mass == 0.0
        assert belief.probability_at(world.opp_pos) == 1.0
        assert np.array_equal(belief._trail_bb[0], bitboard.from_cells(world.opp_trail))
        assert int(belief._opp_count[0]) == world.opp_count
        ownership = np.select([world.territory == 1, world.territory == 3], [1, 2], 0)
        trail = np.select([world.trail == 2, world.trail == 4], [1, 2], 0)
        assert np.array_equal(board[probmap.FREE:probmap.ENEMY + 1].argmax(axis=0), ownership), turn
        assert np.array_equal(board[probmap.NONE:probmap.ENEMY_TRAIL + 1].argmax(axis=0), trail), turn
        assert np.allclose(board[[probmap.OWN_CONF, probmap.TRAIL_CONF]], 1.0, atol=1e-5)


def test_the_plan_mixes_belief_and_oracle_games_and_a_share_of_one_changes_nothing():
    args = (200, [("noisy", 1), ("self", 1)], 0.25, 100, 0.03, "heuristic", "", 7)
    plain = collect.plan(*args)
    assert not any(s.oracle for s in plain) and collect.plan(*args, belief_share=1.0) == plain
    mixed = collect.plan(*args, belief_share=0.2)
    share = sum(not s.oracle for s in mixed) / len(mixed)
    assert 0.1 < share < 0.3


def test_an_oracle_game_is_recorded_as_one_and_replays_to_the_same_inputs(tmp_path):
    spec = collect.GameSpec(index=0, opponent="self", seat=0, close=True, turns=80, seed=2, explore=0.1,
                            model="heuristic", out=str(tmp_path / "raw"), opponent_randomness=1.0, oracle=True)
    summary = collect.play_one(spec)
    assert summary["failed_turns"] == 0 and summary["spec"]["oracle"]
    original = Path(summary["record"])
    meta, a = load_record(original)
    assert meta["oracle"] is True
    again = replay.replay(original, tmp_path / "replayed")
    _, b = load_record(again)
    for key in ("grid_bits", "scalars", "probmap", "my_count", "opp_count", "d5", "d7"):
        assert np.array_equal(a[key], b[key]), key
