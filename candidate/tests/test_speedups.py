"""The simulation speed-ups give exactly what the code they replaced gave."""

import numpy as np

from candidate.agent.controller import Controller
from candidate.agent.data.territory import distinct_rows
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.geometry import bitboard
from candidate.agent.q.strategy import QStrategy
from candidate.offline import match
from candidate.offline.noisy_enemy import NoisyEnemy


def test_popcount_counts_cells_as_unpacking_the_bits_did():
    rng = np.random.default_rng(3)
    boards = rng.integers(0, 2**50, size=(200, 50), dtype=np.int64)
    unpacked = np.unpackbits(np.ascontiguousarray(boards).view(np.uint8), axis=-1).sum(axis=-1)
    assert np.array_equal(bitboard.popcount(boards), unpacked)
    assert bitboard.popcount(boards[7]) == int(unpacked[7]) and bitboard.popcount(np.zeros(50, np.int64)) == 0


def test_distinct_rows_finds_the_same_rows_as_np_unique():
    rng = np.random.default_rng(4)
    for _ in range(100):
        pool = rng.integers(0, 2**50, size=(int(rng.integers(1, 20)), 100), dtype=np.int64)
        rows = pool[rng.integers(0, len(pool), size=int(rng.integers(1, 300)))]
        unique, inverse = distinct_rows(rows)
        assert np.array_equal(unique[inverse], rows)
        assert {r.tobytes() for r in unique} == {r.tobytes() for r in np.unique(rows, axis=0)}
        assert len(unique) == len({r.tobytes() for r in rows})
    empty, inv = distinct_rows(np.zeros((0, 100), dtype=np.int64))
    assert empty.shape == (0, 100) and inv.shape == (0,)


def test_the_result_only_engine_plays_the_same_match_as_the_full_one():
    results = []
    for keep in (False, True):                       # False: _ResultOnlyEngine; True: the framework's engine
        ours = match.Taped(match.AgentAdapter(Controller(QStrategy(), Diagnostics())))
        theirs = match.Taped(NoisyEnemy(0.7, 5))
        r = match.play([ours, theirs], 150, match.CLOSE, keep=keep)
        results.append((ours.tape, theirs.tape, r.counts, r.deaths, r.kills, r.failed_turns))
    assert results[0] == results[1]
