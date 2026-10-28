"""Training games in many situations (offline/scenarios.py) and the paired gate (offline/online.py)."""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from candidate.offline import collect, scenarios

pytest.importorskip("torch")

from candidate.offline import online  # noqa: E402


def test_every_spawn_is_mirrored_with_player_0_on_the_left_and_the_start_lands_apart():
    rng = random.Random(0)
    for kind in ("very_close", "random"):
        for _ in range(300):
            (y0, x0), (y1, x1) = scenarios.spawn(kind, rng)
            assert y0 == y1 and x1 == 49 - x0 and x0 < x1                          # the agent's mirror assumption
            assert 3 <= y0 <= 46 and x0 >= 3 and x1 - x0 >= 5                       # 3 x 3 start lands never touch
            if kind == "very_close":
                assert x1 - x0 <= 11
    assert scenarios.spawn("far", rng) is None and scenarios.spawn("close", rng) == scenarios.CLOSE
    with pytest.raises(ValueError):
        scenarios.spawn("nowhere", rng)


def test_a_scenario_is_drawn_from_the_games_seed_with_the_mix_weights():
    spec = collect.GameSpec(index=0, opponent="noisy", seat=0, close=False, turns=2000, seed=1, explore=0.0,
                            model="heuristic", out="")
    spawns = scenarios.parse_weights("far=2,close=1,very_close=1,random=2")
    turns = scenarios.parse_weights("2000=2,1000=1,500=1,300=1", int)
    agents = scenarios.DEFAULT_WARMUP_AGENTS.split(",")
    drawn = [scenarios.with_scenario(collect.GameSpec(**{**spec.__dict__, "seed": s}), spawns, turns, 0.3, agents)
             for s in range(3000)]
    assert drawn[7] == scenarios.with_scenario(collect.GameSpec(**{**spec.__dict__, "seed": 7}), spawns, turns, 0.3, agents)
    kinds = Counter(d.spawn for d in drawn)
    assert abs(kinds["far"] / 3000 - 2 / 6) < 0.04 and abs(kinds["very_close"] / 3000 - 1 / 6) < 0.04
    assert abs(Counter(d.turns for d in drawn)[2000] / 3000 - 0.4) < 0.04
    warm = [d for d in drawn if d.warmup_turns]
    assert abs(len(warm) / 3000 - 0.3) < 0.04 and {d.warmup_agent for d in warm} == set(agents)
    assert all(0.1 * d.turns - 1 <= d.warmup_turns <= 0.5 * d.turns for d in warm)
    assert all((d.respawns is None) == (d.spawn == "far") and d.close == (d.spawn == "close") for d in drawn)


@pytest.mark.parametrize("agent", scenarios.WARMUP_CHOICES)
def test_a_game_taken_over_midway_records_q_only_from_the_take_over(agent, tmp_path):
    spec = collect.GameSpec(index=0, opponent="noisy", seat=1, close=False, turns=80, seed=3, explore=0.0,
                            model="heuristic", out=str(tmp_path), opponent_randomness=0.5,
                            respawns=((20, 20), (20, 29)), spawn="very_close", warmup_turns=30, warmup_agent=agent)
    summary = collect.play_one(spec)
    from candidate.agent.q.record import load_record

    meta, arrays = load_record(summary["record"])
    sidecar = json.loads(Path(summary["record"]).with_suffix(".json").read_text())
    assert summary["failed_turns"] == 0 and int(arrays["turn"][0]) == 30 and len(arrays["turn"]) == 50
    assert meta["warmup_turns"] == 30 and sidecar["config"]["respawns"] == [[20, 20], [20, 29]]
    assert len(sidecar["tapes"][1]) == 80 and "-" not in sidecar["tapes"][1]


def test_the_dataset_leaves_out_turns_another_agent_played():
    from candidate.offline import dataset as ds

    n = 40
    arrays = dict(turn=np.arange(n), my_count=np.arange(n), opp_count=np.zeros(n), d5=np.zeros(n),
                  d7=np.ones(n, dtype=np.float32), state=np.zeros(n, dtype=np.int8), explored=np.zeros(n, dtype=bool),
                  grid_bits=np.zeros((n, 386), dtype=np.uint8), scalars=np.zeros((n, 23), dtype=np.float32),
                  probmap=np.zeros((n, 10, 10, 10), dtype=np.float16))
    game = ds.Game(path=Path("g.npz"), meta=dict(game_id="g", warmup_turns=15), arrays=arrays,
                   sidecar=dict(config=dict(seat=0), final=dict(counts=[n, 0])))
    data = ds.build([game], window=5)
    assert len(data) == n - 15


def test_the_gate_wants_a_paired_advantage_beyond_the_noise():
    player = dict(mean_diff=100.0, diffs=[100.0, 300.0, -100.0, 100.0])
    clear = dict(mean_diff=200.0, diffs=[210.0, 390.0, 0.0, 200.0])           # better in every game, by about 100
    noisy = dict(mean_diff=110.0, diffs=[600.0, -200.0, -300.0, 340.0])       # a little better on average, by luck
    accepted, paired = online.gate_decision(player, clear, 0.0, 1.0)
    assert accepted and paired["better"] == 4 and paired["mean"] == pytest.approx(100.0)
    assert not online.gate_decision(player, noisy, 0.0, 1.0)[0]
    assert not online.gate_decision(player, clear, 150.0, 1.0)[0]             # the margin still applies
    assert online.gate_decision(dict(mean_diff=1.0), dict(mean_diff=2.0), 0.0, 1.0) == (True, None)
