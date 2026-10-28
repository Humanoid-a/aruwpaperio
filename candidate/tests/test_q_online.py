"""The online training loop (offline/online.py) at a tiny scale: it goes through, records every epoch, and resumes."""

import json

import pytest

pytest.importorskip("torch")

from candidate.offline import collect, online  # noqa: E402


def test_the_noisy_and_self_opponents_play_and_say_what_they_drew(tmp_path):
    for name in ("noisy", "self"):
        spec = collect.GameSpec(index=0, opponent=name, seat=1, close=True, turns=30, seed=4, explore=0.0,
                                model="heuristic", out=str(tmp_path), record=False, opponent_randomness=1.0)
        summary = collect.play_one(spec)
        assert summary["failed_turns"] == 0 and summary["opponent_draw"]["randomness"] == 1.0


def test_the_enemy_copy_is_the_original_enemy_with_no_randomness(tmp_path):
    spec = collect.GameSpec(index=0, opponent="enemy_copy", seat=1, close=True, turns=30, seed=4, explore=0.0,
                            model="heuristic", out=str(tmp_path), record=False, opponent_randomness=1.0)
    enemy, drawn = collect.make_opponent(spec)
    assert drawn == dict(randomness=0.0, bite=7, lookahead=2, vertical=1.0, aim=0.0, strike=0.0, wander=0.0)
    assert collect.play_one(spec)["failed_turns"] == 0
    assert "enemy_copy" in online.DEFAULT_OPPONENTS


def test_a_self_opponent_can_get_the_true_world_while_our_agent_keeps_the_belief(tmp_path, monkeypatch):
    from candidate.offline.oracle import EngineTruth, OracleBelief

    spec = collect.GameSpec(index=0, opponent="self", seat=0, close=True, turns=40, seed=4, explore=0.0,
                            model="heuristic", out=str(tmp_path), record=False, opponent_randomness=1.0,
                            opponent_oracle=True)
    rival, _ = collect.make_opponent(spec, EngineTruth())
    assert isinstance(rival.agent.runtime.belief, OracleBelief)
    given, make_belief = {}, collect.make_belief

    def watched(s, truth, seat):
        given[seat] = truth is not None
        return make_belief(s, truth, seat)

    monkeypatch.setattr(collect, "make_belief", watched)
    summary = collect.play_one(spec)
    assert given == {0: False, 1: True}                  # our agent (seat 0) runs the belief, the opponent the truth
    assert summary["failed_turns"] == 0 and not summary["spec"]["oracle"] and summary["spec"]["opponent_oracle"]

    # the online loop draws it for a share of the belief games against our agent only, from each game's seed
    specs = collect.plan(400, [("self", 1), ("noisy", 1)], 0.25, 40, 0.0, "heuristic", "", 3, belief_share=0.5)
    drawn = [online._with_randomness(s, 0.5, 0.5, 0.5) for s in specs]
    marked = [s for s in drawn if s.opponent_oracle]
    assert all(s.opponent == "self" and not s.oracle for s in marked)
    belief_self = [s for s in drawn if s.opponent == "self" and not s.oracle]
    assert 0.35 < len(marked) / len(belief_self) < 0.65
    assert drawn == [online._with_randomness(s, 0.5, 0.5, 0.5) for s in specs]
    assert not any(online._with_randomness(s, 0.5, 0.5).opponent_oracle for s in specs)


def test_the_costliest_games_start_first_and_the_results_keep_the_plans_order(monkeypatch):
    started = []

    class InOrder:                                      # a stand-in for the process pool: runs in the calling process
        def __init__(self, max_workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, specs):
            started.extend(s.index for s in specs)
            return [dict(spec=dict(index=s.index), label=str(s.index), my_count=0, opp_count=0, won=False, my_deaths=0,
                         my_kills=0, failed_turns=0, stats={}, seconds=0.0) for s in specs]

    monkeypatch.setattr(collect, "ProcessPoolExecutor", InOrder)
    base = dict(seat=0, close=False, turns=10, seed=1, explore=0.0, model="heuristic", out="")
    specs = [collect.GameSpec(index=0, opponent="noisy", oracle=True, **base),
             collect.GameSpec(index=1, opponent="self", **base),
             collect.GameSpec(index=2, opponent="noisy", **base),
             collect.GameSpec(index=3, opponent="self", opponent_oracle=True, **base),
             collect.GameSpec(index=4, opponent="self", oracle=True, **base)]
    results = collect.run(specs, workers=4, echo=lambda line: None)
    assert started == [1, 3, 2, 4, 0]                    # two belief filters, then one (self costs a little more), then none
    assert [r["spec"]["index"] for r in results] == [0, 1, 2, 3, 4]


def test_two_epochs_train_record_and_resume(tmp_path):
    kwargs = dict(epochs=2, games=3, workers=1, turns=40, opponents="noisy=1,self=1", enemy_randomness=1.0,
                  self_randomness=1.0, explore=0.2, close_share=0.5, train_epochs=1, patience=1, root=tmp_path,
                  quiet=True, gate_games=2)
    first = online.run("t", **kwargs)
    run = tmp_path / "t"
    assert [s["epoch"] for s in first] == [0, 1]
    for epoch in (0, 1):
        here = online.epoch_dir(run, epoch)
        assert (here / "weights.pt").exists() and (here / "summary.json").exists() and (here / "gate.json").exists()
        assert len(json.loads((here / "games.json").read_text())) == first[epoch]["games"]["all"]["games"]
        assert list((here / "raw").glob("*.npz"))
        assert first[epoch]["train"]["branches"] == ["grid", "scalars", "probmap"]         # every branch, every time
    assert first[0]["played_with"] == "heuristic"
    learner0 = str(online.epoch_dir(run, 0) / "weights.pt")
    assert first[0]["state"]["learner"] == learner0
    assert first[1]["played_with"] == (learner0 if first[0]["gate"]["accepted"] else "heuristic")
    assert first[1]["train"]["start_val"] is not None                                    # epoch 1 fine-tunes the learner
    assert first[1]["train"]["val"] <= first[1]["train"]["start_val"] + 1e-6            # never worse than it started
    assert (run / "latest.pt").exists() and len((run / "history.jsonl").read_text().splitlines()) == 2
    assert (run / "best.pt").exists() == any(s["gate"]["accepted"] for s in first)

    again = online.run("t", **kwargs)                                                    # nothing left to do
    assert json.dumps(again) == json.dumps(first)                                       # (nan == nan as text)
    assert len((run / "history.jsonl").read_text().splitlines()) == 2

    with pytest.raises(ValueError):
        online.run("bad", opponents="nobody=1", root=tmp_path, quiet=True)


def test_the_gate_keeps_the_better_network_and_the_batches_grow(tmp_path, monkeypatch):
    calls = []

    def fake_score(model, k_model, specs, workers):               # the stand-in scores 100; learners 50, 150, 120
        calls.append(model)
        epoch = None if model == "heuristic" else int(model.split("epoch_")[1][:3])
        mean = {None: 100.0, 0: 50.0, 1: 150.0, 2: 120.0}[epoch]
        return dict(model=model, k_model=k_model, games=2, wins=1, mean_diff=mean, se=0.0, deaths=0,
                    opponents={"noisy": dict(games=2, wins=1, mean_diff=mean)}, diffs=[mean, mean])

    monkeypatch.setattr(online, "gate_score", fake_score)
    out = online.run("g", epochs=3, games=2, workers=1, turns=30, opponents="noisy=1", enemy_randomness=1.0,
                     explore=0.2, close_share=0.5, train_epochs=1, patience=1, root=tmp_path, quiet=True,
                     gate_games=2, grow=2.0, max_games=3, learner_share=1.0)
    run = tmp_path / "g"
    learner = [str(online.epoch_dir(run, e) / "weights.pt") for e in range(3)]
    assert [s["gate"]["accepted"] for s in out] == [False, True, False]
    assert calls == ["heuristic", learner[0], learner[1], learner[2]]                    # the player is measured once
    assert [s["played_with"] for s in out] == ["heuristic", "heuristic", learner[1]]
    assert [s["games"]["all"]["games"] for s in out] == [2, 2, 3]                       # grown on acceptance, capped
    assert out[1]["learner_played"] == learner[0]                                        # the rejected learner still plays
    epoch1 = json.loads((online.epoch_dir(run, 1) / "games.json").read_text())
    assert {g["spec"]["model"] for g in epoch1} == {learner[0]}                          # learner_share 1
    assert out[2]["state"]["player"] == learner[1] and (run / "best.pt").read_bytes() == (online.epoch_dir(run, 1) / "weights.pt").read_bytes()
    assert len(out[2]["dataset"]["epochs"]) == 3                                        # history: the last 3 epochs


def test_a_run_can_pretrain_on_another_runs_games_and_inherit_its_k(tmp_path, monkeypatch):
    tiny = dict(workers=1, turns=40, opponents="noisy=1,enemy_copy=1", enemy_randomness=1.0, explore=0.2,
                close_share=0.5, train_epochs=1, patience=1, root=tmp_path, quiet=True, gate_games=2, belief_share=1.0,
                play_k="trained")
    online.run("src", epochs=2, games=3, **tiny)
    src = tmp_path / "src"
    k = str(online.epoch_dir(src, 1) / "k.npz")

    monkeypatch.setattr(online.locations, "ONLINE", tmp_path)
    assert online.pretrain_dirs(["src:0-1"]) == [online.epoch_dir(src, 0) / "raw", online.epoch_dir(src, 1) / "raw"]
    assert online.pretrain_dirs(["src:1"]) == [online.epoch_dir(src, 1) / "raw"]
    assert online.pretrain_dirs([str(online.epoch_dir(src, 0))]) == [online.epoch_dir(src, 0) / "raw"]
    with pytest.raises(FileNotFoundError):
        online.pretrain_dirs(["nowhere:0-3"])

    out = online.run("dst", epochs=1, games=3, pretrain=["src:0-1"], pretrain_epochs=1, start_k=k, **tiny)
    dst = tmp_path / "dst"
    pre = dst / "pretrain" / "weights.pt"
    assert pre.exists() and json.loads(pre.with_suffix(".json").read_text())["meta"]["pretrain"] is True
    assert out[0]["played_with"] == "heuristic" and out[0]["learner_played"] == str(pre)   # Q: the stand-in plays
    games = json.loads((online.epoch_dir(dst, 0) / "games.json").read_text())
    assert {g["spec"]["k_model"] for g in games} == {k}                                    # K: inherited
    assert json.loads((online.epoch_dir(dst, 0) / "gate.json").read_text())["player"]["k_model"] == k
    assert json.loads((dst / "config.json").read_text())["start_k"] == k


def test_the_champion_is_our_agent_with_a_fixed_q_and_the_hand_set_k(tmp_path):
    spec = collect.GameSpec(index=0, opponent="champion", seat=0, close=True, turns=40, seed=4, explore=0.0,
                            model="heuristic", out=str(tmp_path), record=False, opponent_randomness=0.1,
                            champion="heuristic")
    rival, drawn = collect.make_opponent(spec)
    assert drawn == dict(randomness=0.1, model="heuristic")
    assert type(rival.agent.runtime.belief).__name__ == "BeliefFilter"                  # the hand-set K, its own belief
    assert collect.play_one(spec)["failed_turns"] == 0
    assert collect.cost(spec) > collect.cost(collect.GameSpec(**{**spec.__dict__, "opponent": "noisy"}))
    old = collect.GameSpec(**{**spec.__dict__})
    object.__delattr__(old, "champion")                                                    # a spec from an older run
    assert collect.spec_dict(old)["champion"] is None
    assert "champion" in online.DEFAULT_OPPONENTS


def test_exploration_rises_after_rejections_and_resets_on_acceptance(tmp_path, monkeypatch):
    scores = iter([100.0, 50.0, 50.0, 50.0, 200.0])                    # the stand-in, then learners: 3 rejected, 1 accepted

    def fake_score(model, k_model, specs, workers):
        mean = next(scores)
        return dict(model=model, k_model=k_model, games=2, wins=1, mean_diff=mean, se=0.0, deaths=0,
                    opponents={"noisy": dict(games=2, wins=1, mean_diff=mean)}, diffs=[mean, mean])

    monkeypatch.setattr(online, "gate_score", fake_score)
    out = online.run("x", epochs=4, games=2, workers=1, turns=30, opponents="noisy=1", explore=0.04, close_share=0.5,
                     train_epochs=1, patience=1, root=tmp_path, quiet=True, gate_games=2, width=1,
                     explore_boost_after=3, explore_boost=2.0, explore_max=0.1)
    assert [s["state"]["explore"] for s in out] == [0.04, 0.04, 0.08, 0.04]
    assert [s["gate"]["accepted"] for s in out] == [False, False, False, True]
    games = json.loads((online.epoch_dir(tmp_path / "x", 3) / "games.json").read_text())
    assert {g["spec"]["explore"] for g in games} == {0.08}                                # epoch 3 played at 0.08


def test_by_default_our_agents_play_with_the_hand_set_k_while_k_still_trains(tmp_path):
    out = online.run("h", epochs=2, games=3, workers=1, turns=40, opponents="noisy=1,enemy_copy=1", enemy_randomness=1.0,
                     explore=0.2, close_share=0.5, train_epochs=1, patience=1, root=tmp_path, quiet=True, gate_games=2,
                     belief_share=1.0, width=1)
    run = tmp_path / "h"
    assert (online.epoch_dir(run, 0) / "k.npz").exists() and out[1]["k"]["val_nll"] is not None   # K is trained
    for epoch in (0, 1):
        games = json.loads((online.epoch_dir(run, epoch) / "games.json").read_text())
        assert {g["spec"]["k_model"] for g in games} == {None}                                  # but not played
        gate = json.loads((online.epoch_dir(run, epoch) / "gate.json").read_text())
        assert gate["player"]["k_model"] is None and gate["learner"]["k_model"] is None
