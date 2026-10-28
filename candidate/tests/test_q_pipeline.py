"""The offline pipeline: collecting a recorded game, replaying it exactly, labelling it, and training on it."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from candidate.agent.q.network import load_network, predict  # noqa: E402
from candidate.agent.q.record import load_record  # noqa: E402
from candidate.offline import collect, match, replay  # noqa: E402
from candidate.offline import dataset as ds  # noqa: E402
from candidate.offline import train  # noqa: E402
from territorygame.api.agent_controller import AgentController  # noqa: E402
from territorygame.api.direction import Direction as FD  # noqa: E402

COMPARED = ("turn", "previous", "state", "direction", "my_count", "opp_count", "d5", "d7", "trail_len",
            "grid_bits", "scalars", "probmap")


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """Two short recorded games (against the hunter at the close respawns, and the enemy far), with exploration."""
    out = tmp_path_factory.mktemp("games")
    specs = collect.plan(2, [("hunter", 1), ("enemy", 1)], close_share=0.5, turns=70, explore=0.2, model="heuristic",
                         out=str(out), seed=3)
    summaries = collect.run(specs, workers=1, echo=lambda line: None)
    return out, summaries


# ---- collection ---------------------------------------------------------------------------------------------------------


def test_a_collected_game_has_its_record_and_a_sidecar_with_both_tapes(recorded):
    out, summaries = recorded
    for summary in summaries:
        record = Path(summary["record"])
        meta, arrays = load_record(record)
        side = json.loads(record.with_suffix(".json").read_text())
        assert meta["complete"] and meta["frames"] == 70 and meta["explore_rate"] == 0.2
        assert [len(t) for t in side["tapes"]] == [70, 70] and "-" not in side["tapes"][side["config"]["seat"]]
        seat = side["config"]["seat"]
        assert side["final"]["counts"][seat] == summary["my_count"]
        assert arrays["explored"].any()
        letters = "NESW"
        assert "".join(letters[d] for d in arrays["direction"]) == side["tapes"][seat]   # the record and the tape agree


def test_replaying_a_game_regenerates_exactly_the_recorded_inputs(recorded, tmp_path):
    out, summaries = recorded
    for summary in summaries:
        original = Path(summary["record"])
        again = replay.replay(original, tmp_path)
        _, a = load_record(original)
        _, b = load_record(again)
        for key in COMPARED:
            assert np.array_equal(a[key], b[key]), key


class _Stumbles(AgentController):
    """Paces east and west; on its third turn it fails (raises), which the engine records as a forfeit."""

    def __init__(self):
        self.turns = 0
        self._left = None

    def take_turn(self, game):
        if game.get_remaining_turns() != self._left:
            self._left = game.get_remaining_turns()
            self.turns += 1
        if self.turns == 3:
            raise RuntimeError("stumbled")
        game.move(FD.EAST if self.turns % 2 else FD.WEST)

    def get_debug_state(self):
        return ""


def test_a_forfeited_turn_is_taped_and_replayed_as_a_forfeit():
    first = match.Taped(_Stumbles())
    second = match.Taped(_Stumbles())
    result = match.play([first, second], 12)
    assert first.tape[2] == "-" and second.tape[2] == "-" and result.failed_turns == 2
    again = match.play([match.Replayer(first.tape), match.Replayer(second.tape)], 12)
    assert again.counts == result.counts and again.failed_turns == 2


# ---- labels -----------------------------------------------------------------------------------------------------------------


def synthetic_game(my, opp, d5, d7, complete=True, final=None, name="g"):
    n = len(my)
    arrays = dict(turn=np.arange(n), my_count=np.array(my), opp_count=np.array(opp), d5=np.array(d5),
                  d7=np.array(d7, dtype=np.float32), state=np.zeros(n, dtype=np.int8), explored=np.zeros(n, dtype=bool),
                  grid_bits=np.zeros((n, 386), dtype=np.uint8), scalars=np.zeros((n, 23), dtype=np.float32),
                  probmap=np.zeros((n, 10, 10, 10), dtype=np.float16))
    sidecar = None if final is None else dict(config=dict(seat=0), final=dict(counts=list(final)))
    return ds.Game(Path(name), dict(game_id=name, complete=complete, frames=n), arrays, sidecar)


def test_the_label_is_the_score_change_over_the_window_plus_a_safe_open_loop():
    game = synthetic_game(my=[10, 10, 12, 15, 15], opp=[9, 9, 9, 9, 11], d5=[0, 3, 5, 7, 0], d7=[5, 5, 5, -1, 5],
                          final=(20, 12))
    y, has = ds.labels(game, window=2)
    # t=0: diff 1 -> t=2 diff 3, plus d5=5 (d7=5 > 0): 2 + 5
    # t=1: diff 1 -> t=3 diff 6, d5=7 but d7=-1 (unsafe): 5
    # t=2: diff 3 -> t=4 diff 4, d5=0: 1
    # t=3, t=4: the window passes the end: final diff 8 minus their diffs, no open-loop credit
    assert has.all()
    assert y.tolist() == [7.0, 5.0, 1.0, 2.0, 4.0]


def test_a_game_recorded_without_its_end_only_labels_the_frames_whose_window_it_contains():
    game = synthetic_game(my=[1, 2, 3, 4], opp=[0, 0, 0, 0], d5=[0] * 4, d7=[1] * 4, complete=False)
    y, has = ds.labels(game, window=2)
    assert has.tolist() == [True, True, False, False] and y[:2].tolist() == [2.0, 2.0]


def test_a_complete_game_without_engine_counts_ends_on_its_last_frame():
    game = synthetic_game(my=[1, 2, 3, 9], opp=[0, 0, 0, 0], d5=[0] * 4, d7=[1] * 4, complete=True)
    y, has = ds.labels(game, window=2)
    # t=0: 1 -> 3; t=1: 2 -> 9; t=2 and t=3 run past the end: the last frame's diff (9) minus theirs
    assert has.all() and y.tolist() == [2.0, 7.0, 6.0, 0.0]


def test_every_game_weighs_the_same():
    long = synthetic_game(my=list(range(30)), opp=[0] * 30, d5=[0] * 30, d7=[1] * 30, final=(30, 0), name="long")
    short = synthetic_game(my=list(range(6)), opp=[0] * 6, d5=[0] * 6, d7=[1] * 6, final=(6, 0), name="short")
    data = ds.build([long, short], window=2)
    assert len(data) == 36 and data.games == ["long", "short"]
    assert data.weight[data.game == 0].sum() == pytest.approx(data.weight[data.game == 1].sum())
    assert data.weight.mean() == pytest.approx(1.0)


def test_the_split_is_by_game_covers_every_game_once_and_never_moves_a_game():
    names = [f"20261011-1200{i:02d}-{i}-1-g{i:04d}-noisy-p0" for i in range(400)]
    train_g, val_g, test_g = train.split_games(names)
    together = np.concatenate([train_g, val_g, test_g])
    assert sorted(together.tolist()) == list(range(400)) and len(test_g) == 0       # no test split by default
    assert 0.10 < len(val_g) / 400 < 0.20                                              # about 15% for validation
    # the next epoch's dataset drops the oldest games and adds new ones: every game keeps its part
    later = names[100:] + [f"x{i}" for i in range(100)]
    _, later_val, _ = train.split_games(later)
    assert {later[i] for i in later_val} & set(names) == {names[i] for i in val_g if i >= 100}
    _, val3, test3 = train.split_games(names, test_fraction=0.15)
    assert np.array_equal(val3, val_g) and 0.10 < len(test3) / 400 < 0.20 and not set(test3) & set(val3)
    assert len(train.split_games(["a", "b"])[1]) == 1 and len(train.split_games(["a"])[1]) == 0


def test_a_frame_stride_keeps_every_kth_frame_with_its_full_window_label():
    game = synthetic_game(my=list(range(40)), opp=[0] * 40, d5=[0] * 40, d7=[1] * 40, final=(40, 0), name="g")
    every = ds.build([game], window=5)
    half = ds.build([game], window=5, frame_stride=2)
    assert len(half) == (len(every) + 1) // 2 and half.meta["frame_stride"] == 2
    assert np.array_equal(half.label, every.label[::2])                    # the same labels, every second frame
    assert half.weight.mean() == pytest.approx(1.0)


def test_records_made_by_older_code_are_replayed_skipped_or_used_as_asked(recorded, tmp_path):
    out, summaries = recorded
    source = Path(summaries[0]["record"])
    meta, arrays = load_record(source)
    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    stale = stale_dir / source.name
    np.savez_compressed(stale, meta=np.array(json.dumps(dict(meta, inputs_version="old"))), **arrays)
    stale.with_suffix(".json").write_text(source.with_suffix(".json").read_text())
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            ds.load_dataset([stale_dir], stale="skip", echo=None)              # nothing left to build from
    assert len(ds.load_dataset([stale_dir], stale="use", echo=None)) == 70
    replayed = ds.load_dataset([stale_dir], stale="replay", replay_dir=tmp_path / "fresh", echo=None)
    assert len(replayed) == 70 and np.array_equal(replayed.scalars, ds.load_dataset([source], echo=None).scalars)


# ---- training ------------------------------------------------------------------------------------------------------------------


def learnable(n_games=30, frames=120, seed=0):
    """Random inputs whose label is a simple function of a scalar and of the action taken."""
    rng = np.random.default_rng(seed)
    n = n_games * frames
    scalars = rng.normal(size=(n, 23)).astype(np.float32)
    action = rng.integers(0, 5, n)
    label = (20.0 * scalars[:, 0] + 10.0 * action - 15.0).astype(np.float32)
    return ds.Dataset(
        grid=np.zeros((n, 7, 21, 21), dtype=np.uint8), scalars=scalars,
        probmap=np.zeros((n, 10, 10, 10), dtype=np.float16), action=action.astype(np.int64), label=label,
        game=np.repeat(np.arange(n_games), frames).astype(np.int32), weight=np.ones(n, dtype=np.float32),
        explored=np.zeros(n, dtype=bool), games=[f"g{i}" for i in range(n_games)], game_meta=[{}] * n_games,
    )


def test_training_learns_a_learnable_target_and_exports_a_network_that_outputs_cells(tmp_path):
    data = learnable()
    model, report = train.staged(data, epochs=40, patience=8, device="cpu", echo=None, stages=(("scalars",),),
                                 test_fraction=0.15)
    assert report["test_r2"] > 0.9 and report["test"] < 0.1 * report["test_constant_baseline"]
    assert report["val_r2"] > 0.9
    path = train.export(model, tmp_path / "q.pt", report)
    network = load_network(path)
    test_idx = data.subset(train.split_games(data.games, test_fraction=0.15)[2])
    values = predict(network, data.grid[test_idx], data.scalars[test_idx], data.probmap[test_idx])
    taken = values[np.arange(len(test_idx)), data.action[test_idx]]
    mse = float(((taken - data.label[test_idx]) ** 2).mean())
    assert mse == pytest.approx(report["test"], rel=1e-3)                     # the export is in cells, like the report
    saved = json.loads(path.with_suffix(".json").read_text())
    assert saved["meta"]["branches"] == ["scalars"] and saved["report"]["test_r2"] > 0.9


def test_a_branch_is_kept_only_if_it_clearly_helps():
    # The grid and probmap carry nothing here. The labels are noisy, like real ones: with a noiseless target an
    # extra branch (even fed zeros) speeds up fitting an underfit model and can win by a few percent.
    data = learnable(n_games=20, frames=80, seed=1)
    data.label = data.label + np.random.default_rng(5).normal(0.0, 20.0, len(data.label)).astype(np.float32)
    model, report = train.staged(data, epochs=15, patience=4, device="cpu", echo=None)
    assert report["branches"] == ["scalars"] and [s["kept"] for s in report["stages"]] == [True, False, False]
    for stage in report["stages"][1:]:                            # each was judged against a control trained as long
        assert "control_val" in stage


def test_the_calibration_table_bins_each_states_predictions():
    pred = np.tile(np.arange(10, dtype=float)[:, None], (1, 6))
    action = np.zeros(10, dtype=int)
    table = train.calibration(pred, action, np.arange(10, dtype=float), bins=5)
    assert table["EXPAND"]["n"] == 10 and len(table["EXPAND"]["bins"]) == 5
    assert all(b["pred"] == b["actual"] for b in table["EXPAND"]["bins"]) and table["RETURN"] == {"n": 0}


def test_the_device_falls_back_to_the_cpu_when_cuda_cannot_run():
    assert train.pick_device("cpu").type == "cpu"
    assert train.pick_device().type in ("cpu", "cuda")


def test_belief_games_can_be_given_the_same_total_weight_as_true_world_games():
    def game(name, oracle, n):
        g = synthetic_game([9] * n, [9] * n, [0] * n, [1] * n, final=(9, 9), name=name)
        g.meta["oracle"] = oracle
        return g

    games = [game("a", True, 300), game("b", True, 300), game("c", True, 300), game("d", False, 300)]
    data = ds.build(games, window=50, balance_belief=True)
    oracle = np.isin(data.game, [0, 1, 2])
    assert data.meta["belief_balanced"]
    assert np.isclose(data.weight[oracle].sum(), data.weight[~oracle].sum(), rtol=1e-4)
    plain = ds.build(games, window=50)                # the default: every game the same
    assert np.isclose(plain.weight[oracle].sum(), 3 * plain.weight[~oracle].sum(), rtol=1e-4)


# ---- labels weighted by distance -------------------------------------------------------------------------------------


def test_weighted_labels_keep_the_intensity_of_a_steady_gain_and_count_near_changes_more():
    ends, w = ds.label_weights(100, (20, 50), (1.0, 0.6, 0.3))
    assert ends == [20, 50, 100] and sum(wi * (e - s) for wi, s, e in zip(w, [0, 20, 50], ends)) == pytest.approx(100.0)
    assert w[0] > w[1] > w[2] and w[0] / w[2] == pytest.approx(1.0 / 0.3)
    n = 300
    steady = synthetic_game(my=list(range(n)), opp=[0] * n, d5=[0] * n, d7=[0] * n, final=(n - 1, 0))
    y, has = ds.labels(steady, 100)
    assert np.allclose(y[:150], 100.0)                                     # a steady gain of 1 a frame: still 100
    jump = [0] * 10 + [30] * (n - 10)                                      # +30 cells at frame 10
    game = synthetic_game(my=jump, opp=[0] * n, d5=[0] * n, d7=[0] * n, final=(30, 0))
    y, _ = ds.labels(game, 100)
    assert y[5] == pytest.approx(30 * w[0])                                # 5 frames ahead: the near weight
    assert y[0] == pytest.approx(30 * w[0])                                # 10 ahead: still the first segment
    late = [0] * 90 + [30] * (n - 90)                                      # the same +30, 80 frames after frame 10
    y_late, _ = ds.labels(synthetic_game(my=late, opp=[0] * n, d5=[0] * n, d7=[0] * n, final=(30, 0)), 100)
    assert y_late[10] == pytest.approx(30 * w[2]) and y_late[10] < y[5]


def test_equal_weights_give_the_unweighted_label_and_the_tail_takes_the_last_weight():
    n = 260
    rng = np.random.default_rng(0)
    my = np.cumsum(rng.integers(-2, 4, n)).tolist()
    d5 = rng.integers(0, 20, n).tolist()
    d7 = rng.integers(-2, 3, n).tolist()
    game = synthetic_game(my=my, opp=[0] * n, d5=d5, d7=d7, final=(my[-1], 0))
    flat, _ = ds.labels(game, 100, horizons=(20, 50), weights=(1.0, 1.0, 1.0))
    t = 30
    open_loop = d5[t + 100] if d7[t + 100] > 0 else 0
    assert flat[t] == pytest.approx(my[t + 100] - my[t] + open_loop)        # the brief's label, exactly
    assert flat[200] == pytest.approx(my[-1] - my[200])                     # past the end: the final difference
    ends, w = ds.label_weights(100, (20, 50), (1.0, 0.6, 0.3))
    weighted, _ = ds.labels(game, 100)
    expected = w[0] * (my[t + 20] - my[t]) + w[1] * (my[t + 50] - my[t + 20]) + w[2] * (my[t + 100] - my[t + 50])
    assert weighted[t] == pytest.approx(expected + w[2] * open_loop)
    with pytest.raises(ValueError):
        ds.label_weights(100, (20, 50), (1.0, 0.5))
