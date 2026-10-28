"""The training tools: the data handler's saved datasets and bootstrapped labels, the trainer's command line and
device choice. (The whole loop, collection to training, is tested in test_q_online.py.)"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from candidate.offline import collect, locations  # noqa: E402
from candidate.offline import dataset as ds  # noqa: E402
from candidate.offline import train  # noqa: E402
from candidate.agent.q.config import Q  # noqa: E402
from candidate.agent.q.network import load_network  # noqa: E402


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    out = tmp_path_factory.mktemp("raw") / "run"
    specs = collect.plan(3, [("hunter", 1), ("enemy", 1), ("basic", 1)], close_share=0.5, turns=60, explore=0.2,
                         model="heuristic", out=str(out), seed=5)
    collect.run(specs, workers=1, echo=lambda line: None)
    return out


# ---- the data handler -------------------------------------------------------------------------------------------------


def test_raw_games_are_found_recursively_but_replays_are_not_listed(raw, tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    copied = nested / Path(next(raw.glob("*.npz"))).name
    copied.write_bytes(next(raw.glob("*.npz")).read_bytes())
    (tmp_path / "a" / "replayed-xyz").mkdir()
    (tmp_path / "a" / "replayed-xyz" / "r.npz").write_bytes(copied.read_bytes())
    assert ds.find_records([tmp_path]) == [copied]
    assert len(ds.find_records([raw])) == 3


def test_a_built_dataset_saves_and_loads_unchanged_with_its_meta(raw, tmp_path):
    data = ds.load_dataset([raw], echo=None)
    path = data.save(tmp_path / "set.npz")
    back = ds.Dataset.load(path)
    for key in ds.Dataset._ARRAYS:
        assert np.array_equal(getattr(back, key), getattr(data, key)), key
    assert back.games == data.games and back.meta["inputs_version"] == data.meta["inputs_version"]
    assert back.meta["label_window"] == Q.label_window and back.meta["handling"] == {"ok": 3}
    summary = back.summary()
    assert summary["games"] == 3 and summary["frames"] == 180 and not summary["stale_inputs"]
    assert set(summary["opponents"]) == {"hunter", "enemy", "basic"}
    assert sum(summary["state_share"].values()) == pytest.approx(1.0)


def test_a_dataset_name_is_found_in_the_datasets_directory():
    assert locations.dataset_path("round0b") == locations.DATASETS / "round0b.npz"
    assert locations.dataset_path("x/y.npz") == Path("x/y.npz")
    assert locations.weights_path("q_round0") == locations.WEIGHTS / "q_round0.pt"


class _Constant:
    """A stand-in network whose value of every state is the same number (a QNet whose last layer outputs only its
    bias)."""

    def __new__(cls, value):
        from candidate.agent.q.network import QNet

        model = QNet(("scalars",))
        with torch.no_grad():
            model.output_layer.weight.zero_()
            model.output_layer.bias.fill_(value)
        model.meta = {"name": "constant"}
        return model


def test_bootstrapped_labels_use_the_previous_rounds_value_at_the_end_of_the_window(raw):
    game = ds.load_game(next(raw.glob("*.npz")))
    plain, has = ds.labels(game, window=10)
    boot, has2 = ds.labels(game, window=10, bootstrap=_Constant(7.0))
    a = game.arrays
    diff = a["my_count"].astype(float) - a["opp_count"].astype(float)
    n = len(diff)
    inside = np.arange(n) + 10 < n
    assert np.array_equal(has, has2)
    assert np.allclose(boot[inside], diff[10:] - diff[:-10] + 7.0)
    assert np.allclose(boot[~inside], plain[~inside])                    # the end of the game is unchanged


def test_the_data_handlers_command_line_builds_and_describes_a_dataset(raw, tmp_path, capsys):
    path = tmp_path / "cli.npz"
    assert ds.main(["build", "--raw", str(raw), "--name", str(path)]) == 0
    assert path.exists() and "3 games, 180 frames" in capsys.readouterr().out
    assert ds.main(["info", str(path)]) == 0
    assert "states taken" in capsys.readouterr().out


# ---- the trainer ------------------------------------------------------------------------------------------------------------


def test_the_trainer_runs_from_a_saved_dataset_in_smoke_mode(raw, tmp_path):
    data = ds.load_dataset([raw], echo=None)
    dataset_file = data.save(tmp_path / "set.npz")
    out = tmp_path / "w.pt"
    assert train.main(["--dataset", str(dataset_file), "--smoke", "--name", str(out)]) == 0
    network = load_network(out)
    report = json.loads(out.with_suffix(".json").read_text())
    assert network.meta["smoke"] is True and report["report"]["device"] in ("cpu", "cuda")


def test_the_device_is_the_gpu_only_if_it_really_runs_kernels(monkeypatch):
    assert train.pick_device("cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def broken(*args, **kwargs):
        raise RuntimeError("no kernel image is available for execution on the device")

    real_zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, device=None, **k: broken() if device == "cuda" else real_zeros(*a, **k))
    assert train.pick_device().type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_training_runs_on_the_gpu_when_it_works(raw):
    if train.pick_device().type != "cuda":
        pytest.skip("this PyTorch build cannot run kernels on the GPU")
    data = ds.load_dataset([raw], echo=None)
    model, report = train.staged(data, epochs=1, patience=1, echo=None)
    assert report["device"] == "cuda"


# ---- every branch, every time ----------------------------------------------------------------------------------------


def test_a_network_from_scratch_trains_all_three_branches(raw):
    data = ds.load_dataset([raw], echo=None)
    model, report = train.full(data, epochs=1, patience=1, device="cpu", echo=None)
    assert model.branches == ("grid", "scalars", "probmap") and report["branches"] == ["grid", "scalars", "probmap"]


def test_switching_on_branches_keeps_the_outputs_and_then_they_train(raw):
    from candidate.agent.q.network import QNet

    data = ds.load_dataset([raw], echo=None)
    torch.manual_seed(0)
    model = QNet(("scalars",))
    g, s, p = (torch.as_tensor(np.asarray(a[:64], dtype=np.float32)) for a in (data.grid, data.scalars, data.probmap))
    before = model(g, s, p).detach()
    assert train.add_branches(model) == ["grid", "probmap"] and model.branches == ("grid", "scalars", "probmap")
    assert torch.allclose(model(g, s, p), before)                         # same outputs to begin with
    assert train.add_branches(model) == []
    tuned, report = train.finetune(QNet(("scalars",)), data, epochs=2, patience=2, device="cpu", echo=None,
                                   branches=("grid", "scalars", "probmap"))
    assert report["branches"] == ["grid", "scalars", "probmap"]
    first = tuned.head[0].weight.detach()
    assert first[:, :64].abs().sum() > 0 or report["val"] == report["start_val"]  # the grid's weights moved off zero
