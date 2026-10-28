"""The Q network (one PyTorch module, for training and play): architecture, branch switches, files, devices, and the
strategy's view of it."""

import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from candidate.agent.q.network import (  # noqa: E402
    BRANCHES, NetworkQ, QNet, load_network, parameter_count, pick_device, predict, save_network,
)


def random_inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    grid = (rng.random((n, 7, 21, 21)) < 0.3).astype(np.uint8)
    scalars = rng.normal(size=(n, 23)).astype(np.float32)
    probmap = rng.random((n, 10, 10, 10)).astype(np.float16)
    return grid, scalars, probmap


def test_the_network_has_the_briefs_size_and_shapes():
    assert parameter_count() == 79_478                                 # "about 80,000"; 10 map channels, 6 states
    out = QNet()(torch.zeros(3, 7, 21, 21), torch.zeros(3, 23), torch.zeros(3, 10, 10, 10))
    assert out.shape == (3, 6)


def test_the_branches_shrink_as_the_brief_says():
    model = QNet()
    sizes = []
    x = torch.zeros(1, 7, 21, 21)
    for layer in model.grid:
        x = layer(x)
        if isinstance(layer, torch.nn.Conv2d):
            sizes.append(x.shape[-1])
    assert sizes == [21, 11, 6, 3]
    sizes = []
    x = torch.zeros(1, 10, 10, 10)
    for layer in model.probmap:
        x = layer(x)
        if isinstance(layer, torch.nn.Conv2d):
            sizes.append(x.shape[-1])
    assert sizes == [10, 5, 3]


def test_a_switched_off_branch_ignores_its_input():
    torch.manual_seed(1)
    model = QNet(("scalars",)).eval()
    grid, scalars, probmap = random_inputs(4)
    a = predict(model, grid, scalars, probmap)
    b = predict(model, np.zeros_like(grid), scalars, np.zeros_like(probmap))
    assert np.array_equal(a, b)
    model.set_branches(BRANCHES)
    assert not np.array_equal(predict(model, grid, scalars, probmap), a)
    with pytest.raises(ValueError):
        model.set_branches(("vision",))


def test_saving_and_loading_keeps_the_weights_the_branches_and_the_meta(tmp_path):
    torch.manual_seed(2)
    model = QNet(("scalars", "grid"))
    path = save_network(model, tmp_path / "q.pt", {"name": "test", "round": 0})
    loaded = load_network(path)
    assert loaded.branches == ("grid", "scalars") and loaded.meta == {"name": "test", "round": 0}
    grid, scalars, probmap = random_inputs(8)
    assert np.array_equal(predict(loaded, grid, scalars, probmap), predict(model.eval(), grid, scalars, probmap))


def test_the_device_is_the_gpu_only_if_a_kernel_really_runs(monkeypatch):
    assert pick_device("cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    real_zeros = torch.zeros

    def zeros(*args, device=None, **kwargs):
        if device == "cuda":
            raise RuntimeError("no kernel image is available for execution on the device")
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros)
    assert pick_device().type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_the_gpu_and_the_cpu_give_the_same_values():
    if pick_device().type != "cuda":
        pytest.skip("this PyTorch build cannot run kernels on the GPU")
    torch.manual_seed(3)
    model = QNet()
    grid, scalars, probmap = random_inputs(32)
    cpu = predict(model.to("cpu"), grid, scalars, probmap)
    gpu = predict(model.to("cuda"), grid, scalars, probmap)
    assert np.abs(cpu - gpu).max() < 1e-3


class _Inputs:
    def __init__(self, grid, scalars, probmap):
        self.grid, self.scalars, self.probmap = grid, scalars, probmap


def test_network_q_gives_one_frames_six_values_and_needs_the_inputs():
    torch.manual_seed(4)
    q = NetworkQ(QNet(), device="cpu")
    grid, scalars, probmap = random_inputs(1)
    values = q.values(None, _Inputs(grid[0], scalars[0], probmap[0]))
    assert values.shape == (6,) and values.dtype == np.float64
    assert np.allclose(values, predict(q.model, grid, scalars, probmap)[0])
    with pytest.raises(ValueError):
        q.values(None, None)


def test_the_controller_does_not_import_pytorch_unless_a_network_is_asked_for():
    code = ("import sys; import candidate.agent.q.strategy, candidate.candidate_controller as c; "
            "print('torch' in sys.modules); c.load_q(None); print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


def test_the_guis_controller_takes_its_weights_from_the_variable_and_warns_about_old_inputs(monkeypatch, tmp_path):
    import candidate.candidate_controller as cc
    from candidate.agent.diagnostics import Diagnostics
    from candidate.agent.q.version import inputs_version

    current = save_network(QNet(("scalars",)), tmp_path / "now.pt", {"name": "now", "inputs_version": inputs_version()})
    old = save_network(QNet(("scalars",)), tmp_path / "old.pt", {"name": "old", "inputs_version": "0123456789abcdef"})
    said = []
    monkeypatch.setattr(cc, "_echo", said.append)
    monkeypatch.setenv(cc.WEIGHTS_VARIABLE, str(current))
    q = cc.CandidateController(record=False, diagnostics=Diagnostics(echo=said.append))._agent.strategy.q_model
    assert isinstance(q, NetworkQ) and q.inputs_match and any("the network" in s for s in said)
    assert not any("another code version" in s for s in said)
    monkeypatch.setenv(cc.WEIGHTS_VARIABLE, str(old))
    q = cc.CandidateController(record=False, diagnostics=Diagnostics(echo=said.append))._agent.strategy.q_model
    assert isinstance(q, NetworkQ) and not q.inputs_match and any("another code version" in s for s in said)
    monkeypatch.delenv(cc.WEIGHTS_VARIABLE)
    plain = cc.CandidateController(record=False, diagnostics=Diagnostics(echo=None))
    assert not isinstance(plain._agent.strategy.q_model, NetworkQ)       # DEFAULT_WEIGHTS: the stand-in
