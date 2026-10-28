"""The trainer: the Q network on a training set (brief, "Labels and training" and "Build it in stages").

    python -m candidate.offline.train --dataset round0b --name q_round0          # a set built by offline.dataset
    python -m candidate.offline.train --raw q_training/raw/round0b --name q_round0   # or straight from raw games
    python -m candidate.offline.train --dataset round0b --smoke                  # pipeline check: 1 epoch, few games

Weights go to q_training/weights/<name>.pt (agent.q.network's format, which the controller loads as it is), with
<name>.json (the report) next to them. The device is the GPU when PyTorch can really run a kernel on it, else the CPU
(agent.q.network.pick_device); the network is small enough for either.

  * Loss: squared error between the taken state's output and the label. The other four outputs get no gradient for
    that example (their outcomes were never observed). No Huber loss: deaths are large, rare errors, and they are the
    signal. Every game weighs the same.
  * Adam, learning rate 1e-3, batch 256, weight decay 1e-4. Split by game, never by frame, and fixed by each game's
    id (split_games): 85% train, 15% validation, so a game is in the same part in every dataset it appears in (the
    online loop trains on the last few epochs' games; re-splitting at random put games a network had trained on into
    its validation set). No test split by default (--test-share for one): the online loop's test is the gate, which
    plays games. Stop when the validation loss has not improved for `patience` epochs, and keep the best epoch.
  * Augmentation: the whole board is rotated by a random quarter turn and mirrored at random, the same transform for
    the grid and the probability map (the scalars are counts and distances and do not change).
  * Fine-tuning (finetune): an existing network trained further on new games with its branches as they are, the
    online loop's step (offline/online.py). It keeps the starting weights if no epoch beats them on validation.
  * All branches (full, and finetune with `branches`), what the online loop uses: every network reads the scalars, the
    grid and the probability map. A branch switched on in an existing network starts with its head weights at zero, so
    the network's outputs do not change until training moves them (add_branches).
  * Stages (staged, the brief's build; the CLI's default): the scalar branch alone; then add the grid, kept only if the validation loss clearly drops (by `keep`,
    2%); then the probability-map branch under the same test. Each stage starts from the weights chosen so far. Because a
    later stage also trains for longer, it is compared not with the earlier stage's loss but with a control: the
    chosen configuration trained on for the same budget from the same weights. A branch is kept only if it beats that
    control; otherwise the control (now better trained) carries on.

The weights record what they were trained on in their metadata. Labels are in cells and range over about +-100; training divides them by
their standard deviation (on the training games) and the export multiplies the last layer back, so the exported network
outputs cells, as the brief requires, and every loss reported is in cells squared. The JSON report holds: losses per
stage, the validation loss against a constant baseline (val_r2), and with a test split the test loss against one
(test_r2) and per-state calibration (predicted value against the actual label, in bins, on the test games).
"""

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from candidate.agent.q.network import BRANCHES, QNet, pick_device, predict, save_network
from candidate.agent.behavior.states import State
from candidate.agent.q.version import inputs_version
from candidate.offline import locations
from candidate.offline.dataset import Dataset, load_dataset


def _copy_state(model: QNet) -> dict:
    """A detached copy of the weights, to come back to (best epoch, chosen stage)."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


VAL_FRACTION = 0.15


def game_fraction(name: str) -> float:
    """A number in [0, 1) fixed by the game's name (its game id) alone."""
    return zlib.crc32(name.encode("utf-8")) / 2**32


def split_games(names: list[str], val_fraction: float = VAL_FRACTION, test_fraction: float = 0.0
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Game indices for train, validation and test. A game's part is decided by its name alone, so a game seen in
    several datasets (the online loop trains on the last few epochs' games) is in the same part every time: a network
    is never validated on games it was trained on before. With two or more games there is at least one validation
    game (the one whose name hashes lowest), and with three or more and a test share at least one test game."""
    u = np.array([game_fraction(n) for n in names], dtype=np.float64)
    val = u < val_fraction
    test = (u >= val_fraction) & (u < val_fraction + test_fraction)
    n = len(names)
    if n >= 2 and not val.any():
        val[int(np.argmin(u))] = True
    if test_fraction > 0 and n >= 3 and not test.any():
        free = np.flatnonzero(~val)
        test[free[int(np.argmin(u[free]))]] = True
    train = ~val & ~test
    return np.flatnonzero(train), np.flatnonzero(val), np.flatnonzero(test)


class _Tensors:
    """The whole dataset on the training device once (inputs kept compact: uint8 grid, float16 probmap), so a batch
    is an index into it rather than a copy from the CPU. Augmentation happens per batch, on the device."""

    def __init__(self, data: Dataset, device: torch.device, scale: float = 1.0) -> None:
        self.scale = float(scale)                                 # labels are held divided by this
        self.grid = torch.from_numpy(data.grid).to(device)
        self.scalars = torch.from_numpy(data.scalars).to(device)
        self.probmap = torch.from_numpy(data.probmap).to(device)
        self.action = torch.from_numpy(data.action).to(device)
        self.label = torch.from_numpy(data.label / self.scale).to(device)
        self.weight = torch.from_numpy(data.weight).to(device)

    def batch(self, idx: torch.Tensor, quarter_turns: int = 0, flip: bool = False):
        grid = self.grid[idx].float()
        probmap = self.probmap[idx].float()
        if flip:
            grid, probmap = grid.flip(-1), probmap.flip(-1)
        if quarter_turns:
            grid = torch.rot90(grid, quarter_turns, dims=(2, 3))
            probmap = torch.rot90(probmap, quarter_turns, dims=(2, 3))
        return grid, self.scalars[idx], probmap, self.action[idx], self.label[idx], self.weight[idx]


def weighted_loss(output: torch.Tensor, action: torch.Tensor, label: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    taken = output.gather(1, action[:, None])[:, 0]
    return (weight * (taken - label) ** 2).sum() / weight.sum()


@torch.no_grad()
def evaluate(model: QNet, data: _Tensors, idx: np.ndarray, batch: int = 4096) -> float:
    """The weighted squared error on the taken states, in cells squared."""
    model.eval()
    total = weight_sum = 0.0
    index = torch.as_tensor(idx, device=data.label.device)
    for start in range(0, len(index), batch):
        g, s, c, a, y, w = data.batch(index[start:start + batch])
        out = model(g, s, c)
        taken = out.gather(1, a[:, None])[:, 0]
        total += float((w * (taken - y) ** 2).sum())
        weight_sum += float(w.sum())
    return total / max(weight_sum, 1e-12) * data.scale ** 2


def fit(model: QNet, data: _Tensors, train_idx: np.ndarray, val_idx: np.ndarray, *, epochs: int = 60, patience: int = 6,
        lr: float = 1e-3, weight_decay: float = 1e-4, batch: int = 256, augment: bool = True, seed: int = 0,
        keep_start: bool = False, checks: int = 1, echo=print) -> dict:
    """Train in place, keeping the point with the best validation loss. Returns the history. With `keep_start` the
    starting weights count as a candidate too (fine-tuning: never end worse than the network started). `checks` is how
    many times per pass the validation loss is measured (and the best point kept): a network that already fits its data
    can start to memorise within its first pass, so the useful point may be part of the way through it. Patience counts
    whole passes."""
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, since_best, history = math.inf, None, 0, []
    start_val = None
    checks = max(1, int(checks))
    if keep_start:
        start_val = best = evaluate(model, data, val_idx)
        best_state = _copy_state(model)
        if echo:
            echo(f"  start      val {start_val:9.3f}")
    train_index = torch.as_tensor(train_idx)
    stop = False
    for epoch in range(epochs):
        model.train()
        started = time.perf_counter()
        order = train_index[torch.randperm(len(train_index), generator=generator)].to(data.label.device)
        running = 0.0
        steps = 0
        starts = list(range(0, len(order), batch))
        cut = {starts[int(round(len(starts) * (i + 1) / checks)) - 1] for i in range(checks)} if starts else set()
        improved = False
        for start in starts:
            k, flip = (int(rng.integers(4)), bool(rng.integers(2))) if augment else (0, False)
            g, s, c, a, y, w = data.batch(order[start:start + batch], k, flip)
            loss = weighted_loss(model(g, s, c), a, y, w)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += float(loss.detach())
            steps += 1
            if start in cut:
                val = evaluate(model, data, val_idx)
                model.train()
                if val < best - 1e-9:
                    best, improved = val, True
                    best_state = _copy_state(model)
        train_loss = running / max(steps, 1) * data.scale ** 2
        history.append(dict(epoch=epoch, train=train_loss, val=val, best=best, seconds=time.perf_counter() - started))
        if echo:
            echo(f"  epoch {epoch:3d}  train {train_loss:9.3f}  val {val:9.3f}  best {best:9.3f}  ({history[-1]['seconds']:.1f}s)")
        if improved:
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                stop = True
        if stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return dict(best_val=best, epochs=len(history), history=history, start_val=start_val)


def calibration(pred: np.ndarray, action: np.ndarray, label: np.ndarray, bins: int = 5) -> dict:
    """Per state: the taken examples cut into `bins` equal-count bins by predicted value; mean prediction against the
    mean actual label in each. A well calibrated model has the two close in every bin."""
    out = {}
    for state in State:
        mask = action == int(state)
        if mask.sum() < bins:
            out[state.name] = dict(n=int(mask.sum()))
            continue
        p, y = pred[mask, int(state)], label[mask]
        order = np.argsort(p)
        chunks = np.array_split(order, bins)
        out[state.name] = dict(n=int(mask.sum()), bins=[
            dict(pred=float(p[c].mean()), actual=float(y[c].mean()), n=int(len(c))) for c in chunks])
    return out


def _prepare(dataset: Dataset, seed: int, device: str | None, echo, test_fraction: float = 0.0):
    """What every way of training needs: the device, the split by game (split_games: fixed by each game's id), the
    label mean and scale (labels are trained divided by their standard deviation on the training games), and the data
    on the device."""
    torch.manual_seed(seed)
    dev = pick_device(device)
    train_g, val_g, test_g = split_games(dataset.games, test_fraction=test_fraction)
    idx = dataset.subset(train_g), dataset.subset(val_g), dataset.subset(test_g)
    w = dataset.weight[idx[0]].astype(np.float64)
    mean = float((dataset.label[idx[0]] * w).sum() / w.sum())
    scale = max(1.0, float(np.sqrt((w * (dataset.label[idx[0]] - mean) ** 2).sum() / w.sum())))
    data = _Tensors(dataset, dev, scale)
    if echo:
        echo(f"split by game: train {len(train_g)} games / {len(idx[0])} frames, val {len(val_g)} / {len(idx[1])}, "
             f"test {len(test_g)} / {len(idx[2])}; device {dev}")
    return dev, (train_g, val_g, test_g), idx, mean, scale, data


def _finish(model: QNet, dataset: Dataset, data: _Tensors, games, idx, mean: float, scale: float, val: float,
            report: dict, echo) -> dict:
    """Test loss and calibration on the test games, and the last layer multiplied back so the network outputs cells."""
    train_g, val_g, test_g = games
    train_idx, val_idx, test_idx = idx
    test = evaluate(model, data, test_idx) if len(test_idx) else float("nan")
    w, y = dataset.weight[test_idx], dataset.label[test_idx]
    # no test split (the default: the online loop's real test is the gate) leaves the test numbers nan
    constant = float((w * (y - mean) ** 2).sum() / w.sum()) if len(test_idx) else float("nan")
    wv, yv = dataset.weight[val_idx], dataset.label[val_idx]
    val_constant = float((wv * (yv - mean) ** 2).sum() / wv.sum()) if len(val_idx) else float("nan")
    with torch.no_grad():                                         # from here on the network outputs cells
        model.output_layer.weight.mul_(scale)
        model.output_layer.bias.mul_(scale)
    pred = predict(model, dataset.grid[test_idx], dataset.scalars[test_idx], dataset.probmap[test_idx]) if len(test_idx) else None
    report.update(
        branches=list(model.branches), val=val, test=test, test_constant_baseline=constant,
        test_r2=1.0 - test / constant if constant > 0 else None,
        val_constant_baseline=val_constant, val_r2=1.0 - val / val_constant if val_constant > 0 else None,
        calibration=calibration(pred, dataset.action[test_idx], y) if len(test_idx) else {},
        choice_frequency={s.name: float((dataset.action == int(s)).mean()) for s in State},
        games=dict(train=len(train_g), val=len(val_g), test=len(test_g)),
        frames=dict(train=int(len(train_idx)), val=int(len(val_idx)), test=int(len(test_idx))),
        label_mean=float(dataset.label.mean()), label_std=float(dataset.label.std()),
    )
    if echo:
        on_test = report["test_r2"] is not None
        loss, base, r2 = (test, constant, report["test_r2"]) if on_test else (val, val_constant, report["val_r2"])
        echo(f"branches {'+'.join(model.branches)}: {'test' if on_test else 'validation'} loss {loss:.3f} vs constant "
             f"baseline {base:.3f} (R^2 {r2 if r2 is not None else float('nan'):.3f})")
    return report


def staged(dataset: Dataset, *, seed: int = 0, keep: float = 0.02, device: str | None = None, echo=print,
           stages=(("scalars",), ("scalars", "grid"), ("scalars", "grid", "probmap")), test_fraction: float = 0.0,
           **fit_args) -> tuple[QNet, dict]:
    """The brief's staged build, from scratch. Returns the chosen model and a report."""
    dev, games, idx, mean, scale, data = _prepare(dataset, seed, device, echo, test_fraction)
    train_idx, val_idx, _ = idx

    model = QNet(stages[0]).to(dev)
    with torch.no_grad():                                         # start the outputs at the mean label
        model.output_layer.bias.fill_(mean / scale)
    report = dict(stages=[], device=str(dev), label_scale=scale)
    if echo:
        echo(f"stage {'+'.join(stages[0])}")
    history = fit(model, data, train_idx, val_idx, seed=seed, echo=echo, **fit_args)
    chosen, chosen_val = tuple(stages[0]), history["best_val"]
    chosen_state = _copy_state(model)
    report["stages"].append(dict(branches=list(chosen), best_val=chosen_val, epochs=history["epochs"], kept=True,
                                 history=history["history"]))
    for branches in stages[1:]:
        if echo:
            echo(f"stage {'+'.join(branches)} (and a control: {'+'.join(chosen)} trained for as long)")
        model.load_state_dict(chosen_state)
        model.set_branches(branches)
        trial = fit(model, data, train_idx, val_idx, seed=seed, echo=echo, **fit_args)       # with the new branch
        trial_state = _copy_state(model)
        model.load_state_dict(chosen_state)
        model.set_branches(chosen)
        control = fit(model, data, train_idx, val_idx, seed=seed, echo=echo, **fit_args)     # without it, as long
        control_state = _copy_state(model)
        kept = trial["best_val"] < control["best_val"] * (1.0 - keep)
        report["stages"].append(dict(branches=list(branches), best_val=trial["best_val"], control_val=control["best_val"],
                                     epochs=trial["epochs"], kept=kept, history=trial["history"]))
        if echo:
            echo(f"stage {'+'.join(branches)}: validation loss {trial['best_val']:.3f} against the control's "
                 f"{control['best_val']:.3f} -> {'kept' if kept else 'not kept'}")
        if kept:
            chosen, chosen_val, chosen_state = tuple(branches), trial["best_val"], trial_state
        else:
            chosen_val, chosen_state = control["best_val"], control_state
    model.load_state_dict(chosen_state)
    model.set_branches(chosen)
    return model, _finish(model, dataset, data, games, idx, mean, scale, chosen_val, report, echo)


def full(dataset: Dataset, *, seed: int = 0, device: str | None = None, echo=print, branches=BRANCHES,
         test_fraction: float = 0.0, width: int = 1, **fit_args) -> tuple[QNet, dict]:
    """A network with all its branches trained together from scratch (no stage can drop a branch). Returns the model
    and a report like staged()'s."""
    dev, games, idx, mean, scale, data = _prepare(dataset, seed, device, echo, test_fraction)
    train_idx, val_idx, _ = idx
    model = QNet(branches, width=width).to(dev)
    with torch.no_grad():                                         # start the outputs at the mean label
        model.output_layer.bias.fill_(mean / scale)
    if echo:
        echo(f"from scratch, {'+'.join(model.branches)}, width {width} ({sum(p.numel() for p in model.parameters()):,} parameters)")
    history = fit(model, data, train_idx, val_idx, seed=seed, echo=echo, **fit_args)
    report = dict(device=str(dev), label_scale=scale, stages=[
        dict(branches=list(model.branches), best_val=history["best_val"], epochs=history["epochs"], kept=True,
             history=history["history"])])
    return model, _finish(model, dataset, data, games, idx, mean, scale, history["best_val"], report, echo)


def add_branches(model: QNet, branches=BRANCHES) -> list[str]:
    """Switch on the branches of `branches` the model does not use yet, without changing a single output: the head's
    weights from a newly switched-on branch start at zero (a switched-off branch fed it zeros, so those weights never
    trained) and learn from there. Returns the branches added."""
    added = [b for b in BRANCHES if b in branches and b not in model.branches]
    if added:
        first = model.head[0]
        offsets = np.cumsum([0] + [model.widths[b] for b in BRANCHES])
        with torch.no_grad():
            for b in added:
                i = BRANCHES.index(b)
                first.weight[:, offsets[i]:offsets[i + 1]] = 0.0
        model.set_branches(tuple(model.branches) + tuple(added))
    return added


def finetune(model: QNet, dataset: Dataset, *, seed: int = 0, device: str | None = None, echo=print,
             branches=None, test_fraction: float = 0.0, **fit_args) -> tuple[QNet, dict]:
    """Train an existing network (which outputs cells) further on `dataset`; with `branches`, those it lacks are
    switched on first (add_branches: the starting outputs stay the same). Returns the model (outputting cells again)
    and a report like staged()'s, with the starting weights' validation loss."""
    dev, games, idx, mean, scale, data = _prepare(dataset, seed, device, echo, test_fraction)
    train_idx, val_idx, _ = idx
    if branches is not None:
        added = add_branches(model, branches)
        if added and echo:
            echo(f"switched on {'+'.join(added)} (head weights from them start at zero: same outputs to begin with)")
    model = model.to(dev)
    with torch.no_grad():                                         # training works on labels / scale
        model.output_layer.weight.div_(scale)
        model.output_layer.bias.div_(scale)
    if echo:
        echo(f"fine-tune {'+'.join(model.branches)}")
    history = fit(model, data, train_idx, val_idx, seed=seed, echo=echo, keep_start=True, **fit_args)
    report = dict(device=str(dev), label_scale=scale, start_val=history["start_val"], stages=[
        dict(branches=list(model.branches), best_val=history["best_val"], start_val=history["start_val"],
             epochs=history["epochs"], kept=True, history=history["history"])])
    return model, _finish(model, dataset, data, games, idx, mean, scale, history["best_val"], report, echo)


def export(model: QNet, path: str | Path, report: dict, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = dict(extra or {})
    meta = dict(extra, name=extra.get("name", path.stem), branches=list(model.branches), inputs_version=inputs_version(),
                created=time.strftime("%Y-%m-%dT%H:%M:%S"), val=report.get("val"), test=report.get("test"),
                test_r2=report.get("test_r2"), games=report.get("games"), frames=report.get("frames"))
    save_network(model, path, meta)
    summary = dict(report)
    for stage in summary.get("stages", []):
        stage.pop("history", None)
    path.with_suffix(".json").write_text(json.dumps(dict(meta=meta, report=summary), indent=1), encoding="utf-8")
    return path


def smoke_subset(dataset: Dataset, games: int = 6, frames_per_game: int = 200) -> Dataset:
    """A small slice of a dataset (the first frames of a few games) for checking that training runs end to end."""
    keep_games = list(range(min(games, len(dataset.games))))
    idx = np.concatenate([np.flatnonzero(dataset.game == g)[:frames_per_game] for g in keep_games])
    arrays = {k: getattr(dataset, k)[idx] for k in Dataset._ARRAYS}
    return Dataset(**arrays, games=[dataset.games[g] for g in keep_games],
                   game_meta=[dataset.game_meta[g] for g in keep_games], meta=dict(dataset.meta, smoke=True))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="a training set built by candidate.offline.dataset (name or path)")
    source.add_argument("--raw", nargs="+", help="raw game files or directories, built into a training set on the fly")
    parser.add_argument("--name", default=None, help="weights name (q_training/weights/<name>.pt) or a path")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", type=float, default=0.02)
    parser.add_argument("--test-share", type=float, default=0.0, help="share of the games held out as a test set")
    parser.add_argument("--device", default=None, help="cuda or cpu; default: the GPU if it really works, else the CPU")
    parser.add_argument("--stale", choices=("replay", "skip", "use"), default="replay")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="check the pipeline: a few games, one epoch per stage")
    args = parser.parse_args(argv)

    dataset = Dataset.load(args.dataset) if args.dataset else load_dataset(args.raw, stale=args.stale)
    epochs, patience = args.epochs, args.patience
    if args.smoke:
        dataset, epochs, patience = smoke_subset(dataset), 1, 1
    name = args.name or ("smoke" if args.smoke else f"q_{Path(args.dataset or args.raw[0]).stem}")
    model, report = staged(dataset, seed=args.seed, keep=args.keep, device=args.device, epochs=epochs,
                           patience=patience, batch=args.batch, augment=not args.no_augment,
                           test_fraction=args.test_share)
    source_names = [args.dataset] if args.dataset else [str(p) for p in args.raw]
    path = export(model, locations.weights_path(name), report, dict(data=source_names, smoke=args.smoke))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
