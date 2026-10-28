"""Training the belief's motion model K (agent/belief/neural.py) on what opponents really did.

    python -m candidate.offline.k_train --raw q_training/online/online1/epoch_005/raw --out k.npz [--start old_k.npz]

The online loop (online.py) calls this every epoch, so K and Q are trained together; this command does the same step
on its own.

The data. A recorded game's two move tapes replay it exactly with our own rules (no engine needed, checked against the
engine's final counts; motion_fit.replay_boards). Before every opponent move, its true situation is put into a
one-hypothesis belief and classified by the belief's own code, and `neural.features` turns that into exactly the
numbers the belief computes in play. A row is those features for its four moves, which of them are legal, and the move
it actually made. Only games against the enemy state machines and our own agent count (K_OPPONENTS): those are the
opponents the belief has to model; the basic and random examples and the scripted hunter would teach it other habits.

The fit. The network outputs a score per move; the belief turns scores into probabilities as a softmax over the legal
moves mixed with the floor epsilon (BeliefConfig), and training maximises the probability of the move made under
exactly that model. Every opponent type weighs the same, and every game within a type. Split by game (85 / 15),
early stopping on the validation games, Adam. Starting from the previous K when there is one (it is trained further,
not from scratch). The report gives, per opponent type on the validation games, the mean log-likelihood of the move
made and how often K's most likely move was it, for the trained K, the hand-set K and a uniform guess.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

K_OPPONENTS = ("enemy", "enemy_copy", "noisy", "self", "champion")


def game_rows(sidecar_path: str) -> dict:
    """The rows of one game. Runs in a worker process."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    from candidate.agent.belief import neural
    from candidate.agent.belief.generate import Situation, classify
    from candidate.agent.data.hypotheses import HypothesisStore
    from candidate.agent.data.territory import TerritoryDiffs
    from candidate.agent.geometry import bitboard
    from candidate.offline.motion_fit import LETTERS, replay_boards

    sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    tapes, opp_seat = sidecar["tapes"], 1 - int(sidecar["config"]["seat"])
    feats, legal, label = [], [], []
    try:
        for turn, w, rules, opp_last in replay_boards(sidecar, want_opponent_rows=True):
            opp_land, opp_trail, my_trail, my_land = w.opp_territory[0], w.opp_trail[0], w.my_trail[0], w.my_territory[0]
            n = int(bitboard.popcount(opp_trail))
            store = HypothesisStore.build(
                pos=[int(w.opp_pos[0])], last_dir=[opp_last if n else -1], trail_len=[n], trail_bb=opp_trail[None, :],
                opp_land_id=[0], my_land_id=[0], opp_count=[int(bitboard.popcount(opp_land))], flags=[0], weight=[1.0],
                hash=[0])
            sit = Situation(rules, None, opp_land, TerritoryDiffs(), int(w.my_pos[0]), my_trail, my_land, TerritoryDiffs(),
                            int(bitboard.popcount(my_land)))
            batch = classify(SimpleNamespace(store=store), sit).batch
            feats.append(neural.features(batch)[0])
            legal.append(batch.legal[0])
            label.append(LETTERS.index(tapes[opp_seat][turn]))
    except ValueError as error:
        return dict(path=sidecar_path, error=str(error))
    return dict(path=sidecar_path, opponent=sidecar["config"]["opponent"], features=np.asarray(feats, dtype=np.float32),
                legal=np.asarray(legal, dtype=bool), label=np.asarray(label, dtype=np.int8))


def sidecars(dirs, opponents=K_OPPONENTS) -> list[str]:
    """The recorded games under `dirs` against the given opponents (a sidecar next to its record)."""
    out = []
    for d in dirs:
        for path in sorted(Path(d).rglob("*.json")):
            if not path.with_suffix(".npz").exists() or path.parent.name.startswith("replayed-"):
                continue
            try:
                if json.loads(path.read_text(encoding="utf-8"))["config"]["opponent"] in opponents:
                    out.append(str(path))
            except (KeyError, ValueError):
                continue
    return out


def build(dirs, workers: int = 16, echo=print) -> dict:
    """All the rows of the games under `dirs` against K_OPPONENTS, as arrays (and per-row game and opponent)."""
    paths = sidecars(dirs)
    if not paths:
        raise ValueError(f"no games against {K_OPPONENTS} under {list(map(str, dirs))}")
    if workers <= 1:
        results = [game_rows(p) for p in paths]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(game_rows, paths, chunksize=2))
    good = [r for r in results if "error" not in r and len(r["label"])]
    if echo:
        echo(f"[K] {sum(len(r['label']) for r in good)} opponent moves from {len(good)} games "
             f"({len(results) - len(good)} could not be replayed)")
    return dict(
        features=np.concatenate([r["features"] for r in good]), legal=np.concatenate([r["legal"] for r in good]),
        label=np.concatenate([r["label"] for r in good]).astype(np.int64),
        game=np.concatenate([np.full(len(r["label"]), i) for i, r in enumerate(good)]),
        opponent=np.concatenate([np.full(len(r["label"]), r["opponent"]) for r in good]),
    )


def _weights(data: dict, rows: np.ndarray) -> np.ndarray:
    """Every opponent type the same total, every game within a type the same."""
    w = np.zeros(len(rows))
    types = np.unique(data["opponent"][rows])
    for t in types:
        mine = rows[data["opponent"][rows] == t]
        games, counts = np.unique(data["game"][mine], return_counts=True)
        per = dict(zip(games.tolist(), counts.tolist()))
        w[np.isin(rows, mine)] = [1.0 / (len(games) * per[g]) for g in data["game"][mine]]
    return w / len(types)


def _probabilities(scores: np.ndarray, legal: np.ndarray, eps: float) -> np.ndarray:
    from candidate.agent.belief.motion import move_probabilities

    return move_probabilities(scores, legal, 1.0, eps)


def _hand_scores(f: np.ndarray) -> np.ndarray:
    """The hand-set K (FeatureMotionModel with its default weights) from the same features."""
    from candidate.agent.belief.motion import MotionWeights

    w = MotionWeights()
    return (w.self_trail * f[..., 0] + w.kills_my_trail * f[..., 1] + w.capture * f[..., 2] * 5.0
            + w.home_pull * f[..., 3] * f[..., 4] + w.chase * f[..., 5] + w.momentum * f[..., 6] + w.explore * f[..., 7])


def report(data: dict, rows: np.ndarray, model, eps: float) -> dict:
    out = {}
    f, legal, label = data["features"][rows], data["legal"][rows], data["label"][rows]
    for name in list(np.unique(data["opponent"][rows])) + ["all"]:
        m = np.ones(len(rows), bool) if name == "all" else data["opponent"][rows] == name
        entry = {}
        for kind, scores in (("trained", model.scores_of(f[m])), ("hand", _hand_scores(f[m])),
                             ("uniform", np.zeros(legal[m].shape))):
            p = _probabilities(scores, legal[m], eps)
            made = p[np.arange(m.sum()), label[m]]
            entry[kind] = dict(log_likelihood=float(np.mean(np.log(np.maximum(made, 1e-300)))),
                               accuracy=float(np.mean(p.argmax(axis=1) == label[m])))
        entry["moves"] = int(m.sum())
        out[str(name)] = entry
    return out


def fit(data: dict, start=None, epochs: int = 30, patience: int = 4, lr: float = 1e-3, batch: int = 4096,
        seed: int = 0, device: str | None = None, echo=print):
    """Train K. Returns (NeuralMotionModel, report)."""
    import torch

    from candidate.agent.belief.config import BeliefConfig
    from candidate.agent.belief.neural import NeuralMotionModel
    from candidate.agent.q.network import pick_device

    eps = BeliefConfig().epsilon
    dev = pick_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    games = np.unique(data["game"])
    val_games = []
    for t in np.unique(data["opponent"]):                                # 15% of each type's games for validation
        of_type = np.unique(data["game"][data["opponent"] == t])
        val_games += list(rng.permutation(of_type)[: max(1, int(round(0.15 * len(of_type))))]) if len(of_type) > 1 else []
    val_mask = np.isin(data["game"], val_games)
    train_rows, val_rows = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    if not len(val_rows):                                                # too few games: validate on the training rows
        val_rows = train_rows

    start = start if start is not None else NeuralMotionModel.initial(seed)
    layers = torch.nn.ModuleList(torch.nn.Linear(w.shape[0], w.shape[1]) for w, _ in start.layers).to(dev)
    with torch.no_grad():
        for lin, (w, b) in zip(layers, start.layers):
            lin.weight.copy_(torch.as_tensor(w.T))
            lin.bias.copy_(torch.as_tensor(b))

    def scores(x):
        for i, lin in enumerate(layers):
            x = lin(x)
            if i < len(layers) - 1:
                x = torch.relu(x)
        return x.squeeze(-1)

    tensors = {k: torch.as_tensor(data[k]).to(dev) for k in ("features", "legal", "label")}
    w_train = torch.as_tensor(_weights(data, train_rows), dtype=torch.float32, device=dev)
    w_val = torch.as_tensor(_weights(data, val_rows), dtype=torch.float32, device=dev)

    def loss_of(rows, weight):
        f, legal, label = tensors["features"][rows], tensors["legal"][rows], tensors["label"][rows]
        logits = torch.where(legal, scores(f), torch.tensor(-1e9, device=dev))
        soft = torch.softmax(logits, dim=1).gather(1, label[:, None])[:, 0]
        n_legal = legal.sum(dim=1).clamp(min=1).float()
        p = (1 - eps) * soft + eps / n_legal
        return -(weight * torch.log(p)).sum() / weight.sum()

    optimiser = torch.optim.Adam(layers.parameters(), lr=lr)
    train_t = torch.as_tensor(train_rows, device=dev)
    val_t = torch.as_tensor(val_rows, device=dev)
    with torch.no_grad():
        best = float(loss_of(val_t, w_val))
    best_state = [(lin.weight.detach().clone(), lin.bias.detach().clone()) for lin in layers]
    start_val, since, history = best, 0, []
    for epoch in range(epochs):
        order = torch.randperm(len(train_t), device=dev)
        for i in range(0, len(order), batch):
            pick = order[i:i + batch]
            loss = loss_of(train_t[pick], w_train[pick])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            val = float(loss_of(val_t, w_val))
        history.append(val)
        if echo:
            echo(f"  [K] epoch {epoch:2d}  validation NLL {val:.4f}")
        if val < best - 1e-6:
            best, since = val, 0
            best_state = [(lin.weight.detach().clone(), lin.bias.detach().clone()) for lin in layers]
        else:
            since += 1
            if since >= patience:
                break
    model = NeuralMotionModel([(w.T.cpu().numpy(), b.cpu().numpy()) for w, b in best_state],
                              dict(name="neural K", trained=time.strftime("%Y-%m-%dT%H:%M:%S")))
    rep = dict(device=str(dev), moves=int(len(data["label"])), games=int(len(games)), start_val_nll=start_val,
               val_nll=best, epochs=len(history), validation=report(data, val_rows, model, eps))
    model.meta.update(val_nll=best, games=int(len(games)), opponents=sorted(map(str, np.unique(data["opponent"]))))
    return model, rep


def format_report(rep: dict) -> str:
    lines = [f"[K] validation, mean log-likelihood of the move made / how often K's top move was it "
             f"(start {rep['start_val_nll']:.4f} -> {rep['val_nll']:.4f} NLL):"]
    for name, e in rep["validation"].items():
        cell = lambda m: f"{m['log_likelihood']:7.3f} / {m['accuracy']:5.1%}"  # noqa: E731
        lines.append(f"  {name:10s} trained {cell(e['trained'])}   hand-set {cell(e['hand'])}   uniform {cell(e['uniform'])}"
                     f"   ({e['moves']} moves)")
    return "\n".join(lines)


def train_step(dirs, out: Path, start_path: str | None = None, workers: int = 16, epochs: int = 30, patience: int = 4,
               device: str | None = None, seed: int = 0, echo=print) -> dict:
    """Rows from `dirs`, K trained (further, from `start_path` if given), saved to `out`. Returns the report."""
    from candidate.agent.belief.neural import NeuralMotionModel

    data = build(dirs, workers, echo)
    start = NeuralMotionModel.load(start_path) if start_path else None
    model, rep = fit(data, start, epochs=epochs, patience=patience, device=device, seed=seed, echo=None)
    model.meta.update(trained_from=str(start_path) if start_path else None, data=[str(d) for d in dirs])
    model.save(out)
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    if echo:
        echo(format_report(rep))
    return rep


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", nargs="+", required=True, help="directories of recorded games")
    parser.add_argument("--out", required=True, help="the K file to write (.npz)")
    parser.add_argument("--start", default=None, help="a K file to train further")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    train_step(args.raw, Path(args.out), args.start, args.workers, args.epochs, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())

