"""The data handler: from recorded games (q_training/raw) to training sets (q_training/datasets).

    python -m candidate.offline.dataset build --raw q_training/raw/round0b --name round0b
    python -m candidate.offline.dataset info round0b

Every frame of a recorded game is one example (brief, "Labels and training"): Q's three inputs as seen in play, the
state taken after that frame's decision, and the label

    y_t = (score_diff[t + N] - score_diff[t])  +  D5[t + N] * [D7[t + N] > 0]

where score_diff = my count - the enemy's, N = 50 of my frames, and the second term credits a loop still open at the
end of the window, but only if it was safe then (D5 and D7 are the raw values, cells and turns). Near the end of the
game the window stops at the final turn: y_t = final_diff - score_diff[t], with no open-loop credit, because a trail
still out when the game ends is worth nothing. The final difference comes from the engine (the sidecar) when the
game was collected headless; for a game recorded in the GUI it is the difference at my last frame. A game that was
not recorded to its end only labels the frames whose window it contains.

Weighted by distance (QConfig.label_horizons, label_weights; the online loop's default): the window is cut into
segments, by default frames 1-20, 21-50 and 51-100, weighted 1.0, 0.6 and 0.3, and the value at the end of the window
(the open-loop credit, or the bootstrapped value) takes the last weight:

    y_t = c * ( sum_j w_j * (score_diff[t + n_j] - score_diff[t + n_(j-1)])  +  w_last * tail[t + N] )

What happens soon after a choice depends mostly on that choice; what happens 80 frames later depends on much else, so
weighting it less takes noise out of the label. c = N / sum_j w_j (n_j - n_(j-1)) keeps the intensity: a steady gain of
g cells a frame still gives N g, so labels stay in cells per window and the chooser's switching costs keep their
meaning. Equal weights give the unweighted label exactly. A point past the game's end takes the final difference.

Bootstrapped labels (the brief's "later improvement", step 10): with a previous round's network, the open-loop credit
at the end of the window is replaced by that network's value of the state taken at t + N, which credits TRACK TARGET
and PATROL for information that pays off after more than N turns.

Frame stride: with frame_stride k only every k-th labelled frame of a game is kept (labels are computed on the
whole game first, so each kept frame's label still looks over the full window). Consecutive frames are nearly the same
board, so k = 2 loses little information and halves the memory and the training time.

Weighting: every game counts equally, so each frame weighs 1 / (labelled frames of its game), scaled so the mean
weight is 1. Optionally (balance_belief, off by default), when a set holds both games played with the real belief and
games given the true world (offline/oracle.py), the two kinds are balanced to the same total weight. Tried and not
adopted: the round-17 network, with the real belief, stayed home all game; fine-tuned once on round 17's games with the
100-frame bootstrapped labels it went out (territory 41-266 against the enemy copy, real belief, 600 turns), but with
the belief games balanced it stayed home again (9-18): the bootstrap term in those games is that network's own
pessimistic value of the belief case, and weighting them four times over amplified it.

Stale inputs: a record made by older code (its inputs_version differs from version.inputs_version()) is replayed to
regenerate the inputs (replay.py) when its sidecar has the move tapes; otherwise it is skipped, unless stale="use".

A built dataset is saved as one .npz (inputs kept compact: the grid as uint8, the probability map as float16) with a JSON
`meta`: the games it holds, where they came from, the inputs version, the label window and how it was labelled.
"""

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from candidate.agent.q.config import Q
from candidate.agent.q.record import load_record, unpack_grid
from candidate.agent.behavior.states import State
from candidate.agent.q.version import inputs_version
from candidate.offline import locations


@dataclass
class Game:
    path: Path
    meta: dict
    arrays: dict
    sidecar: dict | None = None

    @property
    def complete(self) -> bool:
        return bool(self.meta.get("complete", False))

    @property
    def stale(self) -> bool:
        return self.meta.get("inputs_version") != inputs_version()

    def final_diff(self) -> float | None:
        """My final count minus the enemy's: from the engine if known, else my last frame's if the game is complete."""
        if self.sidecar is not None and "final" in self.sidecar:
            seat = self.sidecar["config"]["seat"]
            counts = self.sidecar["final"]["counts"]
            return float(counts[seat] - counts[1 - seat])
        if self.complete:
            a = self.arrays
            return float(a["my_count"][-1] - a["opp_count"][-1])
        return None


# ---- finding and loading raw games ------------------------------------------------------------------------------------


def find_records(paths) -> list[Path]:
    """Every game record under the given files or directories (searched recursively). Records regenerated by a replay
    (in a replayed-* directory) are not listed: they stand in for their originals when those are stale."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(q for q in p.rglob("*.npz") if not any(part.startswith("replayed-") for part in q.parts)))
        elif p.suffix == ".npz":
            out.append(p)
    return out


def load_game(path: Path) -> Game:
    meta, arrays = load_record(path)
    side = path.with_suffix(".json")
    sidecar = json.loads(side.read_text(encoding="utf-8")) if side.exists() else None
    return Game(path, meta, arrays, sidecar)


def fresh_game(path: Path, stale: str = "replay", replay_dir: str | Path | None = None) -> tuple[Game | None, str]:
    """The game at `path` with inputs valid for the current code: as recorded, or regenerated by a replay. Returns the
    game (None if it has to be skipped) and what was done: "ok", "replayed", "used stale", or "skipped"."""
    game = load_game(path)
    if not game.stale:
        return game, "ok"
    if stale == "use":
        return game, "used stale"
    if stale == "replay" and game.sidecar is not None and "tapes" in game.sidecar:
        from candidate.offline.replay import replay

        target = Path(replay_dir) if replay_dir is not None else path.parent / f"replayed-{inputs_version()}"
        label = game.meta.get("label", path.stem)
        existing = sorted(target.glob(f"*{label}-replay.npz")) if target.exists() else []
        return load_game(existing[-1] if existing else replay(path, target)), "replayed"
    return None, "skipped"


# ---- labels ------------------------------------------------------------------------------------------------------------


def label_weights(window: int, horizons=Q.label_horizons, weights=Q.label_weights) -> tuple[list[int], list[float]]:
    """The segments' end points (ending with `window`) and their weights, scaled so that a steady gain per frame gives
    the unweighted label (see the module notes)."""
    ends = [int(h) for h in horizons if 0 < int(h) < window] + [int(window)]
    w = [float(x) for x in weights]
    if len(w) != len(horizons) + 1:
        raise ValueError(f"{len(horizons)} horizons {list(horizons)} need {len(horizons) + 1} weights, got {w}")
    w = w[:len(ends)]                                 # a shorter window drops the segments past it (and their weights)
    starts = [0] + ends[:-1]
    c = window / sum(wi * (e - s) for wi, s, e in zip(w, starts, ends))
    return ends, [c * wi for wi in w]


def labels(game: Game, window: int = Q.label_window, bootstrap=None, horizons=Q.label_horizons,
           weights=Q.label_weights) -> tuple[np.ndarray, np.ndarray]:
    """(y, has_label): the label of every frame, and whether it has one. `bootstrap` (a trained QNet) replaces the
    open-loop credit at the end of the window by its value of the state taken there. The segments are weighted by
    distance (horizons, weights; see the module notes)."""
    a = game.arrays
    n = len(a["turn"])
    diff = (a["my_count"].astype(np.float64) - a["opp_count"].astype(np.float64))
    if bootstrap is None:
        tail_value = np.where(a["d7"] > 0, a["d5"].astype(np.float64), 0.0)
    else:
        tail_value = network_values(bootstrap, a)[np.arange(n), a["state"].astype(np.int64)]
    ends, scaled = label_weights(window, horizons, weights)
    final = game.final_diff()
    t = np.arange(n)

    def at(s: int) -> np.ndarray:                     # the difference s frames on; past the game's end, the final one
        j = t + s
        past = final if final is not None else np.nan
        return np.where(j < n, diff[np.minimum(j, n - 1)], past)

    y = np.zeros(n, dtype=np.float64)
    before = diff
    for end, w in zip(ends, scaled):
        now = at(end)
        y += w * (now - before)
        before = now
    inside = t + window < n
    y[inside] += scaled[-1] * tail_value[(t + window)[inside]]
    has = inside | (final is not None)
    y[~has] = 0.0
    return y.astype(np.float32), has


def network_values(network, arrays: dict) -> np.ndarray:
    """(n, 6): a network's values of every recorded frame."""
    from candidate.agent.q.network import predict

    return predict(network, unpack_grid(arrays["grid_bits"]), arrays["scalars"], arrays["probmap"])


# ---- building, saving, loading ----------------------------------------------------------------------------------------


@dataclass
class Dataset:
    grid: np.ndarray            # (N, 7, 21, 21) uint8
    scalars: np.ndarray         # (N, 22) float32
    probmap: np.ndarray      # (N, 10, 10, 10) float16
    action: np.ndarray          # (N,) int64: the state taken
    label: np.ndarray           # (N,) float32
    game: np.ndarray            # (N,) int32: index into `games`
    weight: np.ndarray          # (N,) float32: every game counts equally, mean 1
    explored: np.ndarray        # (N,) bool
    games: list[str] = field(default_factory=list)
    game_meta: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    _ARRAYS = ("grid", "scalars", "probmap", "action", "label", "game", "weight", "explored")

    def __len__(self) -> int:
        return int(self.label.shape[0])

    def subset(self, game_ids) -> np.ndarray:
        """Indices of the examples that belong to the given games."""
        return np.flatnonzero(np.isin(self.game, np.asarray(list(game_ids))))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta, games=self.games, game_meta=self.game_meta)
        np.savez(path, meta=np.array(json.dumps(meta)), **{k: getattr(self, k) for k in self._ARRAYS})
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        with np.load(locations.dataset_path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            arrays = {k: data[k] for k in cls._ARRAYS}
        games, game_meta = meta.pop("games"), meta.pop("game_meta")
        return cls(**arrays, games=games, game_meta=game_meta, meta=meta)

    def summary(self) -> dict:
        """What is in it: sizes, the labels, how often each state was taken, and the games by opponent."""
        opponents = Counter(m.get("opponent", m.get("label", "?")) for m in self.game_meta)
        return dict(
            games=len(self.games), frames=len(self), label_mean=float(self.label.mean()),
            label_std=float(self.label.std()), label_min=float(self.label.min()), label_max=float(self.label.max()),
            explored_share=float(self.explored.mean()),
            state_share={s.name: float((self.action == int(s)).mean()) for s in State},
            opponents=dict(opponents), oracle_games=sum(bool(m.get("oracle")) for m in self.game_meta),
            inputs_version=self.meta.get("inputs_version"),
            stale_inputs=self.meta.get("inputs_version") != inputs_version(),
        )


def build(games: list[Game], window: int = Q.label_window, bootstrap=None, balance_belief: bool = False,
          frame_stride: int = 1, horizons=Q.label_horizons, weights=Q.label_weights) -> Dataset:
    parts = {k: [] for k in Dataset._ARRAYS}
    names, metas = [], []
    for game in games:
        y, has = labels(game, window, bootstrap, horizons, weights)
        warmup = int(game.meta.get("warmup_turns") or 0)
        if warmup:                                   # turns another agent played (a replay records them too): not Q's
            has = has & (game.arrays["turn"] >= warmup)
        if frame_stride > 1:
            has = has & (np.arange(has.size) % frame_stride == 0)
        k = int(has.sum())
        if k == 0:
            continue
        index = len(names)
        names.append(game.meta.get("game_id", game.path.stem))
        summary = {key: game.sidecar.get(key) for key in ("my_count", "opp_count", "won")} if game.sidecar else None
        metas.append(dict(game.meta, sidecar_summary=summary, source=str(game.path)))
        a = game.arrays
        parts["grid"].append(unpack_grid(a["grid_bits"][has]))
        parts["scalars"].append(a["scalars"][has].astype(np.float32))
        parts["probmap"].append(a["probmap"][has].astype(np.float16))
        parts["action"].append(a["state"][has].astype(np.int64))
        parts["label"].append(y[has])
        parts["game"].append(np.full(k, index, dtype=np.int32))
        parts["weight"].append(np.full(k, 1.0 / k, dtype=np.float64))
        parts.setdefault("_oracle", []).append(np.full(k, bool(game.meta.get("oracle")), dtype=bool))
        parts["explored"].append(a["explored"][has].astype(bool))
    if not names:
        raise ValueError("no labelled frames in these games")
    joined = {k: np.concatenate(v) for k, v in parts.items()}
    oracle = joined.pop("_oracle")
    weight = joined["weight"]
    balanced = balance_belief and oracle.any() and (~oracle).any()
    if balanced:                                                     # belief games: the same total weight as the others
        weight = weight.copy()
        weight[~oracle] *= weight[oracle].sum() / weight[~oracle].sum()
    joined["weight"] = (weight * len(weight) / weight.sum()).astype(np.float32)
    meta = dict(inputs_version=inputs_version(), label_window=window, belief_balanced=bool(balanced), frame_stride=frame_stride,
                label_horizons=list(horizons), label_weights=list(weights),
                bootstrap=getattr(bootstrap, "meta", {}).get("name") if bootstrap is not None else None,
                created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return Dataset(**joined, games=names, game_meta=metas, meta=meta)


def load_dataset(paths, stale: str = "replay", replay_dir: str | Path | None = None, window: int = Q.label_window,
                 complete_only: bool = False, bootstrap=None, balance_belief: bool = False, frame_stride: int = 1,
                 horizons=Q.label_horizons, weights=Q.label_weights, echo=print) -> Dataset:
    """Every recorded game under `paths` (files or directories), as one Dataset. stale: "replay" (regenerate from the
    tapes, skipping a stale record that has none), "skip", or "use" (train on the old inputs anyway)."""
    games: list[Game] = []
    done = Counter()
    for path in find_records(paths):
        game, what = fresh_game(path, stale, replay_dir)
        if game is not None and complete_only and not game.complete:
            game, what = None, "skipped"
        done[what] += 1
        if game is not None:
            games.append(game)
    if done["skipped"]:
        warnings.warn(f"skipped {done['skipped']} recorded games (stale inputs with no tape, or incomplete)")
    dataset = build(games, window, bootstrap, balance_belief, frame_stride, horizons, weights)
    dataset.meta.update(sources=[str(p) for p in paths], handling=dict(done))
    if echo:
        echo(f"dataset: {len(dataset.games)} games, {len(dataset)} frames ({dict(done)})")
    return dataset


# ---- the command line ----------------------------------------------------------------------------------------------------


def _print_summary(name: str, data: Dataset) -> None:
    s = data.summary()
    print(f"{name}: {s['games']} games, {s['frames']} frames; labels mean {s['label_mean']:.1f} std {s['label_std']:.1f} "
          f"range [{s['label_min']:.0f}, {s['label_max']:.0f}]; explored {100 * s['explored_share']:.1f}% of frames")
    print("  states taken: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in s["state_share"].items()))
    print("  games: " + ", ".join(f"{k} {v}" for k, v in sorted(s["opponents"].items())))
    print(f"  inputs version {s['inputs_version']}" + ("  (STALE: built by older code)" if s["stale_inputs"] else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build or inspect training sets from recorded games.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build a training set from raw games and save it in q_training/datasets")
    b.add_argument("--raw", nargs="+", required=True, help="raw game files or directories (searched recursively)")
    b.add_argument("--name", required=True, help="dataset name (q_training/datasets/<name>.npz) or a path")
    b.add_argument("--stale", choices=("replay", "skip", "use"), default="replay")
    b.add_argument("--window", type=int, default=Q.label_window)
    b.add_argument("--complete-only", action="store_true")
    b.add_argument("--bootstrap", default=None, help="weights of a previous round: bootstrapped labels")
    i = sub.add_parser("info", help="describe a saved training set")
    i.add_argument("name")
    args = parser.parse_args(argv)
    if args.command == "build":
        bootstrap = None
        if args.bootstrap:
            from candidate.agent.q.network import load_network

            bootstrap = load_network(locations.weights_path(args.bootstrap))
        data = load_dataset(args.raw, stale=args.stale, window=args.window, complete_only=args.complete_only,
                            bootstrap=bootstrap)
        path = data.save(locations.dataset_path(args.name))
        _print_summary(str(path), data)
    else:
        _print_summary(args.name, Dataset.load(args.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
