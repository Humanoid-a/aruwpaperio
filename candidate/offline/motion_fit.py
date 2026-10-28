"""Fitting the belief's opponent motion model K to recorded games, and checking the belief with it.

    python -m candidate.offline.motion_fit build --workers 16     # replay every recorded game's tapes -> the data
    python -m candidate.offline.motion_fit fit                    # fit K's weights; K-level report on held-out games
    python -m candidate.offline.motion_fit fit --only enemy noisy hunter basic --name purposeful   # on some types only
    python -m candidate.offline.motion_fit check --fits all purposeful --workers 16   # the whole belief, each K

K (agent/belief/motion.py) says how likely each of the opponent's four moves is: a weighted sum of seven features of
the move's outcome (dies on its own trail, kills me, captures, goes home, chases my trail, keeps its direction, leaves
its land), turned into probabilities by a softmax over the legal moves mixed with a small floor. Its weights were set
by hand. Here they are fitted to what opponents really did.

The data. Every headless game was recorded with both players' move tapes (offline/collect.py). The game rules are
deterministic, so the tapes replay the whole game exactly; this uses our own batched rules (agent/rules/transition.py,
checked against the engine on every turn of whole matches), not the engine, so a game replays in about a second, and
the replay is checked against the engine's final counts stored with the game. At every opponent turn the opponent's
true situation (its cell, trail, land, count, last direction; my cell, trail and land) is put into a one-hypothesis
belief and classified by the belief's own code (belief.generate.classify), so the features are the ones the belief
computes in play. The row is those features for the four moves, which are legal, and the move it actually made. The
opponent's own decisions do not depend on which version of our agent played, so every recorded game counts.

The fit. Maximum likelihood of the moves actually made, under exactly the belief's probability model (softmax with
beta, mixed with the floor epsilon, from BeliefConfig), with a little L2 pulling the weights towards the current
hand-set ones (a prior), not towards zero. That matters for rare situations: in 1.4 million recorded opponent moves a
move that would kill me was available only 48 times (our safety rules rarely allow it); every opponent took it every
time, but 48 rows cannot outweigh a penalty towards zero, which would have fitted "kills me" at about 0. With the
prior, a weight the data says little about stays where it was set by hand, and one it says a lot about moves. Games are split train /
validation / test by game (70 / 15 / 15, within each opponent type), and every opponent type weighs the same, so the
types with the most games do not decide K alone. Two constants shape the features rather than weigh them (how fast the
pull home grows with the trail, and how far the opponent sees my trail); they are chosen on the validation games.

The checks. K level: on the test games, per opponent type, the mean log-likelihood of the move made and how often K's
most likely move was the move made, for a uniform K, the current hand-set K, the fitted K, and K fitted to that one
type (the best one set of weights could do for it). Belief level: test games are replayed through our agent's real
run-time and belief filter, once with each K, and every turn the belief is scored against the opponent's true state:
the probability on its true cell (and the mean log of it, the proper score of a probabilistic guess), whether the
true cell was possible at all, whether the most likely cell was right, and the mass on the exactly right hidden state.

Nothing here changes the agent: the fitted weights are written to q_training/belief/k_fit.json, and adopting them is
a separate decision (they become the MotionWeights defaults). That changes Q's inputs, so Q's recorded games would be
replayed when a training set is next built (q/version.py).
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from candidate.offline import locations

FEATURES = ("self_trail", "kills_my_trail", "capture", "home_pull", "chase", "momentum", "explore")
LETTERS = "NESW"                                   # our Direction order: NORTH, EAST, SOUTH, WEST = 0..3
DATA = locations.BELIEF / "k_data.npz"
FIT = locations.BELIEF / "k_fit.json"
CHECK = locations.BELIEF / "k_check.json"
HALVES = (2.0, 3.0, 4.0, 6.0, 9.0, 14.0, 20.0)     # candidates for home_pressure_half
RADII = (2, 3, 4, 5, 6, 8, 10)                     # candidates for view_radius


# ---- finding and replaying recorded games ---------------------------------------------------------------------------


def find_games(roots=None) -> list[Path]:
    """Every recorded game's sidecar with move tapes, under the raw runs and the online runs."""
    roots = [Path(r) for r in roots] if roots else [locations.RAW, locations.ONLINE]
    out = []
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            # a game's sidecar sits next to its record (<game id>.npz); replays of stale records are not new games
            if path.with_suffix(".npz").exists() and not path.parent.name.startswith("replayed-"):
                out.append(path)
    return out


def _respawns(config: dict) -> list[int]:
    """The two respawn cells (flat, per player) the game was played with."""
    from candidate.agent.geometry.grid import flat
    from candidate.offline import match

    respawns = config.get("respawns")
    positions = match.config_for(config["turns"], [tuple(r) for r in respawns] if respawns else None).respawn_positions
    return [flat(p.y, p.x) for p in positions]


def replay_boards(sidecar: dict, want_opponent_rows: bool = True):
    """Replay a game's tapes with our rules, from the candidate's point of view. Yields, before every opponent move
    (want_opponent_rows) or before every one of my moves (otherwise): (turn index of that player, the BitWorld,
    the opponent's last direction or -1). Checks the final counts against the engine's."""
    from candidate.agent.data.types import Side
    from candidate.agent.rules.match import MatchRules
    from candidate.agent.rules.transition import BitWorld, step

    config, tapes = sidecar["config"], sidecar["tapes"]
    seat, turns = int(config["seat"]), int(config["turns"])
    respawn = _respawns(config)
    rules = MatchRules.from_respawns(respawn[seat], respawn[1 - seat])
    world = BitWorld.from_world(rules.initial_world())
    opp_last = -1
    for k in range(2 * turns):
        mover = k % 2
        letter = tapes[mover][k // 2]
        if letter not in LETTERS:
            raise ValueError(f"turn {k}: no accepted move recorded ({letter!r})")
        d = LETTERS.index(letter)
        side = Side.ME if mover == seat else Side.OPP
        if (side == Side.OPP) == want_opponent_rows:
            yield k // 2, world, rules, opp_last
        result = step(world, rules, side, [d])
        if int(result.result[0]) == 3:
            raise ValueError(f"turn {k}: the rules refuse a move the engine accepted")
        world = result.world
        if side == Side.OPP:
            opp_last = d
    mine, theirs = world.counts()
    final = sidecar.get("final", {}).get("counts")
    if final is not None and (int(mine[0]), int(theirs[0])) != (final[seat], final[1 - seat]):
        raise ValueError(f"replay ended at {int(mine[0])}/{int(theirs[0])}, the engine at {final[seat]}/{final[1 - seat]}")


# ---- one row per opponent move --------------------------------------------------------------------------------------


def opponent_rows(sidecar_path: str) -> dict:
    """The rows of one game (see the module notes). Runs in a worker process."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    from candidate.agent.belief.generate import Situation, classify
    from candidate.agent.data.hypotheses import HypothesisStore
    from candidate.agent.data.territory import TerritoryDiffs
    from candidate.agent.geometry import bitboard
    from candidate.agent.geometry.distance import taxicab_distance

    path = Path(sidecar_path)
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    tapes = sidecar["tapes"]
    opp_seat = 1 - int(sidecar["config"]["seat"])
    cols = {k: [] for k in ("legal", "label", "self_trail", "kills_me", "capture_gain", "home_change", "trail_len",
                            "chase_change", "my_trail_dist", "momentum", "explore", "turn")}
    home_cache: dict[bytes, np.ndarray] = {}
    started = time.perf_counter()
    try:
        for turn, w, rules, opp_last in replay_boards(sidecar, want_opponent_rows=True):
            opp_land, opp_trail, my_trail, my_land = w.opp_territory[0], w.opp_trail[0], w.my_trail[0], w.my_territory[0]
            pos, my_pos = int(w.opp_pos[0]), int(w.my_pos[0])
            trail_len = int(bitboard.popcount(opp_trail))
            opp_diffs, my_diffs = TerritoryDiffs(), TerritoryDiffs()
            store = HypothesisStore.build(
                pos=[pos], last_dir=[opp_last if trail_len else -1], trail_len=[trail_len], trail_bb=opp_trail[None, :],
                opp_land_id=[0], my_land_id=[0], opp_count=[int(bitboard.popcount(opp_land))], flags=[0], weight=[1.0],
                hash=[0],
            )
            sit = Situation(rules, None, opp_land, opp_diffs, my_pos, my_trail, my_land, my_diffs,
                            int(bitboard.popcount(my_land)))
            b = classify(SimpleNamespace(store=store), sit).batch
            dest = b.dest[0].astype(np.int64)
            key = opp_land.tobytes()
            if key not in home_cache:
                home_cache[key] = taxicab_distance(opp_land).reshape(-1).astype(np.int64)
            home = home_cache[key]
            mine = taxicab_distance(my_trail).reshape(-1).astype(np.int64)
            cols["legal"].append(b.legal[0])
            cols["label"].append(LETTERS.index(tapes[opp_seat][turn]))
            cols["self_trail"].append(b.self_trail[0])
            cols["kills_me"].append(b.kills_me[0])
            cols["capture_gain"].append(b.capture_gain[0])
            cols["home_change"].append(np.clip(home[pos] - home[dest], -1, 1))
            cols["trail_len"].append(trail_len)
            cols["chase_change"].append(np.clip(mine[pos] - mine[dest], -1, 1))
            cols["my_trail_dist"].append(int(min(mine[pos], 30000)))
            cols["momentum"].append(np.arange(4) == (opp_last if trail_len else -1))
            cols["explore"].append((trail_len == 0) & ~b.enters_own[0])
            cols["turn"].append(turn)
    except ValueError as error:
        return dict(path=str(path), error=str(error))
    dtypes = dict(legal=bool, label=np.int8, self_trail=bool, kills_me=bool, capture_gain=np.int32,
                  home_change=np.int8, trail_len=np.int16, chase_change=np.int8, my_trail_dist=np.int16,
                  momentum=bool, explore=bool, turn=np.int16)
    rows = {k: np.asarray(v, dtype=dtypes[k]) for k, v in cols.items()}
    return dict(path=str(path), opponent=sidecar["config"]["opponent"], rows=rows,
                seconds=round(time.perf_counter() - started, 2))


def features_from(d: dict, half: float, radius: int) -> np.ndarray:
    """(N, 4, 7) features in FEATURES order, for the given shape constants (FeatureMotionModel.features' formulas)."""
    trail = d["trail_len"].astype(np.float64)
    pressure = trail / (trail + half)
    in_view = (d["my_trail_dist"] <= radius)[:, None]
    parts = [
        d["self_trail"].astype(np.float64),
        d["kills_me"].astype(np.float64),
        np.log1p(d["capture_gain"].astype(np.float64)),
        d["home_change"].astype(np.float64) * pressure[:, None],
        d["chase_change"].astype(np.float64) * in_view,
        d["momentum"].astype(np.float64),
        d["explore"].astype(np.float64),
    ]
    return np.stack(parts, axis=-1)


def build(roots=None, workers: int = 16, out: Path = DATA, echo=print) -> dict:
    """Replay every recorded game and save the rows (one file). Returns a summary."""
    paths = [str(p) for p in find_games(roots)]
    echo(f"replaying {len(paths)} recorded games with {workers} workers")
    started = time.perf_counter()
    if workers <= 1:
        results = [opponent_rows(p) for p in paths]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(opponent_rows, paths, chunksize=4))
    failed = [r for r in results if "error" in r]
    good = [r for r in results if "error" not in r]
    if not good:
        raise RuntimeError("no game could be replayed: " + "; ".join(f"{r['path']}: {r['error']}" for r in failed[:5]))
    arrays = {k: np.concatenate([r["rows"][k] for r in good]) for k in good[0]["rows"]}
    arrays["game"] = np.concatenate([np.full(len(r["rows"]["label"]), i, dtype=np.int32) for i, r in enumerate(good)])
    opponents = sorted({r["opponent"] for r in good})
    meta = dict(games=[r["path"] for r in good], opponent=[r["opponent"] for r in good], opponents=opponents,
                failed=[dict(path=r["path"], error=r["error"]) for r in failed],
                created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), **arrays)
    summary = dict(games=len(good), failed=len(failed), rows=int(len(arrays["label"])),
                   per_opponent={o: sum(r["opponent"] == o for r in good) for o in opponents},
                   seconds=round(time.perf_counter() - started, 1), file=str(out))
    echo(f"{summary['rows']} opponent moves from {summary['games']} games ({summary['failed']} could not be replayed) "
         f"in {summary['seconds']:.0f}s -> {out}; games per opponent {summary['per_opponent']}")
    for f in failed[:5]:
        echo(f"  not replayed: {f['path']}: {f['error']}")
    return summary


def load(path: Path = DATA) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return meta, {k: z[k] for k in z.files if k != "meta"}


# ---- the split, the likelihood and the fit --------------------------------------------------------------------------


def split(meta: dict, seed: int = 0, fractions=(0.7, 0.15, 0.15)) -> dict[str, np.ndarray]:
    """Game indices for train / val / test, split within each opponent type."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    opponent = np.array(meta["opponent"])
    for name in meta["opponents"]:
        games = rng.permutation(np.flatnonzero(opponent == name))
        n_val = max(1, int(round(fractions[1] * len(games))))
        n_test = max(1, int(round(fractions[2] * len(games))))
        parts["test"].extend(games[:n_test])
        parts["val"].extend(games[n_test:n_test + n_val])
        parts["train"].extend(games[n_test + n_val:])
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in parts.items()}


def probabilities(F: np.ndarray, legal: np.ndarray, w: np.ndarray, beta: float, eps: float) -> np.ndarray:
    """The belief's move probabilities (belief.motion.move_probabilities) for features F and weights w."""
    from candidate.agent.belief.motion import move_probabilities

    return move_probabilities(F @ w, legal, beta, eps)


def fit_weights(F: np.ndarray, legal: np.ndarray, label: np.ndarray, sample_w: np.ndarray, start: np.ndarray,
                beta: float, eps: float, l2: float = 1e-3, prior: np.ndarray | None = None) -> np.ndarray:
    """Maximum likelihood weights (L-BFGS, float64, on the CPU), with L2 towards `prior` (zero if None)."""
    import torch

    Ft = torch.as_tensor(F, dtype=torch.float64)
    Lt = torch.as_tensor(legal)
    yt = torch.as_tensor(label.astype(np.int64))
    st = torch.as_tensor(sample_w / sample_w.sum(), dtype=torch.float64)
    n_legal = Lt.sum(dim=1).clamp(min=1).to(torch.float64)
    w = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    centre = torch.as_tensor(np.zeros(len(start)) if prior is None else prior, dtype=torch.float64)
    optimiser = torch.optim.LBFGS([w], lr=1.0, max_iter=300, tolerance_grad=1e-9, tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimiser.zero_grad()
        logits = torch.where(Lt, beta * (Ft @ w), torch.tensor(-torch.inf, dtype=torch.float64))
        soft = torch.softmax(logits, dim=1)
        p = (1 - eps) * soft.gather(1, yt[:, None])[:, 0] + eps / n_legal
        loss = -(st * torch.log(p)).sum() + l2 * ((w - centre) ** 2).sum()
        loss.backward()
        return loss

    optimiser.step(closure)
    return w.detach().numpy().copy()


def _type_weights(meta: dict, game: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Per-row weights making every opponent type weigh the same (and every game within a type the same)."""
    opponent = np.array(meta["opponent"])[game[rows]]
    out = np.zeros(len(rows), dtype=np.float64)
    types = np.unique(opponent)
    for name in types:
        mask = opponent == name
        games, counts = np.unique(game[rows][mask], return_counts=True)
        per_game = dict(zip(games.tolist(), counts.tolist()))
        out[mask] = [1.0 / (len(games) * per_game[g]) for g in game[rows][mask]]
    return out / len(types)


def metrics(F, legal, label, w, beta, eps) -> dict:
    p = probabilities(F, legal, w, beta, eps)
    p_made = p[np.arange(len(label)), label]
    return dict(moves=int(len(label)), log_likelihood=float(np.mean(np.log(np.maximum(p_made, 1e-300)))),
                accuracy=float(np.mean(p.argmax(axis=1) == label)), p_made=float(p_made.mean()))


def current_weights():
    """The MotionWeights in use now (the hand-set defaults), as (weights vector, half, radius)."""
    from candidate.agent.belief.motion import MotionWeights

    mw = MotionWeights()
    return np.array([getattr(mw, f) for f in FEATURES], dtype=np.float64), mw.home_pressure_half, mw.view_radius


def fit_path(name: str) -> Path:
    """Where a fit is saved: k_fit.json for "all" (every opponent type), else k_fit_<name>.json."""
    return FIT if name == "all" else FIT.with_name(f"k_fit_{name}.json")


def fit(data: Path = DATA, out: Path | None = None, seed: int = 0, only=None, name: str = "all", echo=print) -> dict:
    """Fit K on the training games, choose the shape constants on the validation games, report on the test games.
    `only` restricts training and validation to some opponent types; the test report still covers every type, to show
    what the fit does to the others."""
    out = out or fit_path(name)
    from candidate.agent.belief.config import BeliefConfig

    cfg = BeliefConfig()
    beta, eps = cfg.beta, cfg.epsilon
    meta, d = load(data)
    parts = split(meta, seed)
    game = d["game"]
    rows = {k: np.flatnonzero(np.isin(game, v)) for k, v in parts.items()}
    if only:
        unknown = set(only) - set(meta["opponents"])
        if unknown:
            raise ValueError(f"no recorded games against {sorted(unknown)}; there are {meta['opponents']}")
        kind = np.array(meta["opponent"])[game]
        for k in ("train", "val"):
            rows[k] = rows[k][np.isin(kind[rows[k]], list(only))]
    legal, label = d["legal"], d["label"].astype(np.int64)
    sw_train = _type_weights(meta, game, rows["train"])
    sw_val = _type_weights(meta, game, rows["val"])
    w_now, half_now, radius_now = current_weights()
    echo(f"{len(label)} moves; games train {len(parts['train'])}, val {len(parts['val'])}, test {len(parts['test'])}")

    best = None
    for half in HALVES:
        for radius in RADII:
            F = features_from(d, half, radius)
            tr, va = rows["train"], rows["val"]
            w = fit_weights(F[tr], legal[tr], label[tr], sw_train, w_now, beta, eps, prior=w_now)
            p = probabilities(F[va], legal[va], w, beta, eps)[np.arange(len(va)), label[va]]
            val_ll = float((sw_val * np.log(np.maximum(p, 1e-300))).sum() / sw_val.sum())
            echo(f"  home_pressure_half {half:5.1f}  view_radius {radius:2d}: validation log-likelihood {val_ll:.4f}")
            if best is None or val_ll > best[0]:
                best = (val_ll, half, radius, w)
    val_ll, half, radius, w_fit = best
    echo(f"chosen: home_pressure_half {half}, view_radius {radius} (validation log-likelihood {val_ll:.4f})")

    F_fit = features_from(d, half, radius)
    F_now = features_from(d, half_now, radius_now)
    opponent = np.array(meta["opponent"])[game]
    report = dict(beta=beta, epsilon=eps, chosen=dict(home_pressure_half=half, view_radius=radius), test={})
    test = rows["test"]
    for name in meta["opponents"] + ["all"]:
        r = test if name == "all" else test[opponent[test] == name]
        if not len(r):
            continue
        entry = dict(
            games=int(len(np.unique(game[r]))),
            uniform=metrics(F_now[r], legal[r], label[r], np.zeros(7), beta, eps),
            current=metrics(F_now[r], legal[r], label[r], w_now, beta, eps),
            fitted=metrics(F_fit[r], legal[r], label[r], w_fit, beta, eps),
        )
        if name != "all":                                   # the best one set of weights could do for this type alone
            tr = rows["train"][opponent[rows["train"]] == name]
            w_own = fit_weights(F_fit[tr], legal[tr], label[tr], np.ones(len(tr)), w_fit, beta, eps, prior=w_now)
            entry["fitted_to_this_type"] = metrics(F_fit[r], legal[r], label[r], w_own, beta, eps)
            entry["fitted_to_this_type"]["weights"] = dict(zip(FEATURES, map(float, np.round(w_own, 3))))
        report["test"][name] = entry
    weights = dict(zip(FEATURES, map(float, w_fit)), home_pressure_half=float(half), view_radius=int(radius))
    current = dict(zip(FEATURES, map(float, w_now)), home_pressure_half=float(half_now), view_radius=int(radius_now))
    result = dict(name=name, trained_on=sorted(only) if only else meta["opponents"], weights=weights, current=current,
                  report=report, data=str(data), split_seed=seed,
                  created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    echo(format_fit(result))
    return result


def format_fit(result: dict) -> str:
    lines = ["", "K's weights        current    fitted"]
    for name in FEATURES + ("home_pressure_half", "view_radius"):
        lines.append(f"  {name:18s} {result['current'][name]:8.3f}  {result['weights'][name]:8.3f}")
    lines.append("")
    lines.append("held-out games: mean log-likelihood of the move made (higher is better) / how often K's top move was it")
    lines.append(f"  {'opponent':8s} {'games':>5s}   {'uniform':>15s}   {'current K':>15s}   {'fitted K':>15s}   "
                 f"{'fitted to type':>15s}")
    for name, e in result["report"]["test"].items():
        cell = lambda m: f"{m['log_likelihood']:7.3f} / {m['accuracy']:5.1%}"  # noqa: E731
        own = cell(e["fitted_to_this_type"]) if "fitted_to_this_type" in e else ""
        lines.append(f"  {name:8s} {e['games']:5d}   {cell(e['uniform'])}   {cell(e['current'])}   {cell(e['fitted'])}   {own}")
    return "\n".join(lines)


# ---- the whole belief, replayed, current K against fitted K ---------------------------------------------------------


class _TapeStrategy:
    """Plays my recorded tape, and scores the belief against the true opponent every turn."""

    def __init__(self, tape: str, truth: dict) -> None:
        self.tape, self.truth, self.rows = tape, truth, []

    def reset(self) -> None:
        self.rows = []

    def choose(self, view):
        from candidate.agent.data.types import Direction

        t = view.obs.turn
        if view.retry:
            raise RuntimeError(f"turn {t}: the engine refused a recorded move")
        if view.belief is not None:
            self.rows.append(_score(view.belief, self.truth, t))
        return Direction(LETTERS.index(self.tape[t]))

    def on_result(self, view, direction, result) -> None:
        pass

    def debug_state(self) -> str:
        return "tape"


def _score(belief, truth: dict, t: int) -> dict:
    cell, trail, count = int(truth["pos"][t]), truth["trail"][t], int(truth["count"][t])
    p = float(belief.probability_at(cell))
    exact = 0.0
    if belief.n_hypotheses:
        match = (belief._pos == cell) & (belief._opp_count == count) & (belief._trail_bb == trail).all(axis=1)
        exact = float(belief._weight[match].sum())
    return dict(p=p, possible=bool(belief.support.reshape(-1)[cell]), top=belief.most_likely()[0] == cell,
                exact=exact, seen=belief.sighted_at is not None, n=int(belief.n_hypotheses))


def belief_game(args: tuple) -> dict:
    """Replay one game through our agent's run-time and belief with the given K, scoring the belief every turn."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    sidecar_path, weights, turns = args
    from candidate.agent.belief.filter import BeliefFilter
    from candidate.agent.belief.motion import FeatureMotionModel, MotionWeights
    from candidate.agent.controller import Controller
    from candidate.agent.diagnostics import Diagnostics
    from candidate.agent.runtime import AgentRuntime
    from candidate.offline import match

    sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    seat = int(sidecar["config"]["seat"])
    turns = min(turns, int(sidecar["config"]["turns"]))
    truth = dict(pos=[], trail=[], count=[])
    for _, w, _, _ in replay_boards(sidecar, want_opponent_rows=False):        # the board each of my turns faces
        truth["pos"].append(int(w.opp_pos[0]))
        truth["trail"].append(w.opp_trail[0].copy())
        truth["count"].append(int(w.counts()[1][0]))
    strategy = _TapeStrategy(sidecar["tapes"][seat], truth)
    diagnostics = Diagnostics()
    controller = Controller(strategy, diagnostics)
    controller.runtime = AgentRuntime(diagnostics, belief=BeliefFilter(motion=FeatureMotionModel(MotionWeights(**weights))))
    controllers = [match.Replayer(sidecar["tapes"][0]), match.Replayer(sidecar["tapes"][1])]
    controllers[seat] = match.AgentAdapter(controller)
    started = time.perf_counter()
    result = match.play(controllers, turns, [tuple(r) for r in sidecar["config"]["respawns"]]
                        if sidecar["config"].get("respawns") else None)
    rows = strategy.rows
    p = np.array([r["p"] for r in rows])
    seen = np.array([r["seen"] for r in rows])
    return dict(
        path=sidecar_path, opponent=sidecar["config"]["opponent"], turns=len(rows), failed_turns=result.failed_turns,
        p_true=float(p.mean()), p_true_unseen=float(p[~seen].mean()) if (~seen).any() else None,
        log_p=float(np.mean(np.log(np.maximum(p, 1e-12)))), log_p_unseen=float(np.mean(np.log(np.maximum(p[~seen], 1e-12))))
        if (~seen).any() else None,
        possible=float(np.mean([r["possible"] for r in rows])), top=float(np.mean([r["top"] for r in rows])),
        top_unseen=float(np.mean([r["top"] for r, s in zip(rows, seen) if not s])) if (~seen).any() else None,
        exact=float(np.mean([r["exact"] for r in rows])), hypotheses=float(np.mean([r["n"] for r in rows])),
        seconds=round(time.perf_counter() - started, 1), errors=len(diagnostics.errors()),
    )


def check(fits=("all",), data: Path = DATA, games_per_opponent: int = 6, turns: int = 2000, workers: int = 16,
          out: Path = CHECK, seed: int = 0, echo=print) -> dict:
    """The whole belief on held-out games, with the current K and with each fitted one (fit names, see fit_path)."""
    loaded = {name: json.loads(fit_path(name).read_text(encoding="utf-8")) for name in fits}
    meta, _ = load(data)
    test = split(meta, seed)["test"]
    opponent = np.array(meta["opponent"])
    chosen = []
    for name in meta["opponents"]:
        chosen += [meta["games"][g] for g in test[opponent[test] == name][:games_per_opponent]]
    variants = {"current": next(iter(loaded.values()))["current"]}
    variants.update({f"fitted:{name}": f["weights"] for name, f in loaded.items()})
    jobs = [(path, weights, turns) for weights in variants.values() for path in chosen]
    echo(f"replaying {len(chosen)} held-out games through the belief, with each of {list(variants)} ({len(jobs)} runs, "
         f"{workers} workers)")
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(belief_game, jobs))
    per_variant = {name: results[i * len(chosen):(i + 1) * len(chosen)] for i, name in enumerate(variants)}
    keys = ("p_true", "p_true_unseen", "log_p", "log_p_unseen", "possible", "top", "top_unseen", "exact", "hypotheses")
    report = dict(games=len(chosen), turns=turns, seconds=round(time.perf_counter() - started, 1), variants={})
    for name, rs in per_variant.items():
        groups = {o: [r for r in rs if r["opponent"] == o] for o in meta["opponents"]}
        groups["all"] = rs
        report["variants"][name] = {
            o: {k: (float(np.mean([r[k] for r in g if r[k] is not None])) if any(r[k] is not None for r in g) else None)
                for k in keys} | dict(games=len(g), errors=sum(r["errors"] for r in g),
                                      failed_turns=sum(r["failed_turns"] for r in g))
            for o, g in groups.items() if g
        }
    report["paired_log_p"] = {name: _paired(per_variant["current"], per_variant[name], "log_p")
                              for name in variants if name != "current"}
    out.write_text(json.dumps(dict(report=report, games=results), indent=1), encoding="utf-8")
    echo(format_check(report))
    return report


def _paired(a: list[dict], b: list[dict], key: str) -> dict:
    d = np.array([y[key] - x[key] for x, y in zip(a, b)])
    return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0,
                better=int((d > 0).sum()), worse=int((d < 0).sum()))


def format_check(report: dict) -> str:
    lines = ["", f"the whole belief on {report['games']} held-out games ({report['turns']} turns each), per turn:",
             "  p(true cell) mean, overall / out of view; log p(true cell) (higher is better); true cell possible; "
             "most likely cell right, overall / out of view; mass on the exact hidden state; hypotheses"]
    for name, groups in report["variants"].items():
        lines.append(f"  K {name}:")
        for o, m in groups.items():
            f = lambda v, s="{:.3f}": "  -  " if v is None else s.format(v)  # noqa: E731
            lines.append(f"    {o:8s} ({m['games']:2d} games)  p {f(m['p_true'])} / {f(m['p_true_unseen'])}   log p "
                         f"{f(m['log_p'], '{:7.2f}')} / {f(m['log_p_unseen'], '{:7.2f}')}   possible {f(m['possible'], '{:.4f}')}"
                         f"   top {f(m['top'])} / {f(m['top_unseen'])}   exact {f(m['exact'])}   hyp "
                         f"{f(m['hypotheses'], '{:5.0f}')}   errors {m['errors']}")
    for name, p in report["paired_log_p"].items():
        lines.append(f"  {name} - current, mean log p(true cell) per game: {p['mean']:+.3f} +- {p['se']:.3f} "
                     f"(better in {p['better']} games, worse in {p['worse']})")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="replay the recorded games into K's training data")
    b.add_argument("--raw", nargs="*", default=None, help="directories to search (default: q_training/raw and online)")
    b.add_argument("--workers", type=int, default=16)
    f = sub.add_parser("fit", help="fit K and report on held-out games")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--only", nargs="*", default=None, help="opponent types to fit on (default: all)")
    f.add_argument("--name", default="all", help="the fit's name: saved as k_fit.json for all, else k_fit_<name>.json")
    c = sub.add_parser("check", help="the whole belief on held-out games, current K vs fitted K")
    c.add_argument("--games-per-opponent", type=int, default=6)
    c.add_argument("--turns", type=int, default=2000)
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--fits", nargs="*", default=["all"], help="the fits to compare with the current K")
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.raw, args.workers)
    elif args.command == "fit":
        fit(seed=args.seed, only=args.only, name=args.name)
    else:
        check(args.fits, games_per_opponent=args.games_per_opponent, turns=args.turns, workers=args.workers,
              seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
