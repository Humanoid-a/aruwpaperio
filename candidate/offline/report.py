"""What training did: curves from an online run's records, and a fixed benchmark of its checkpoints.

    python -m candidate.offline.report curves --run online6
    python -m candidate.offline.report benchmark --run online6 --epochs 0 4 9 14 19 --workers 16

Both write to q_training/analysis/<run>/.

curves     reads every finished epoch (history.jsonl and each epoch's games.json) and draws: win rate and mean
           territory difference per epoch, per opponent, for the games with the real belief (how the agent really
           plays) and with the true world; territory, deaths and kills; how often each state was taken; how often the
           safety rules stepped in; Q's and K's training. Training-time win rates are only a rough guide: the opponents
           change from game to game (the self opponent is the network being trained, the noisy enemy is re-drawn, the
           belief/true-world split is random).
benchmark  the clean measure: the hand-set stand-in and the chosen checkpoints (each epoch's Q with that epoch's K) play
           the same games, with the real belief and no exploration, against the enemy copy (the benchmark opponent,
           deterministic) and the noisy enemy with fixed seeds, both seats, far and close respawns. Paired: the only
           difference between rows is the network.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from candidate.offline import collect, locations

OPPONENT_ORDER = ("enemy_copy", "noisy", "self", "enemy", "basic", "random", "hunter")


def analysis_dir(run: str) -> Path:
    out = locations.ROOT / "analysis" / run
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---- reading a run ----------------------------------------------------------------------------------------------------


def load_run(run: str) -> list[dict]:
    """Every finished epoch: its summary plus its games."""
    root = locations.ONLINE / run
    epochs = []
    for line in (root / "history.jsonl").read_text(encoding="utf-8").splitlines():
        summary = json.loads(line)
        games_file = root / f"epoch_{summary['epoch']:03d}" / "games.json"
        summary["game_list"] = json.loads(games_file.read_text(encoding="utf-8")) if games_file.exists() else []
        epochs.append(summary)
    seen, unique = set(), []
    for e in sorted(epochs, key=lambda s: s["epoch"]):           # a resumed run may have written an epoch twice
        if e["epoch"] not in seen:
            seen.add(e["epoch"])
            unique.append(e)
    return unique


def per_epoch(epochs: list[dict], belief: bool | None, opponent: str | None = None) -> dict:
    """Per epoch, over the games of one kind (belief True / true world False / both None) and opponent (None: all):
    win rate, mean territory difference and its standard error, mean territories, deaths and kills per game, state
    shares, safety-rule interventions per game."""
    out = defaultdict(list)
    for e in epochs:
        games = [g for g in e["game_list"]
                 if (belief is None or bool(g["spec"].get("oracle")) != belief)
                 and (opponent is None or g["spec"]["opponent"] == opponent)]
        out["epoch"].append(e["epoch"])
        n = len(games)
        out["n"].append(n)
        if not n:
            for k in ("win", "diff", "se", "mine", "theirs", "deaths", "kills", "safety", "lookahead"):
                out[k].append(np.nan)
            out["states"].append({})
            continue
        diff = np.array([g["my_count"] - g["opp_count"] for g in games], float)
        out["win"].append(np.mean([g["won"] for g in games]))
        out["diff"].append(diff.mean())
        out["se"].append(diff.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)
        out["mine"].append(np.mean([g["my_count"] for g in games]))
        out["theirs"].append(np.mean([g["opp_count"] for g in games]))
        out["deaths"].append(np.mean([g["my_deaths"] for g in games]))
        out["kills"].append(np.mean([g["my_kills"] for g in games]))
        out["safety"].append(np.mean([g["stats"].get("forced_safety", 0) for g in games]))
        out["lookahead"].append(np.mean([g["stats"].get("forced_lookahead", 0) for g in games]))
        totals = defaultdict(int)
        for g in games:
            for k, v in g["stats"].items():
                if k.startswith("state_"):
                    totals[k[6:]] += v
        total = sum(totals.values()) or 1
        out["states"].append({k: v / total for k, v in totals.items()})
    return {k: (np.array(v) if k != "states" else v) for k, v in out.items()}


# ---- the curves ----------------------------------------------------------------------------------------------------------


def curves(run: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = load_run(run)
    if not epochs:
        raise ValueError(f"no finished epochs in {run}")
    out = analysis_dir(run)
    present = {g["spec"]["opponent"] for e in epochs for g in e["game_list"]}
    opponents = [o for o in OPPONENT_ORDER if o in present]
    colors = dict(zip(OPPONENT_ORDER, ("tab:red", "tab:orange", "tab:blue", "tab:purple", "tab:brown", "tab:gray", "tab:green")))
    files = []

    # 1. results per epoch: win rate and territory difference, real belief vs true world
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for col, (belief, title) in enumerate(((True, "with the real belief (how it really plays)"),
                                           (False, "with the true world (80% / 60% of training games)"))):
        for o in opponents:
            s = per_epoch(epochs, belief, o)
            axes[0, col].plot(s["epoch"], s["win"] * 100, "o-", color=colors[o], label=o, ms=4)
            axes[1, col].plot(s["epoch"], s["diff"], "o-", color=colors[o], label=o, ms=4)
            axes[1, col].fill_between(s["epoch"], s["diff"] - s["se"], s["diff"] + s["se"], color=colors[o], alpha=0.12)
        s = per_epoch(epochs, belief)
        axes[0, col].plot(s["epoch"], s["win"] * 100, "k--", lw=2, label="all")
        axes[0, col].set_title(f"Win rate, {title}", fontsize=10)
        axes[1, col].set_title(f"Mean territory difference (mine - theirs), {title.split(' (')[0]}", fontsize=10)
        axes[0, col].set_ylim(-3, 103)
        axes[0, col].axhline(50, color="gray", lw=0.8, ls=":")
        axes[1, col].axhline(0, color="gray", lw=0.8)
        axes[1, col].set_xlabel("epoch")
    axes[0, 0].set_ylabel("% of games won")
    axes[1, 0].set_ylabel("cells")
    axes[0, 1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{run}: results during training (shaded: +-1 standard error)")
    fig.tight_layout()
    files.append(_save(fig, out / "1_results.png"))

    # 2. territory, deaths, kills, safety interventions (real belief)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for o in opponents:
        s = per_epoch(epochs, True, o)
        axes[0, 0].plot(s["epoch"], s["mine"], "o-", color=colors[o], label=f"mine vs {o}", ms=4)
        axes[0, 0].plot(s["epoch"], s["theirs"], "x:", color=colors[o], label=f"{o}'s", ms=4)
        axes[0, 1].plot(s["epoch"], s["kills"], "o-", color=colors[o], label=f"kills vs {o}", ms=4)
        axes[0, 1].plot(s["epoch"], s["deaths"], "x:", color=colors[o], label=f"deaths vs {o}", ms=4)
    for belief, ls, name in ((True, "-", "real belief"), (False, "--", "true world")):
        s = per_epoch(epochs, belief)
        axes[1, 0].plot(s["epoch"], s["safety"], ls, color="tab:red", label=f"forced RETURN (margin <= 1), {name}")
        axes[1, 0].plot(s["epoch"], s["lookahead"], ls, color="tab:blue", label=f"look-ahead, {name}")
    axes[0, 0].set_title("Final territory per game (real belief)", fontsize=10)
    axes[0, 1].set_title("Kills and deaths per game (real belief)", fontsize=10)
    axes[1, 0].set_title("Safety rules stepping in, per game", fontsize=10)
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[1, 0].legend(fontsize=7)
    seconds = [e["seconds"] / 60 for e in epochs]
    axes[1, 1].bar([e["epoch"] for e in epochs], seconds, color="tab:gray")
    axes[1, 1].set_title("Minutes per epoch (play + dataset + training)", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    fig.suptitle(f"{run}: territory, kills, deaths, safety")
    fig.tight_layout()
    files.append(_save(fig, out / "2_details.png"))

    # 3. which states it takes (real belief vs true world)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    names = ["EXPAND", "EXPAND2", "RETURN", "HUNT", "TRACK", "PATROL"]
    state_colors = ["tab:green", "tab:olive", "tab:blue", "tab:red", "tab:purple", "tab:gray"]
    for ax, (belief, title) in zip(axes, ((True, "real belief"), (False, "true world"))):
        s = per_epoch(epochs, belief)
        shares = np.array([[st.get(n, 0.0) for n in names] for st in s["states"]]) * 100
        ax.stackplot(s["epoch"], shares.T, labels=names, colors=state_colors, alpha=0.85)
        ax.set_title(f"Share of frames in each state ({title})", fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("% of frames")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    files.append(_save(fig, out / "3_states.png"))

    # 4. training: Q (loss, R^2) and K (log-likelihood per opponent)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ep = [e["epoch"] for e in epochs]

    def held_out(e):                                    # (loss, constant guess, R^2, which): test games if any, else validation
        t = e["train"]
        if t.get("test_r2") is not None and np.isfinite(t.get("test", np.nan)):
            return t["test"], t["test_constant_baseline"], t["test_r2"], "test"
        return t["val"], t.get("val_constant_baseline", np.nan), t.get("val_r2") or np.nan, "validation"

    held = [held_out(e) for e in epochs]
    kinds = sorted({h[3] for h in held})
    axes[0].semilogy(ep, [h[0] for h in held], "o-", label="Q's loss")
    axes[0].semilogy(ep, [h[1] for h in held], "x--", label="a constant guess")
    axes[0].set_title(f"Q: squared error on held-out ({'/'.join(kinds)}) games (cells^2)", fontsize=10)
    axes[0].legend(fontsize=8)
    r2 = np.array([h[2] for h in held], dtype=float)
    axes[1].plot(ep, r2, "o-", color="tab:green")
    axes[1].set_ylim(min(0.0, float(np.nanmin(r2)) if np.isfinite(r2).any() else 0.0) - 0.05, 1)
    axes[1].set_title(f"Q: share of the label's variance explained (R^2, {'/'.join(kinds)})", fontsize=10)
    k_epochs = [e for e in epochs if e.get("k")]
    for o in [o for o in opponents if any(o in (e["k"].get("validation") or {}) for e in k_epochs)]:
        xs = [e["epoch"] for e in k_epochs if o in e["k"]["validation"]]
        axes[2].plot(xs, [e["k"]["validation"][o]["trained"]["log_likelihood"] for e in k_epochs if o in e["k"]["validation"]],
                     "o-", color=colors[o], label=f"trained K, {o}", ms=4)
        axes[2].plot(xs, [e["k"]["validation"][o]["hand"]["log_likelihood"] for e in k_epochs if o in e["k"]["validation"]],
                     ":", color=colors[o], label=f"hand-set K, {o}")
    axes[2].set_title("K: log-likelihood of the move the opponent made\n(held-out games; higher is better)", fontsize=10)
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("epoch")
    fig.tight_layout()
    files.append(_save(fig, out / "4_training.png"))

    # 5. the gate (runs with a gate): the learner against the player on the same fixed real-belief games
    gated = [e for e in epochs if e.get("gate")]
    if gated:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        xs = [e["epoch"] for e in gated]
        learner = np.array([e["gate"]["learner"]["mean_diff"] for e in gated])
        se = np.array([e["gate"]["learner"].get("se", 0.0) for e in gated])
        ax.errorbar(xs, learner, yerr=se, fmt="o-", color="tab:blue", capsize=3, label="the learner after this epoch")
        ax.step(xs, [e["gate"]["player"]["mean_diff"] for e in gated], where="mid", color="black", lw=2,
                label="the player it had to match")
        accepted = [e for e in gated if e["gate"]["accepted"]]
        ax.plot([e["epoch"] for e in accepted], [e["gate"]["learner"]["mean_diff"] for e in accepted], "*",
                color="tab:green", ms=14, label="became the player")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("mean territory difference (cells)")
        ax.set_title(f"{run}: the gate (fixed real-belief games, no exploration; +-1 SE)", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        files.append(_save(fig, out / "6_gate.png"))
    return files


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


# ---- the benchmark -------------------------------------------------------------------------------------------------------


def benchmark_specs(games_per_opponent: int = 8, turns: int = 2000, seed: int = 7) -> list[collect.GameSpec]:
    """The fixed games, 2 x games_per_opponent of them: both seats, far and close respawns, fixed seeds. The enemy copy
    and our agent (no exploration, no randomness) both play deterministically, so there are only 4 different games
    against it (2 seats x 2 respawns): it gets those once each, and the noisy enemy (randomness 0.5, a new draw of
    habits per seed) the rest."""
    specs = []
    rng = np.random.default_rng(seed)
    copies = min(games_per_opponent, 4)
    for opponent, randomness, count in (("enemy_copy", 0.0, copies), ("noisy", 0.5, 2 * games_per_opponent - copies)):
        for i in range(count):
            specs.append(collect.GameSpec(index=len(specs), opponent=opponent, seat=i % 2, close=(i // 2) % 2 == 1,
                                          turns=turns, seed=int(rng.integers(2**31)), explore=0.0, model="heuristic",
                                          out="", record=False, opponent_randomness=randomness))
    return specs


def benchmark(run: str, epochs: list[int], games_per_opponent: int = 8, turns: int = 2000, workers: int = 16,
              echo=print) -> dict:
    root = locations.ONLINE / run
    variants = {"stand-in": ("heuristic", None)}
    for e in epochs:
        here = root / f"epoch_{e:03d}"
        k = here / "k.npz"
        variants[f"epoch {e}"] = (str(here / "weights.pt"), str(k) if k.exists() else None)
    base = benchmark_specs(games_per_opponent, turns)
    jobs, owner = [], []
    for name, (model, k) in variants.items():
        for spec in base:
            jobs.append(replace(spec, model=model, opponent_model=model, k_model=k))
            owner.append(name)
    echo(f"benchmark: {len(variants)} variants x {len(base)} games = {len(jobs)} games, {workers} workers")
    started = time.perf_counter()
    results = collect.run(jobs, workers, echo=lambda line: None)
    table = defaultdict(lambda: defaultdict(list))
    for name, spec, r in zip(owner, jobs, results):
        table[name][spec.opponent].append(r)
    report = dict(run=run, turns=turns, games_per_opponent=games_per_opponent,
                  seconds=round(time.perf_counter() - started, 1), variants={})
    for name in variants:
        entry = {}
        for opponent, rs in table[name].items():
            diff = np.array([r["my_count"] - r["opp_count"] for r in rs], float)
            entry[opponent] = dict(games=len(rs), wins=int(sum(r["won"] for r in rs)), mean_diff=float(diff.mean()),
                                   se=float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0,
                                   mean_mine=float(np.mean([r["my_count"] for r in rs])),
                                   deaths=int(sum(r["my_deaths"] for r in rs)), kills=int(sum(r["my_kills"] for r in rs)),
                                   diffs=diff.tolist())
        report["variants"][name] = entry
    out = analysis_dir(run)
    (out / "benchmark.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    plot_benchmark(report, out / "5_benchmark.png")
    for name, entry in report["variants"].items():
        echo(f"  {name:10s} " + "   ".join(f"{o}: won {v['wins']}/{v['games']}, diff {v['mean_diff']:+7.1f} +- {v['se']:5.1f}, "
                                          f"deaths {v['deaths']}" for o, v in entry.items()))
    return report


def plot_benchmark(report: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(report["variants"])
    opponents = list(next(iter(report["variants"].values())))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    x = np.arange(len(names))
    width = 0.8 / len(opponents)
    colors = {"enemy_copy": "tab:red", "noisy": "tab:orange"}
    for i, o in enumerate(opponents):
        wins = [report["variants"][n][o]["wins"] / report["variants"][n][o]["games"] * 100 for n in names]
        diff = [report["variants"][n][o]["mean_diff"] for n in names]
        se = [report["variants"][n][o]["se"] for n in names]
        axes[0].bar(x + (i - (len(opponents) - 1) / 2) * width, wins, width, label=o, color=colors.get(o))
        axes[1].bar(x + (i - (len(opponents) - 1) / 2) * width, diff, width, yerr=se, capsize=3, label=o, color=colors.get(o))
    for ax, title in zip(axes, ("Win rate on the fixed games (real belief, no exploration)",
                                "Mean territory difference on the fixed games (+-1 SE)")):
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("% won")
    axes[0].set_ylim(0, 100)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_ylabel("cells")
    fig.suptitle(f"{report['run']}: the same {report['games_per_opponent']} games per opponent for each checkpoint "
                 f"({report['turns']} turns)")
    fig.tight_layout()
    return _save(fig, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("curves", help="draw the training curves of a run")
    c.add_argument("--run", required=True)
    b = sub.add_parser("benchmark", help="play the stand-in and some checkpoints on the same fixed games")
    b.add_argument("--run", required=True)
    b.add_argument("--epochs", type=int, nargs="+", required=True)
    b.add_argument("--games", type=int, default=8, help="games per opponent")
    b.add_argument("--turns", type=int, default=2000)
    b.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.command == "curves":
        for f in curves(args.run):
            print(f)
    else:
        benchmark(args.run, args.epochs, args.games, args.turns, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
