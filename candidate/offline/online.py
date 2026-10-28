"""Online training of the Q model: play a batch of games, train on them, keep the new network only if it plays better.

    python -m candidate.offline.online run --name online8 --epochs 30
    python -m candidate.offline.online run --name online8 --epochs 40        # resumes: finished epochs are skipped
    python -m candidate.offline.online smoke                                 # a tiny run, to check it goes through

The games are simulated: the real engine runs headless (no window, no delay between turns), many games at once in
worker processes, so nothing is played in the GUI. Two networks are kept:

    the learner   trained every epoch, never thrown away
    the player    the best network so far (at the start: --start, the hand-set stand-in by default). It plays most of the
                  training games, and best.pt is it. The learner replaces it only by beating it on the gate (step 5).

Each epoch:

  1. play     the epoch's games (--games at first; see "games per epoch"). Our recorded agent is QStrategy with no
              behaviour randomness and the brief's exploration (--explore, 3%: a random valid state now and then,
              without which Q never learns what the states it does not choose are worth). It runs the player, except in
              --learner-share (50%) of the games, where it runs the learner when that differs: the learner has to see
              the results of its own choices, or it never learns from its own mistakes.
              Only --belief-share of the games run the real belief filter; in the others our agents are given the true
              world (oracle.py): much faster, no belief noise in Q's inputs. The results of the belief games are
              reported on their own, since only they show how the agent really plays.
              Its opponent, drawn from the --opponents mix:
                  noisy   NoisyEnemy (noisy_enemy.py): the framework's EnemyStateMachine with its own per-game habits,
                          randomness --enemy-randomness
                  enemy_copy   our copy of the framework's EnemyStateMachine with no randomness: exactly the
                          benchmark opponent, move for move (the games still differ: seats, respawns, our exploration)
                  self    another of our agents with the player's Q, no exploration, per-game behaviour randomness
                          --self-randomness (agent/behavior/style.py); not recorded. --self-oracle-share (0 by
                          default) of the belief games against it give that opponent alone the true world: cheaper, but
                          an opponent that sees everything wins nearly all of them (online7: 2 of 39 games won)
                  champion   the strong opponent: our agent with a fixed Q (--champion: our strongest network, or the
                          stand-in), the hand-set K, no exploration, behaviour randomness --champion-randomness (0.1)
                  enemy, basic, random, hunter   the runtime check's opponents (deterministic)
              Seats alternate at random. Each game is also played in a drawn situation (scenarios.py): where the
              players start (--spawns: far, close, very close, random; always mirrored), how long it is (--turns-mix,
              by default the full length twice as often as each of 1/2, 1/4 and 3/20 of it: short games leave no time
              to win by killing alone), and in --warmup-share (30%) of the games another agent (--warmup-agents) plays
              our seat for the first 10-50% of the turns before our agent takes over in the position it left.
  2. dataset  labels for the recorded frames of this epoch and the --history - 1 (2) epochs before it, every
              --frame-stride (2nd) frame (consecutive frames are nearly the same board): the change in
              score difference over the next QConfig.label_window (100) frames plus, by default (bootstrapping), the
              player's value of the state taken at the end of that window (the learner's while the player is the
              stand-in): a target network, so the learner does not bootstrap from its own estimates and keep its own
              mistakes alive. Near frames count more than far ones (--label-horizons, --label-weights: by default
              frames 1-20, 21-50, 51-100 at 1.0, 0.6, 0.3, scaled so a steady gain gives the same label;
              offline/dataset.py). Several epochs, so that one epoch's swing in play does not swing the next network
              the whole way back.
  3. train    the learner trained further (train.finetune, learning rate --finetune-lr 3e-5, validation checked
              --val-checks (8) times a pass, patience 6: it keeps the
              old weights if no training epoch beats them on the validation games), with all three branches (scalars, grid, probability map): a branch it lacks is
              switched on with its head weights at zero, so its outputs start the same. With no learner yet, a network
              with all three branches from scratch (train.full). --branches staged uses the brief's staged build
              instead, which can drop a branch for good (it dropped both in online6 and online7, at epoch 0).
  4. K        the belief's motion model K trained further on this epoch's games against the enemy state machines and
              our own agent (k_train.py: the opponent's real moves, replayed from the tapes), saved as k.npz; the next
              epoch's belief games run with it. Epoch 0 plays with the hand-set K. --no-train-k skips this step.
  5. gate     --gate-games fixed games (the 4 different games against enemy_copy, the rest against the noisy enemy at
              randomness 0.5, both seats, far and close, the same seeds every epoch), with the real belief and no
              exploration: the learner with this epoch's K. Its mean territory difference is compared with the
              player's on the same games (measured once, when it became the player, with its K then). The learner becomes the player only if its
              advantage on the same games, game by game, is above --gate-margin (0) and at least --gate-z (1) standard
              errors: 16 games are noisy, and online9 accepted two networks that were only as good as the stand-in.
              The games per epoch then grow. --gate-games 0: no gate, the
              learner always plays (the old loop).
  6. record   the epoch's results, training reports and the gate

Before epoch 0 (optional): --pretrain trains the first learner on games recorded elsewhere (e.g. online10:0-6, another
run's epochs 0 to 6), and --start-k starts K (and the player's K, so the gate compares like with like) from a trained
K, while Q starts from --start.

Games per epoch: --games (200) at first, times --grow (1.5) each time the player is replaced, up to --max-games (400).
Exploration: after every --explore-boost-after (3) learners in a row the gate rejects, the exploration rate is raised by
--explore-boost (1.5x), up to --explore-max (0.05: in online12, 0.10-0.15 made every learner worse); it returns to
--explore when a learner is accepted. K is trained every epoch, but our agents play with --play-k: the hand-set K by
default, since the trained K's belief has played worse (decision 43); then the gate compares networks with one K. A network built
from scratch has width --width (2: twice the convolutional channels, agent/q/network.py). Early on the
network changes a lot from epoch to epoch and big mistakes show up in a few games: small batches mean more updates per
hour. Later the improvements are small and the games are noisy: more games per update keep it from chasing noise.

Files, under q_training/online/<name>/ (git-ignored, except the networks q_training/.gitignore keeps):
    config.json              the run's settings
    log.txt                  everything printed
    epoch_000/raw/           the epoch's recorded games: records and sidecars (move tapes: every game can be replayed)
    epoch_000/games.json     one entry per game: opponent and what it drew, seat, result, deaths, kills, states taken
    epoch_000/weights.pt     the learner after this epoch's training, and weights.json, its training report
    epoch_000/k.npz          the belief's motion model after this epoch, and k.json, its report
    epoch_000/gate.json      the gate: both networks' results on the gate games, and the decision
    epoch_000/summary.json   the epoch in numbers: results overall and per opponent, states, training, gate, and the
                             loop's state (learner, player, games per epoch) for resuming
    history.jsonl            one line per finished epoch (its summary), to follow the run
    best.pt, best_k.npz      the player and its K: the network to play in the GUI (CANDIDATE_Q_WEIGHTS /
                             CANDIDATE_K_WEIGHTS)
    latest.pt, latest_k.npz  the newest learner and K
A step whose output exists is skipped, so an interrupted run carries on where it stopped.

Training again on games already played, without playing, is offline.train (python -m candidate.offline.train --raw
q_training/online/<name>/epoch_003/raw --name ...).
"""

import argparse
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from candidate.offline import collect, locations, scenarios
from candidate.offline.evaluate import summarise
from candidate.offline.scenarios import DEFAULT_SPAWNS, DEFAULT_WARMUP_AGENTS

OPPONENT_CHOICES = ("noisy", "self", "enemy_copy", "champion", "enemy", "basic", "random", "hunter")
DEFAULT_OPPONENTS = "noisy=1,self=1,enemy_copy=1,champion=1"


class _Log:
    """Prints and appends to the run's log.txt."""

    def __init__(self, path: Path | None, quiet: bool = False) -> None:
        self.path, self.quiet = path, quiet

    def __call__(self, text: str) -> None:
        if not self.quiet:
            print(text, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")


def epoch_dir(run: Path, epoch: int) -> Path:
    return run / f"epoch_{epoch:03d}"


def summarise_epoch(results: list[dict]) -> dict:
    """The epoch's games in numbers: overall and per opponent (evaluate.summarise), and the share of each state."""
    out = dict(all=summarise(results), opponents={})
    belief_games = [r for r in results if not r["spec"].get("oracle")]
    out["belief_games"] = summarise(belief_games) if belief_games else None          # how it really plays
    out["oracle_games"] = len(results) - len(belief_games)
    out["opponent_oracle_games"] = sum(bool(r["spec"].get("opponent_oracle")) for r in belief_games)
    out["scenarios"] = dict(spawn=dict(Counter(r["spec"].get("spawn") or ("close" if r["spec"]["close"] else "far")
                                               for r in results)),
                            turns=dict(Counter(str(r["spec"]["turns"]) for r in results)),
                            warmup=dict(Counter(r["spec"].get("warmup_agent") or "none" for r in results)))
    for name in sorted({r["spec"]["opponent"] for r in results}):
        out["opponents"][name] = summarise([r for r in results if r["spec"]["opponent"] == name])
    states = Counter()
    for r in results:
        states.update({k[len("state_"):]: v for k, v in r["stats"].items() if k.startswith("state_")})
    total = sum(states.values()) or 1
    out["state_share"] = {k: round(v / total, 4) for k, v in sorted(states.items())}
    out["forced"] = {k: sum(r["stats"].get(k, 0) for r in results) for k in ("forced_death", "forced_safety", "forced_lookahead")}
    return out


# ---- the gate --------------------------------------------------------------------------------------------------------


def gate_specs(games: int, turns: int, seed: int = 7) -> list[collect.GameSpec]:
    """The gate's fixed games (report.benchmark_specs): the 4 different games against enemy_copy, the rest against the
    noisy enemy (randomness 0.5), both seats, far and close respawns, the same seeds every epoch; real belief, no
    exploration, not recorded."""
    from candidate.offline.report import benchmark_specs

    return benchmark_specs(max(1, games // 2), turns, seed)


def default_turns_mix(turns: int) -> str:
    """The game lengths drawn by default: the full length twice as often as each of 1/2, 1/4 and 3/20 of it."""
    return f"{turns}=2,{max(1, turns // 2)}=1,{max(1, turns // 4)}=1,{max(1, int(turns * 0.15))}=1"


def gate_decision(player: dict, learner: dict, margin: float, z: float) -> tuple[bool, dict | None]:
    """Whether the learner replaces the player. On the same games (the gate stores each game's difference), the
    learner's advantage per game must be above `margin` and at least `z` standard errors: a learner that is only as
    good, or better by less than the noise of 16 games, does not replace it. Without per-game results: means."""
    p, q = np.asarray(player.get("diffs") or [], dtype=float), np.asarray(learner.get("diffs") or [], dtype=float)
    if len(p) == len(q) and len(q) > 1:
        d = q - p
        mean, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
        paired = dict(mean=mean, se=se, better=int((d > 0).sum()), games=len(d))
        return bool(mean > margin and mean >= z * se), paired
    return bool(learner["mean_diff"] > player["mean_diff"] + margin), None


def gate_score(model: str, k_model: str | None, specs: list[collect.GameSpec], workers: int) -> dict:
    """One network (and its K) on the gate games: mean territory difference, wins, and per opponent."""
    jobs = [replace(s, model=model, opponent_model=model, k_model=k_model) for s in specs]
    results = collect.run(jobs, workers, echo=lambda line: None)
    diff = np.array([r["my_count"] - r["opp_count"] for r in results], dtype=float)
    per = {}
    for name in sorted({s.opponent for s in specs}):
        d = [r["my_count"] - r["opp_count"] for s, r in zip(specs, results) if s.opponent == name]
        per[name] = dict(games=len(d), wins=sum(x > 0 for x in d), mean_diff=float(np.mean(d)))
    return dict(model=model, k_model=k_model, games=len(results), wins=int(sum(r["won"] for r in results)),
                mean_diff=float(diff.mean()), se=float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0,
                deaths=int(sum(r["my_deaths"] for r in results)), opponents=per, diffs=diff.tolist())


def _gate_line(name: str, s: dict) -> str:
    per = ", ".join(f"{o} {v['wins']}/{v['games']} ({v['mean_diff']:+.0f})" for o, v in s["opponents"].items())
    return f"{name}: won {s['wins']}/{s['games']}, mean difference {s['mean_diff']:+.1f} +- {s['se']:.1f} ({per})"


# ---- the loop ----------------------------------------------------------------------------------------------------------


def run(name: str, epochs: int = 10, games: int = 200, workers: int = 16, turns: int = 2000,
        opponents: str = DEFAULT_OPPONENTS, enemy_randomness: float = 0.2, self_randomness: float = 0.2,
        explore: float | None = None, belief_share: float | None = None, close_share: float = 0.25, start: str = "heuristic", history: int = 3,
        bootstrap: bool = True, train_k: bool = True, train_epochs: int = 20, patience: int = 6, device: str | None = None, seed: int = 0,
        root: Path | None = None, quiet: bool = False, self_oracle_share: float = 0.0, branches: str = "all",
        gate_games: int = 16, gate_margin: float = 0.0, learner_share: float = 0.5, max_games: int = 400,
        grow: float = 1.5, frame_stride: int = 2, finetune_lr: float = 3e-5, gate_z: float = 1.0,
        spawns: str = DEFAULT_SPAWNS, turns_mix: str | None = None, warmup_share: float = 0.3,
        warmup_agents: str = DEFAULT_WARMUP_AGENTS, pretrain: list[str] | None = None, pretrain_epochs: int = 40,
        start_k: str | None = None, champion: str = "heuristic", champion_randomness: float = 0.1, width: int = 2,
        explore_boost_after: int = 3, explore_boost: float = 1.5, explore_max: float = 0.05,
        play_k: str = "hand", label_horizons: tuple | None = None, label_weights: tuple | None = None,
        val_checks: int = 8) -> list[dict]:
    """Run (or resume) an online training run. Returns the summaries of all its finished epochs. `explore` None is
    QConfig.exploration_rate."""
    from candidate.agent.q.config import Q
    from candidate.agent.q.network import BRANCHES
    from candidate.offline import train as trainer
    from candidate.offline.dataset import load_dataset

    explore = Q.exploration_rate if explore is None else explore
    label_horizons = tuple(Q.label_horizons if label_horizons is None else label_horizons)
    label_weights = tuple(Q.label_weights if label_weights is None else label_weights)
    from candidate.offline.dataset import label_weights as _check_labels

    _check_labels(Q.label_window, label_horizons, label_weights)            # a bad setting fails now, not in epoch 0
    belief_share = Q.belief_share if belief_share is None else belief_share
    if branches not in ("all", "staged"):
        raise ValueError("branches is 'all' or 'staged'")
    mix = collect.parse_mix(opponents)
    unknown = [n for n, _ in mix if n not in OPPONENT_CHOICES]
    if unknown:
        raise ValueError(f"unknown opponents {unknown}; choose from {OPPONENT_CHOICES}")
    run_dir = (root or locations.ONLINE) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _Log(run_dir / "log.txt", quiet)
    settings = dict(name=name, games=games, max_games=max_games, grow=grow, turns=turns, opponents=opponents,
                    enemy_randomness=enemy_randomness, self_randomness=self_randomness, explore=explore,
                    belief_share=belief_share, self_oracle_share=self_oracle_share, close_share=close_share, start=start,
                    history=history, bootstrap=bootstrap, branches=branches, gate_games=gate_games,
                    gate_margin=gate_margin, gate_z=gate_z, learner_share=learner_share, frame_stride=frame_stride,
                    finetune_lr=finetune_lr, spawns=spawns, turns_mix=turns_mix or default_turns_mix(turns),
                    warmup_share=warmup_share, warmup_agents=warmup_agents, pretrain=pretrain,
                    pretrain_epochs=pretrain_epochs, start_k=start_k, champion=champion, play_k=play_k,
                    label_horizons=list(label_horizons), label_weights=list(label_weights), val_checks=val_checks,
                    champion_randomness=champion_randomness, width=width, explore_boost_after=explore_boost_after,
                    explore_boost=explore_boost, explore_max=explore_max,
                    train_epochs=train_epochs, patience=patience, seed=seed)
    (run_dir / "config.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")

    start_model = collect.resolve_model(start)
    if start_k is not None and not Path(start_k).exists():
        raise FileNotFoundError(f"--start-k {start_k} does not exist")
    champion = collect.resolve_model(champion)
    if play_k not in ("hand", "trained"):
        raise ValueError("play_k is 'hand' or 'trained'")

    def playing(k):                                                   # the K our agents play with
        return k if play_k == "trained" else None

    state = dict(learner=start_model, player=start_model, player_k=start_k, player_score=None, games=games,
                 current_k=start_k, explore=explore, rejected=0)    # the loop's state: what the next epoch plays and trains
    if start_k:
        log(f"[K] starting from {start_k} (the player is measured with it too, so the gate compares like with like)")
    if pretrain:
        state["learner"] = pretrained(run_dir, pretrain, start_model, frame_stride, bootstrap, branches, pretrain_epochs,
                                      patience, finetune_lr, device, seed, log, quiet, width,
                                      (label_horizons, label_weights))
    gated = gate_games > 0
    gates = gate_specs(gate_games, turns) if gated else []
    spawn_mix = scenarios.parse_weights(spawns)
    unknown = [n for n, _ in spawn_mix if n not in scenarios.SPAWN_CHOICES]
    agents = [a.strip() for a in warmup_agents.split(",") if a.strip()]
    unknown += [a for a in agents if a not in scenarios.WARMUP_CHOICES]
    if unknown:
        raise ValueError(f"unknown spawns or warm-up agents {unknown}")
    turn_mix = scenarios.parse_weights(turns_mix or default_turns_mix(turns), int)
    summaries: list[dict] = []
    for epoch in range(epochs):
        here = epoch_dir(run_dir, epoch)
        summary_file = here / "summary.json"
        weights_file = here / "weights.pt"
        k_file = here / "k.npz"
        if summary_file.exists() and weights_file.exists():
            done = json.loads(summary_file.read_text(encoding="utf-8"))
            summaries.append(done)
            if "state" in done:
                state = dict(done["state"])
            else:                                                        # an epoch of the loop before the gate
                state.update(learner=str(weights_file), player=str(weights_file),
                             current_k=str(k_file) if k_file.exists() else state["current_k"])
            log(f"epoch {epoch}: done earlier ({summary_file})")
            continue
        started = time.perf_counter()
        learner, player = state["learner"], state["player"]
        n_games = int(state["games"])
        explore_now = float(state.get("explore", explore))
        log(f"==== epoch {epoch}: {n_games} games; player {_short(player)}, learner {_short(learner)}; "
            f"exploration {explore_now:.3f}; "
            f"against {opponents} ====")

        # 1. play
        raw = here / "raw"
        games_file = here / "games.json"
        if games_file.exists():
            results = json.loads(games_file.read_text(encoding="utf-8"))
            log(f"[play] {len(results)} games played earlier")
        else:
            shutil.rmtree(raw, ignore_errors=True)                       # a batch cut short is played again
            raw.mkdir(parents=True)
            specs = collect.plan(n_games, mix, close_share, turns, explore_now, player, str(raw), seed * 100_003 + epoch,
                                 opponent_randomness=0.0, opponent_model=player, belief_share=belief_share,
                                 k_model=playing(state["current_k"]))
            specs = [_with_randomness(s, enemy_randomness, self_randomness, self_oracle_share, champion,
                                      champion_randomness) for s in specs]
            specs = [scenarios.with_scenario(s, spawn_mix, turn_mix, warmup_share, agents) for s in specs]
            if learner != player and learner_share > 0:
                specs = [replace(s, model=learner) if random.Random(s.seed + 3).random() < learner_share else s
                         for s in specs]
            played = time.perf_counter()
            results = collect.run(specs, workers, echo=log)
            log(f"[play] {len(results)} games in {time.perf_counter() - played:.0f}s "
                f"({sum(r['spec']['model'] == learner != player for r in results)} by the learner)")
            games_file.write_text(json.dumps(results, indent=1), encoding="utf-8")
        played_summary = summarise_epoch(results)
        sc = played_summary["scenarios"]
        log(f"[play] situations: spawns {sc['spawn']}, turns {sc['turns']}, warm-up {sc['warmup']}")
        a = played_summary["all"]
        log(f"[play] won {a['wins']}/{a['games']}, mean difference {a['mean_diff']:+.1f}, deaths {a['deaths']}, "
            f"kills {a['kills']}; " + ", ".join(f"{k}: won {v['wins']}/{v['games']} ({v['mean_diff']:+.1f})"
                                                 for k, v in played_summary["opponents"].items()))
        b = played_summary["belief_games"]
        if b:
            log(f"[play] with the real belief ({b['games']} games): won {b['wins']}, mean difference "
                f"{b['mean_diff']:+.1f}, deaths {b['deaths']}; {played_summary['oracle_games']} games had the true world")

        # 2. dataset
        first = max(0, epoch - history + 1)
        dirs = [epoch_dir(run_dir, e) / "raw" for e in range(first, epoch + 1)]
        network, target = None, (player if player != "heuristic" else learner)
        if bootstrap and target != "heuristic":                         # the player: a target that does not chase itself
            from candidate.agent.q.network import load_network

            network = load_network(target)
        data = load_dataset(dirs, bootstrap=network, frame_stride=frame_stride, horizons=label_horizons,
                            weights=label_weights, echo=None)
        log(f"[dataset] {len(data.games)} games, {len(data)} frames from epochs {first}..{epoch}")

        # 3. train
        if weights_file.exists():
            log(f"[train] {weights_file} exists")
            report = json.loads(weights_file.with_suffix(".json").read_text(encoding="utf-8"))["report"]
        else:
            fit_args = dict(epochs=train_epochs, patience=patience, checks=val_checks)
            echo = None if quiet else log
            if learner == "heuristic":
                if branches == "all":
                    log("[train] no network yet: all three branches from scratch")
                    model, report = trainer.full(data, seed=seed + epoch, device=device, echo=echo, width=width,
                                                 **fit_args)
                else:
                    log("[train] no network yet: the staged build from scratch")
                    model, report = trainer.staged(data, seed=seed + epoch, device=device, echo=echo, **fit_args)
            else:
                from candidate.agent.q.network import load_network

                log(f"[train] fine-tuning {learner}")
                model, report = trainer.finetune(load_network(learner), data, seed=seed + epoch, device=device,
                                                 echo=echo, branches=BRANCHES if branches == "all" else None,
                                                 lr=finetune_lr, **fit_args)
            trainer.export(model, weights_file, report, dict(name=f"{name}-epoch{epoch:03d}", online=name, epoch=epoch,
                                                              trained_from=learner, data=[str(d) for d in dirs]))
            shutil.copyfile(weights_file, run_dir / "latest.pt")
            shutil.copyfile(weights_file.with_suffix(".json"), run_dir / "latest.json")
        vr2 = report.get("val_r2")
        log(f"[train] -> {weights_file}: branches {'+'.join(report['branches'])}, validation {report['val']:.2f} "
            f"(constant baseline {report.get('val_constant_baseline', float('nan')):.2f}, "
            f"R^2 {vr2 if vr2 is not None else float('nan'):.3f})")

        # 4. K
        k_report = None
        if train_k:
            from candidate.offline import k_train

            if k_file.exists():
                log(f"[K] {k_file} exists")
                k_report = json.loads(k_file.with_suffix(".json").read_text(encoding="utf-8"))
            elif k_train.sidecars([raw]):
                log(f"[K] training the motion model further ({state['current_k'] or 'from scratch'})")
                k_report = k_train.train_step([raw], k_file, state["current_k"], workers=workers, device=device,
                                              seed=seed + epoch, echo=log)
                shutil.copyfile(k_file, run_dir / "latest_k.npz")
            else:
                log("[K] no games against the enemy machines or our agent this epoch: K unchanged")
            if k_file.exists():
                state["current_k"] = str(k_file)

        # 5. gate
        gate = None
        new_state = dict(state, learner=str(weights_file))
        if gated:
            gate_file = here / "gate.json"
            if gate_file.exists():
                gate = json.loads(gate_file.read_text(encoding="utf-8"))
            else:
                player_score = state["player_score"]
                if player_score is None:
                    log(f"[gate] the player ({_short(player)}) on the {len(gates)} gate games, once")
                    player_score = gate_score(player, playing(state["player_k"]), gates, workers)
                log(f"[gate] the learner on the {len(gates)} gate games")
                learner_score = gate_score(str(weights_file), playing(state["current_k"]), gates, workers)
                accepted, paired = gate_decision(player_score, learner_score, gate_margin, gate_z)
                gate = dict(player=player_score, learner=learner_score, accepted=accepted, margin=gate_margin, z=gate_z,
                            paired=paired)
                gate_file.write_text(json.dumps(gate, indent=1), encoding="utf-8")
            log("[gate] " + _gate_line("player ", gate["player"]))
            log("[gate] " + _gate_line("learner", gate["learner"]))
            if gate.get("paired"):
                p = gate["paired"]
                log(f"[gate] learner - player on the same games: {p['mean']:+.1f} +- {p['se']:.1f} "
                    f"(better in {p['better']}/{p['games']}; needs > {gate_margin:g} and >= {gate_z:g} SE)")
            if gate["accepted"]:
                grown = min(max_games, int(math.ceil(state["games"] * grow)))
                new_state.update(player=str(weights_file), player_k=state["current_k"], player_score=gate["learner"],
                                 games=grown, explore=explore, rejected=0)
                shutil.copyfile(weights_file, run_dir / "best.pt")
                if state["current_k"] and play_k == "trained":            # the K it was measured with
                    shutil.copyfile(state["current_k"], run_dir / "best_k.npz")
                log(f"[gate] the learner is the new player (best.pt); {grown} games per epoch from now on")
            else:
                rejected = int(state.get("rejected", 0)) + 1
                boosted = float(state.get("explore", explore))
                if explore_boost_after and rejected % explore_boost_after == 0:
                    boosted = min(explore_max, max(boosted, 0.01) * explore_boost)
                new_state.update(player_score=gate["player"], rejected=rejected, explore=boosted)
                log(f"[gate] the player stays ({_short(player)}); {rejected} learners in a row rejected"
                    + (f"; exploration up to {boosted:.3f}" if boosted != float(state.get("explore", explore)) else ""))
        else:
            new_state.update(player=str(weights_file), player_k=state["current_k"])
        state = new_state

        # 6. record
        summary = dict(
            epoch=epoch, played_with=player, learner_played=learner if learner != player else None,
            weights=str(weights_file), seconds=round(time.perf_counter() - started, 1),
            games=played_summary, dataset=dict(games=len(data.games), frames=len(data), epochs=[str(d) for d in dirs]),
            train=dict(branches=report["branches"], device=report.get("device"), start_val=report.get("start_val"),
                       val=report["val"], test=report["test"], test_constant_baseline=report["test_constant_baseline"],
                       test_r2=report["test_r2"], val_constant_baseline=report.get("val_constant_baseline"),
                       val_r2=report.get("val_r2")),
            k=None if k_report is None else dict(file=state["current_k"], val_nll=k_report.get("val_nll"),
                                                 validation=k_report.get("validation")),
            gate=None if gate is None else dict(accepted=gate["accepted"],
                                                player={k: v for k, v in gate["player"].items() if k != "diffs"},
                                                learner={k: v for k, v in gate["learner"].items() if k != "diffs"}),
            state=dict(state),                                           # a copy: the next epoch changes its own
        )
        summary_file.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        summaries.append(summary)
        log(f"epoch {epoch} finished in {summary['seconds']:.0f}s")
    return summaries


def pretrain_dirs(items: list[str]) -> list[Path]:
    """The folders of recorded games to pretrain on. Each item is 'run:a-b' (that online run's epochs a..b), 'run'
    (all its epochs), an epoch folder (its raw/ is used), or any folder or file of recorded games."""
    out: list[Path] = []
    for item in items:
        name, _, span = item.partition(":")
        run = locations.ONLINE / name
        if not Path(item).exists() and run.is_dir():
            epochs = sorted(p for p in run.glob("epoch_*") if (p / "raw").is_dir())
            if span:
                lo, _, hi = span.partition("-")
                keep = range(int(lo), int(hi or lo) + 1)
                epochs = [p for p in epochs if int(p.name.split("_")[1]) in keep]
            out += [p / "raw" for p in epochs]
            continue
        p = Path(item)
        if not p.exists():
            raise FileNotFoundError(f"--pretrain {item}: no such folder, file or online run")
        out.append(p / "raw" if (p / "raw").is_dir() else p)
    if not out:
        raise ValueError(f"--pretrain {items}: no recorded games found")
    return out


def pretrained(run_dir: Path, items: list[str], start_model: str, frame_stride: int, bootstrap: bool, branches: str,
               epochs: int, patience: int, finetune_lr: float, device, seed: int, log, quiet: bool, width: int = 1,
               label_shape: tuple | None = None) -> str:
    """Train the first learner on games recorded elsewhere, once (run_dir/pretrain/weights.pt; skipped if it exists).
    From the --start network if there is one (fine-tuned, all branches), else from scratch. Labels bootstrap from the
    --start network when there is one."""
    from candidate.agent.q.network import BRANCHES, load_network
    from candidate.offline import train as trainer
    from candidate.offline.dataset import load_dataset

    weights = run_dir / "pretrain" / "weights.pt"
    if weights.exists():
        log(f"[pretrain] {weights} exists")
        return str(weights)
    dirs = pretrain_dirs(items)
    network = load_network(start_model) if bootstrap and start_model != "heuristic" else None
    shape = dict(horizons=label_shape[0], weights=label_shape[1]) if label_shape else {}
    data = load_dataset(dirs, bootstrap=network, frame_stride=frame_stride, echo=None, **shape)
    log(f"[pretrain] {len(data.games)} games, {len(data)} frames from {len(dirs)} folders")
    echo = None if quiet else log
    fit_args = dict(epochs=epochs, patience=patience)
    if start_model == "heuristic":
        if branches == "all":
            model, report = trainer.full(data, seed=seed, device=device, echo=echo, width=width, **fit_args)
        else:
            model, report = trainer.staged(data, seed=seed, device=device, echo=echo, **fit_args)
    else:
        model, report = trainer.finetune(load_network(start_model), data, seed=seed, device=device, echo=echo,
                                         branches=BRANCHES if branches == "all" else None, lr=finetune_lr, **fit_args)
    trainer.export(model, weights, report, dict(name=f"{run_dir.name}-pretrain", online=run_dir.name, pretrain=True,
                                                trained_from=start_model, data=[str(d) for d in dirs]))
    vr2 = report.get("val_r2")
    log(f"[pretrain] -> {weights}: branches {'+'.join(report['branches'])}, validation {report['val']:.2f} "
        f"(R^2 {vr2 if vr2 is not None else float('nan'):.3f}); it is the first learner")
    return str(weights)


def _short(model: str) -> str:
    """'heuristic', or a weights file as <epoch dir>/<file>."""
    if model == "heuristic":
        return model
    p = Path(model)
    return f"{p.parent.name}/{p.name}"


def _with_randomness(spec: collect.GameSpec, enemy: float, self_: float, self_oracle_share: float = 0.0,
                     champion: str = "heuristic", champion_randomness: float = 0.1) -> collect.GameSpec:
    """The opponent's randomness, and for a belief game against our own agent whether that opponent gets the true
    world (drawn from the game's own seed, so the plan's other draws stay as they were)."""
    if spec.opponent == "noisy":
        return replace(spec, opponent_randomness=enemy)
    if spec.opponent == "champion":
        return replace(spec, opponent_randomness=champion_randomness, champion=champion)
    if spec.opponent == "self":
        truth = not spec.oracle and random.Random(spec.seed).random() < self_oracle_share
        return replace(spec, opponent_randomness=self_, opponent_oracle=truth)
    return spec


def smoke(device: str | None = None, workers: int = 2, quiet: bool = False) -> list[dict]:
    """Two tiny epochs (6 short games each, one training epoch per stage, a 2-game gate) in q_training/smoke/online,
    emptied first: the first trains a network from the stand-in's games, the second plays with it and fine-tunes it."""
    root = locations.SMOKE / "online"
    shutil.rmtree(root, ignore_errors=True)
    started = time.perf_counter()
    out = run("smoke", epochs=2, games=6, workers=workers, turns=60, opponents=DEFAULT_OPPONENTS, enemy_randomness=1.0,
              self_randomness=1.0, explore=0.2, close_share=0.5, train_epochs=1, patience=1, device=device, root=root,
              quiet=quiet, gate_games=2, warmup_share=0.5)
    if not quiet:
        print(f"online smoke test passed in {time.perf_counter() - started:.0f}s: "
              f"{len(out)} epochs, trained on {out[-1]['train']['device']}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run (or resume) an online training run")
    r.add_argument("--name", required=True)
    r.add_argument("--epochs", type=int, default=10)
    r.add_argument("--games", type=int, default=200, help="games per epoch at first")
    r.add_argument("--max-games", type=int, default=400, help="games per epoch at most")
    r.add_argument("--grow", type=float, default=1.5, help="games per epoch times this each time the player is replaced")
    r.add_argument("--workers", type=int, default=16, help="games played at once")
    r.add_argument("--turns", type=int, default=2000, help="turns per player per game")
    r.add_argument("--opponents", default=DEFAULT_OPPONENTS, help=f"mix, from {', '.join(OPPONENT_CHOICES)}")
    r.add_argument("--enemy-randomness", type=float, default=0.2, help="NoisyEnemy's randomness, 0..1")
    r.add_argument("--self-randomness", type=float, default=0.2, help="the self opponent's behaviour randomness, 0..1")
    r.add_argument("--champion", default="heuristic",
                   help="the champion opponent's Q: 'heuristic' or a weights file (our strongest network)")
    r.add_argument("--champion-randomness", type=float, default=0.1, help="the champion's behaviour randomness, 0..1")
    r.add_argument("--width", type=int, default=2,
                   help="width of a network built from scratch (multiplies the convolutional branches' channels)")
    r.add_argument("--explore-boost-after", type=int, default=3,
                   help="after this many learners in a row are rejected, raise the exploration (0: never)")
    r.add_argument("--explore-boost", type=float, default=1.5, help="the factor it is raised by")
    r.add_argument("--explore-max", type=float, default=0.05,
                   help="the most it is raised to (online12: 0.10-0.15 made every learner worse)")
    r.add_argument("--label-horizons", default=None,
                   help="where the label's segments end, e.g. '20,50' (frames 1-20, 21-50, 51-100); default QConfig's")
    r.add_argument("--label-weights", default=None,
                   help="the segments' weights, nearest first, e.g. '1,0.6,0.3' (scaled to keep the label's intensity; "
                        "'1,1,1' is the unweighted label); default QConfig's")
    r.add_argument("--play-k", choices=("hand", "trained"), default="hand",
                   help="the K our agents play with in training and gate games: the hand-set one (default: the trained "
                        "K's belief played worse, decision 43) or the trained one; K is trained and reported either way")
    r.add_argument("--explore", type=float, default=None,
                   help="the learner's exploration rate; default QConfig.exploration_rate (the brief's 3%%)")
    r.add_argument("--belief-share", type=float, default=None,
                   help="share of games with the real belief filter; default QConfig.belief_share (20%%)")
    r.add_argument("--self-oracle-share", type=float, default=0.0,
                   help="share of the belief games against our own agent in which that opponent gets the true world")
    r.add_argument("--close-share", type=float, default=0.25)
    r.add_argument("--start", default="heuristic", help="'heuristic', a weights name or a weights file")
    r.add_argument("--start-k", default=None,
                   help="a trained K (.npz) to start from, e.g. another run's latest_k.npz; Q still starts from --start")
    r.add_argument("--pretrain", nargs="+", default=None,
                   help="recorded games to train the first learner on before epoch 0: 'run:a-b' (an online run's "
                        "epochs a..b, e.g. online10:0-6), 'run', an epoch folder, or a folder of games")
    r.add_argument("--pretrain-epochs", type=int, default=40, help="training epochs (passes) for the pretraining")
    r.add_argument("--history", type=int, default=3, help="train on the games of this many latest epochs")
    r.add_argument("--branches", choices=("all", "staged"), default="all",
                   help="'all': every network uses scalars, grid and probability map; 'staged': the brief's staged "
                        "build, which can drop a branch for good")
    r.add_argument("--gate-games", type=int, default=16,
                   help="fixed real-belief games the learner must do as well as the player on (0: no gate)")
    r.add_argument("--gate-margin", type=float, default=0.0, help="cells per game the learner must beat the player by")
    r.add_argument("--gate-z", type=float, default=1.0,
                   help="standard errors (paired, over the gate games) the learner's advantage must reach")
    r.add_argument("--spawns", default=DEFAULT_SPAWNS,
                   help=f"weighted mix of start positions, from {', '.join(scenarios.SPAWN_CHOICES)}")
    r.add_argument("--turns-mix", default=None,
                   help="weighted mix of game lengths, e.g. '2000=2,1000=1,500=1,300=1' (default: that, scaled to --turns)")
    r.add_argument("--warmup-share", type=float, default=0.3,
                   help="share of the games in which another agent plays our seat first (scenarios.py)")
    r.add_argument("--warmup-agents", default=DEFAULT_WARMUP_AGENTS,
                   help=f"the agents that may play the warm-up, from {', '.join(scenarios.WARMUP_CHOICES)}")
    r.add_argument("--frame-stride", type=int, default=2,
                   help="train on every k-th frame of a game (consecutive frames are nearly the same board)")
    r.add_argument("--learner-share", type=float, default=0.5,
                   help="share of the training games our agent plays with the learner (when it is not the player)")
    r.add_argument("--no-bootstrap", dest="bootstrap", action="store_false",
                   help="do not add the learner's value at the end of each label window (it is added by default, "
                        "so a label looks past the window)")
    r.add_argument("--no-train-k", dest="train_k", action="store_false",
                   help="do not train the belief's motion model K each epoch (belief games then use the hand-set K)")
    r.add_argument("--train-epochs", type=int, default=20)
    r.add_argument("--patience", type=int, default=6)
    r.add_argument("--finetune-lr", type=float, default=3e-5,
                   help="the learning rate when training a network further (online14: at 3e-4 it never moved)")
    r.add_argument("--val-checks", type=int, default=8,
                   help="validation checks per training pass; the best point is kept (it can lie inside the first pass)")
    r.add_argument("--device", default=None, help="cuda or cpu; by default the GPU if it really works")
    r.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("smoke", help="two tiny epochs, to check the loop goes through")
    s.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        smoke(args.device)
        return 0
    run(args.name, args.epochs, args.games, args.workers, args.turns, args.opponents, args.enemy_randomness,
        args.self_randomness, args.explore, args.belief_share, args.close_share, args.start, args.history,
        args.bootstrap, args.train_k, args.train_epochs, args.patience, args.device, args.seed,
        self_oracle_share=args.self_oracle_share, branches=args.branches, gate_games=args.gate_games,
        gate_margin=args.gate_margin, learner_share=args.learner_share, max_games=args.max_games, grow=args.grow,
        frame_stride=args.frame_stride, finetune_lr=args.finetune_lr, gate_z=args.gate_z, spawns=args.spawns,
        turns_mix=args.turns_mix, warmup_share=args.warmup_share, warmup_agents=args.warmup_agents,
        pretrain=args.pretrain, pretrain_epochs=args.pretrain_epochs, start_k=args.start_k, champion=args.champion,
        champion_randomness=args.champion_randomness, width=args.width, explore_boost_after=args.explore_boost_after,
        explore_boost=args.explore_boost, explore_max=args.explore_max, play_k=args.play_k,
        label_horizons=tuple(int(x) for x in args.label_horizons.split(",")) if args.label_horizons else None,
        label_weights=tuple(float(x) for x in args.label_weights.split(",")) if args.label_weights else None,
        val_checks=args.val_checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
