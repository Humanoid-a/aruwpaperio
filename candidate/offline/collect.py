"""Collecting training games: many headless matches in parallel, each one recorded.

    python -m candidate.offline.collect --out round0 --games 96 --workers 16 \\
        --opponents enemy=4,basic=1,random=1,hunter=2 --close-share 0.25 --explore 0.03 --model heuristic

--out is a run name (the games go to q_training/raw/<name>) or a path. --model is "heuristic", a weights name in
q_training/weights, or a weights file.

For every game, in --out:
    <game id>.npz    the game record written by the controller itself (agent.q.record): Q's inputs exactly as seen,
                     values, masks, states, accepted moves and the label ingredients, frame by frame
    <game id>.json   the sidecar: the match config, both players' move tapes (replay.py can replay the match exactly
                     and regenerate the inputs with newer code), the engine's final counts, deaths and kills, and a
                     summary of what the strategy did

The candidate plays with QStrategy: the stand-in Q ("heuristic") or a trained network (a weights file), plus the
brief's exploration (--explore, 3% by default when collecting). Seats and respawns vary: each game picks the
candidate's seat at random, and --close-share of the games use the close respawns so the players meet.

The belief: in --belief-share of the games (QConfig.belief_share, 20%, by default) our agents run the real belief
filter; in the others they are given the true world as a certain belief (oracle.py), which is far faster and takes the
belief's noise out of Q's inputs. Each record says which (meta "oracle"). Evaluation always uses the real belief.

Opponents: the runtime check's (enemy, basic, random, hunter), and two for training games that differ from game to
game (each seeded by the game's seed, so a game can be played again):
    noisy   NoisyEnemy (noisy_enemy.py), the framework's EnemyStateMachine with randomness --opponent-randomness
    enemy_copy   our copy of the framework's EnemyStateMachine (NoisyEnemy at randomness 0, which plays exactly like the
            original, move for move) with no randomness at all, whatever --opponent-randomness says
    self    another of our agents, with the opponent model (--opponent-model, by default the candidate's own) and
            behaviour randomness --opponent-randomness (agent/behavior/style.py); not recorded.
    champion   the strong opponent: our agent with a fixed Q (GameSpec.champion: our strongest network, or the
            stand-in), the hand-set K, no exploration, little behaviour randomness It gets the true world
            in an oracle game, and also in a belief game whose spec says opponent_oracle (online.py sets it for a share
            of them: our agent still runs the real belief, but the game runs one belief filter instead of two)

With several workers the costliest games (belief filters) are started first; the results keep the plan's order.
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path

from candidate.offline import locations


@dataclass(frozen=True)
class GameSpec:
    index: int
    opponent: str
    seat: int
    close: bool
    turns: int
    seed: int
    explore: float
    model: str                      # "heuristic" or the path of a weights file
    out: str
    record: bool = True
    opponent_randomness: float = 0.0    # for "noisy" and "self" (see the module notes)
    opponent_model: str | None = None   # for "self": its Q model; None = the candidate's
    oracle: bool = False                # our agents get the true world instead of running the belief filter
    k_model: str | None = None          # the belief's motion model (a neural K file); None = the hand-set K
    opponent_oracle: bool = False       # "self" in a belief game: the opponent alone gets the true world (faster)
    respawns: tuple | None = None       # (y, x) of player 0 and player 1; None: `close` decides (scenarios.py)
    spawn: str | None = None            # the scenario's spawn kind, for the records ("far", "close", "very_close", ...)
    warmup_turns: int = 0               # turns another agent plays our seat before our agent takes over (scenarios.py)
    warmup_agent: str | None = None     # which agent: "enemy", "noisy", "basic", "random", "stand_in"
    champion: str | None = None         # for "champion": its Q ("heuristic" or a weights file); None: the stand-in


def resolve_out(out: str | Path) -> Path:
    """A run name -> q_training/raw/<name>; a path stays a path."""
    p = Path(out)
    return p if p.is_absolute() or len(p.parts) > 1 else locations.raw_run(str(p))


def resolve_model(model: str) -> str:
    """'heuristic' stays; a weights name -> its file in q_training/weights; a path stays a path."""
    return model if model == "heuristic" else str(locations.weights_path(model))


def make_q(model: str):
    """The Q model a worker plays with. A trained network runs on the CPU here: 16 worker processes each opening a CUDA
    context would crowd an 8 GB card, and for one frame at a time the CPU is as fast (about 0.1 ms)."""
    if model == "heuristic":
        from candidate.agent.q.heuristic import HeuristicQ

        return HeuristicQ()
    from candidate.agent.q.network import NetworkQ

    return NetworkQ.load(model, device="cpu")


def opponents():
    from candidate.runtime_check import OPPONENTS

    return OPPONENTS


def make_belief(spec: GameSpec, truth, seat: int):
    """The belief our agent in `seat` runs: the true world in an oracle game, else the belief filter with the game's K
    (None: the default filter, with the hand-set K)."""
    if truth is not None:
        from candidate.offline.oracle import OracleBelief

        return OracleBelief(truth, seat)
    if spec.k_model:
        from candidate.agent.belief.filter import BeliefFilter
        from candidate.agent.belief.neural import load_motion

        return BeliefFilter(motion=load_motion(spec.k_model))
    return None


def make_opponent(spec: GameSpec, truth=None):
    """The opponent of one game, as a framework AgentController, and what it drew for this game (or None). With a
    truth source (an oracle game, or opponent_oracle) the "self" opponent is given the true world."""
    if spec.opponent in ("noisy", "enemy_copy"):
        from candidate.offline.noisy_enemy import NoisyEnemy

        if spec.opponent == "enemy_copy":
            enemy = NoisyEnemy()                     # randomness 0 and the original's Random(42): the original exactly
        else:
            enemy = NoisyEnemy(randomness=spec.opponent_randomness, seed=spec.seed + 1)
        return enemy, enemy.habits()
    if spec.opponent == "champion":
        return _champion(spec, truth)
    if spec.opponent == "self":
        from candidate.agent.behavior.config import BehaviorConfig
        from candidate.agent.controller import Controller
        from candidate.agent.diagnostics import Diagnostics
        from candidate.agent.q.strategy import QStrategy
        from candidate.offline import match

        q = make_q(spec.opponent_model or spec.model)
        strategy = QStrategy(q, behavior=BehaviorConfig(randomness=spec.opponent_randomness), seed=spec.seed + 2,
                             build_inputs=bool(getattr(q, "needs_inputs", False)))
        drawn = dict(randomness=spec.opponent_randomness, model=spec.opponent_model or spec.model,
                     **(strategy.style.changed if strategy.style is not None else {}),
                     **(dict(jog_rate=round(strategy.style.jog_rate, 4)) if strategy.style is not None else {}))
        belief = make_belief(spec, truth, 1 - spec.seat)
        return match.AgentAdapter(Controller(strategy, Diagnostics(), belief=belief)), drawn
    return opponents()[spec.opponent](), None


def play_one(spec: GameSpec) -> dict:
    """Play and record one game. Runs in a worker process."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    from candidate.agent.controller import Controller
    from candidate.agent.diagnostics import Diagnostics
    from candidate.agent.q.record import GameRecorder
    from candidate.agent.q.strategy import QStrategy
    from candidate.offline import match

    started = time.perf_counter()
    truth = None
    if spec.oracle or (spec.opponent_oracle and spec.opponent == "self"):
        from candidate.offline.oracle import EngineTruth

        truth = EngineTruth()
    q = make_q(spec.model)
    respawns = respawns_of(spec)
    label = (f"g{spec.index:04d}-{spec.opponent}-p{spec.seat}{'-close' if spec.close else ''}"
             + (f"-{spec.spawn}" if spec.spawn not in (None, "far", "close") else "")
             + (f"-t{spec.turns}" if spec.turns != 2000 else "") + (f"-w{spec.warmup_turns}" if spec.warmup_turns else ""))
    recorder = GameRecorder(spec.out, label=label, meta=dict(opponent=spec.opponent, seat=spec.seat, close=spec.close,
                                                             seed=spec.seed, model=spec.model, oracle=spec.oracle,
                                                             k_model=None if spec.oracle else spec.k_model,
                                                             spawn=spec.spawn, warmup_turns=spec.warmup_turns,
                                                             warmup_agent=spec.warmup_agent)
                             ) if spec.record else None
    strategy = QStrategy(q, recorder=recorder, explore_rate=spec.explore, seed=spec.seed)
    belief = make_belief(spec, truth if spec.oracle else None, spec.seat)
    if spec.warmup_turns > 0 and spec.warmup_agent:
        from candidate.offline.scenarios import Driven, Handover, warmup_agent

        driven = Driven(strategy)
        ours = match.AgentAdapter(Controller(driven, Diagnostics(), belief=belief))
        candidate = match.Taped(Handover(ours, driven, warmup_agent(spec.warmup_agent, spec.seed), spec.warmup_turns))
    else:
        candidate = match.Taped(match.AgentAdapter(Controller(strategy, Diagnostics(), belief=belief)))
    rival_controller, opponent_draw = make_opponent(spec, truth)
    rival = match.Taped(rival_controller)
    controllers = [rival, rival]
    controllers[spec.seat] = candidate
    controllers[1 - spec.seat] = rival
    result = match.play(controllers, spec.turns, respawns, on_engine=truth.attach if truth is not None else None)
    strategy.close()

    me, them = spec.seat, 1 - spec.seat
    summary = dict(
        spec=spec_dict(spec), label=label, seconds=round(time.perf_counter() - started, 1),
        my_count=result.counts[me], opp_count=result.counts[them], won=result.counts[me] > result.counts[them],
        my_deaths=result.deaths[me], my_kills=result.kills[me], failed_turns=result.failed_turns,
        opponent_draw=opponent_draw,
        stats={k: int(v) for k, v in strategy.stats.items()},
        record=str(recorder.last_path) if recorder is not None and recorder.last_path else None,
        record_error=recorder.last_error if recorder is not None else None,
    )
    if recorder is not None and recorder.last_path is not None:
        sidecar = dict(
            summary,
            config=dict(turns=spec.turns, respawns=[list(r) for r in respawns] if respawns else None, seat=spec.seat,
                        opponent=spec.opponent, warmup_turns=spec.warmup_turns, warmup_agent=spec.warmup_agent),
            tapes=[controllers[0].tape, controllers[1].tape],
            final=dict(counts=list(result.counts), deaths=list(result.deaths), kills=list(result.kills)),
        )
        recorder.last_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1), encoding="utf-8")
    return summary


def spec_dict(spec: GameSpec) -> dict:
    """asdict, tolerant of a spec made by an older GameSpec (a run started before a field was added): such a spec
    reaches the worker without the new field, which then takes its default."""
    return {f.name: getattr(spec, f.name, f.default) for f in fields(GameSpec)}


def _champion(spec: GameSpec, truth=None):
    """The strong opponent: our own agent with a fixed Q (spec.champion: the strongest network we have, or the
    stand-in), the hand-set K (whose belief plays better than the trained ones so far), no exploration, and only the
    behaviour randomness of the spec (small by default). In a game where our agent gets the true world, so does it."""
    from candidate.agent.behavior.config import BehaviorConfig
    from candidate.agent.controller import Controller
    from candidate.agent.diagnostics import Diagnostics
    from candidate.agent.q.strategy import QStrategy
    from candidate.offline import match

    model = getattr(spec, "champion", None) or "heuristic"
    q = make_q(model)
    strategy = QStrategy(q, behavior=BehaviorConfig(randomness=spec.opponent_randomness), seed=spec.seed + 3,
                         build_inputs=bool(getattr(q, "needs_inputs", False)))
    belief = None
    if truth is not None and spec.oracle:
        from candidate.offline.oracle import OracleBelief

        belief = OracleBelief(truth, 1 - spec.seat)
    drawn = dict(randomness=spec.opponent_randomness, model=model)
    return match.AgentAdapter(Controller(strategy, Diagnostics(echo=None), belief=belief)), drawn


def respawns_of(spec: GameSpec):
    """The respawns a game is played with: the spec's own, else the close ones or the default config's."""
    from candidate.offline import match

    if spec.respawns:
        return tuple(tuple(r) for r in spec.respawns)
    return match.CLOSE if spec.close else match.FAR


def parse_mix(text: str) -> list[tuple[str, int]]:
    out = []
    for part in text.split(","):
        name, _, weight = part.partition("=")
        out.append((name.strip(), int(weight or 1)))
    return out


def plan(games: int, mix: list[tuple[str, int]], close_share: float, turns: int, explore: float, model: str, out: str,
         seed: int, record: bool = True, opponent_randomness: float = 0.0, opponent_model: str | None = None,
         belief_share: float = 1.0, k_model: str | None = None) -> list[GameSpec]:
    """The games to play. `belief_share` of them (each drawn at random) run the real belief; the rest get the truth.
    With 1.0 (the default, used by evaluation) no extra random number is drawn, so the plan is as it always was."""
    rng = random.Random(seed)
    names = [n for n, w in mix for _ in range(w)]
    specs = []
    for i in range(games):
        specs.append(GameSpec(
            index=i, opponent=names[i % len(names)], seat=rng.randint(0, 1), close=rng.random() < close_share,
            turns=turns, seed=rng.randrange(2**31), explore=explore, model=model, out=out, record=record,
            opponent_randomness=opponent_randomness, opponent_model=opponent_model,
            oracle=belief_share < 1.0 and rng.random() >= belief_share, k_model=k_model,
        ))
    return specs


def cost(spec: GameSpec) -> float:
    """A game's rough cost, relative to a true-world game (measured at 16 workers): each belief filter run costs most."""
    ours = spec.opponent in ("self", "champion")                     # an opponent that runs our agent
    filters = (not spec.oracle) + (ours and not (spec.oracle or spec.opponent_oracle))
    return 1.0 + 4.0 * filters + 0.5 * ours


def run(specs: list[GameSpec], workers: int, echo=print) -> list[dict]:
    """Play the games; the results come back in the order of `specs`. With several workers the costliest games start
    first, so no long game is left running alone at the end while the other workers wait."""
    results = []
    if workers <= 1:
        for spec in specs:
            results.append(play_one(spec))
            echo(_line(results[-1]))
        return results
    order = sorted(range(len(specs)), key=lambda i: -cost(specs[i]))
    by_index = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, summary in zip(order, pool.map(play_one, [specs[i] for i in order])):
            by_index[i] = summary
            echo(_line(summary))
    return [by_index[i] for i in range(len(specs))]


def _line(s: dict) -> str:
    return (f"{s['label']:28s} {s['my_count']:4d} vs {s['opp_count']:4d} {'WIN ' if s['won'] else 'loss'} "
            f"deaths {s['my_deaths']:2d} kills {s['my_kills']:2d} failed {s['failed_turns']} "
            f"explored {s['stats'].get('explored', 0):3d}  {s['seconds']:.0f}s")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--turns", type=int, default=2000)
    parser.add_argument("--opponents", default="enemy=4,basic=1,random=1,hunter=2")
    parser.add_argument("--close-share", type=float, default=0.25)
    parser.add_argument("--explore", type=float, default=None, help="default: QConfig.exploration_rate (3%%)")
    parser.add_argument("--model", default="heuristic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-randomness", type=float, default=0.0, help="for the noisy and self opponents")
    parser.add_argument("--opponent-model", default=None, help="for the self opponent; default: --model")
    parser.add_argument("--belief-share", type=float, default=None,
                        help="share of games with the real belief; default QConfig.belief_share (the rest get the truth)")
    args = parser.parse_args(argv)
    from candidate.agent.q.config import Q

    if args.explore is None:
        args.explore = Q.exploration_rate
    if args.belief_share is None:
        args.belief_share = Q.belief_share
    out = resolve_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = plan(args.games, parse_mix(args.opponents), args.close_share, args.turns, args.explore,
                 resolve_model(args.model), str(out), args.seed, opponent_randomness=args.opponent_randomness,
                 opponent_model=resolve_model(args.opponent_model) if args.opponent_model else None,
                 belief_share=args.belief_share)
    started = time.perf_counter()
    results = run(specs, args.workers)
    wins = sum(r["won"] for r in results)
    print(f"{len(results)} games in {time.perf_counter() - started:.0f}s: won {wins}, "
          f"deaths {sum(r['my_deaths'] for r in results)}, failed turns {sum(r['failed_turns'] for r in results)}")
    (out / "collect_summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
