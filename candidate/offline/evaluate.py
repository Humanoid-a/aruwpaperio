"""A/B evaluation: several Q variants on the same games (brief, "Acceptance checks": win rate against
EnemyStateMachine, 200 games per variant).

    python -m candidate.offline.evaluate --variants heuristic q_round0 \\
        --games 40 --opponent enemy --workers 16

Every variant plays exactly the same list of games (the same seats, respawns and seeds), with no exploration, so the
comparison is paired: the per-game difference between two variants removes most of the game-to-game noise. Reports,
per variant, the win rate, the mean final score difference (my count minus the opponent's) with its standard error,
deaths and kills, and for each later variant the paired difference against the first. --record DIR also records the
games (they are on-policy data for the next training round).
"""

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from candidate.offline import collect


def summarise(results: list[dict]) -> dict:
    diff = np.array([r["my_count"] - r["opp_count"] for r in results], dtype=float)
    n = len(diff)
    return dict(
        games=n, wins=int(sum(r["my_count"] > r["opp_count"] for r in results)),
        draws=int(sum(r["my_count"] == r["opp_count"] for r in results)),
        mean_diff=float(diff.mean()) if n else 0.0,
        se_diff=float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        mean_mine=float(np.mean([r["my_count"] for r in results])) if n else 0.0,
        deaths=int(sum(r["my_deaths"] for r in results)), kills=int(sum(r["my_kills"] for r in results)),
        failed_turns=int(sum(r["failed_turns"] for r in results)),
    )


def paired(a: list[dict], b: list[dict]) -> dict:
    d = np.array([(y["my_count"] - y["opp_count"]) - (x["my_count"] - x["opp_count"]) for x, y in zip(a, b)], float)
    return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0,
                better=int((d > 0).sum()), worse=int((d < 0).sum()))


def compare(variants: list[str], games: int = 40, opponent: str = "enemy", turns: int = 2000, workers: int = 16,
            seed: int = 100, close_share: float = 0.0, record: str | Path | None = None, echo=print) -> dict:
    """Play every variant on the same games; the report as described above."""
    variants = [collect.resolve_model(v) for v in variants]
    base = collect.plan(games, [(opponent, 1)], close_share, turns, 0.0, "heuristic", str(record or ""), seed,
                        record=record is not None)
    report = dict(opponent=opponent, games=games, turns=turns, variants={})
    results: dict[str, list[dict]] = {}
    for variant in variants:
        out = str(Path(record) / Path(variant).stem) if record else ""
        specs = [replace(s, model=variant, out=out) for s in base]
        started = time.perf_counter()
        if echo:
            echo(f"--- {variant}")
        results[variant] = collect.run(specs, workers, echo=echo or (lambda line: None))
        summary = summarise(results[variant])
        summary["seconds"] = round(time.perf_counter() - started, 1)
        report["variants"][variant] = summary
        if echo:
            echo(f"{variant}: won {summary['wins']}/{summary['games']}, mean difference {summary['mean_diff']:+.1f} "
                 f"+- {summary['se_diff']:.1f}, mean territory {summary['mean_mine']:.1f}, deaths {summary['deaths']}, "
                 f"kills {summary['kills']}, failed turns {summary['failed_turns']}")
    first = variants[0]
    for variant in variants[1:]:
        p = paired(results[first], results[variant])
        report["variants"][variant]["paired_vs_first"] = p
        if echo:
            echo(f"{variant} vs {first}: paired difference {p['mean']:+.1f} +- {p['se']:.1f} "
                 f"(better in {p['better']} games, worse in {p['worse']})")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variants", nargs="+", required=True, help='"heuristic", weights names or weights files')
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--opponent", default="enemy")
    parser.add_argument("--close-share", type=float, default=0.0)
    parser.add_argument("--turns", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--record", default=None, help="also record the games (a run name in q_training/raw, or a path)")
    parser.add_argument("--out", default=None, help="write the report (JSON) here")
    args = parser.parse_args(argv)
    record = collect.resolve_out(args.record) if args.record else None
    report = compare(args.variants, args.games, args.opponent, args.turns, args.workers, args.seed, args.close_share, record)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
