"""Run-time diagnostics: counters, timings, anomalies, a per-turn log, and the
reports that say how well the pipeline is working in a live match.

Nothing here knows about the game. The runtime and controller feed it with
count(), time(), anomaly() and record(); it can be left out entirely (pass no
echo, no log directory) and costs almost nothing.

Anomaly severities
  error  the pipeline contradicted the engine or itself (a bug to chase)
  info   worth knowing but normal in play (the opponent changed something we
         could not see, a slow turn, remembered territory out of date)
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Anomaly:
    kind: str
    severity: str  # "error" or "info"
    turn: int
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.kind} at turn {self.turn}: {self.detail}"


def _ratio(good: int, bad: int) -> str:
    total = good + bad
    return f"{good}/{total}" if total else "n/a"


class Diagnostics:
    def __init__(
        self,
        echo: Callable[[str], None] | None = None,
        log_dir: str | Path | None = None,
        max_echoed_anomalies: int = 12,
        slow_turn_ms: float = 50.0,
    ) -> None:
        self.echo = echo
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.max_echoed_anomalies = max_echoed_anomalies
        self.slow_turn_ms = slow_turn_ms
        self.match_number = 0
        self.log_path: Path | None = None
        self._log = None
        self._reset_counters()

    # ---- collecting -----------------------------------------------------

    def _reset_counters(self) -> None:
        self.counts: Counter = Counter()
        self.timings: dict[str, list[float]] = defaultdict(list)
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.anomalies: list[Anomaly] = []
        self.max_drift = 0
        self._echoed = 0
        self.first_turn: int | None = None
        self.last_turn = -1
        self.total_turns: int | None = None
        self._finished = False

    def count(self, name: str, n: int = 1) -> None:
        self.counts[name] += n

    def time(self, name: str, milliseconds: float) -> None:
        self.timings[name].append(milliseconds)

    def sample(self, name: str, value: float) -> None:
        """Record one observation of a quantity (reported as mean and maximum)."""
        self.samples[name].append(float(value))

    def anomaly(self, kind: str, turn: int, detail: str, severity: str = "error") -> None:
        found = Anomaly(kind, severity, turn, detail)
        self.anomalies.append(found)
        self.counts[f"anomaly_{severity}"] += 1
        if self.echo is not None and severity == "error":
            if self._echoed < self.max_echoed_anomalies:
                self.echo(f"[candidate] {found}")
            elif self._echoed == self.max_echoed_anomalies:
                self.echo("[candidate] further anomalies are not echoed; see the report")
            self._echoed += 1

    def errors(self) -> list[Anomaly]:
        return [a for a in self.anomalies if a.severity == "error"]

    def note_turn(self, turn: int, total_turns: int | None) -> None:
        if self.first_turn is None:
            self.first_turn = turn
        self.last_turn = max(self.last_turn, turn)
        if total_turns is not None:
            self.total_turns = total_turns

    def note_drift(self, drift: int) -> None:
        if drift:
            self.counts["drift_turns"] += 1
            self.max_drift = max(self.max_drift, abs(drift))

    # ---- matches and the log file --------------------------------------

    def start_match(self) -> None:
        """Called by the runtime when a new match begins: finish the previous
        one (report and close its log) and start fresh counters."""
        if self.match_number > 0 and self.counts["turns"] > 0 and not self._finished:
            self.finish_match()
        self._reset_counters()
        self.match_number += 1
        self._open_log()

    def _open_log(self) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ignore = self.log_dir / ".gitignore"
        if not ignore.exists():
            ignore.write_text("*\n")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = self.log_dir / f"match-{stamp}-{self.match_number}.jsonl"
        self._log = self.log_path.open("w", encoding="utf-8")

    def record(self, data: dict) -> None:
        """One line of the per-turn log (a no-op without a log directory)."""
        if self._log is not None:
            self._log.write(json.dumps(data, separators=(",", ":"), default=str) + "\n")

    def finish_match(self) -> str:
        """Print (via echo) and return the report; flush and close the log."""
        text = self.report()
        if self.echo is not None and not self._finished:
            self.echo(text)
        self._finished = True
        self.close()
        return text

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    # ---- reporting ------------------------------------------------------

    def timing_stats(self, name: str) -> tuple[float, float, float] | None:
        values = self.timings.get(name)
        if not values:
            return None
        arr = np.asarray(values)
        return float(arr.mean()), float(np.percentile(arr, 95)), float(arr.max())

    def summary_line(self, turn: int) -> str:
        c = self.counts
        total = f"/{self.total_turns}" if self.total_turns else ""
        stats = self.timing_stats("total")
        latency = f"{stats[0]:.2f} ms/turn" if stats else "n/a"
        return (
            f"[candidate] turn {turn}{total}: "
            f"captures {c['move_CAPTURED']}, deaths {c['event_SELF_DIED'] + c['event_KILLED_BY_OPPONENT']}, "
            f"errors {c['anomaly_error']}, prediction misses {c['pred_result_bad']}, {latency}"
        )

    def report(self) -> str:
        c = self.counts
        span = f"turns {self.first_turn}-{self.last_turn}" if self.first_turn is not None else "no turns"
        total = f" of {self.total_turns}" if self.total_turns else ""
        lines = [f"=== Candidate runtime report: match {self.match_number} ({span}{total}) ==="]
        lines.append(f"turns {c['turns']}   retries after INVALID {c['retries']}   forfeited turns {c['forfeits']}")
        moves = "  ".join(f"{name} {c['move_' + name]}" for name in ("NORMAL", "CAPTURED", "DIED"))
        lines.append(f"my moves: {moves}  INVALID {c['invalid_moves']}")
        events = {k[len("event_"):]: v for k, v in sorted(c.items()) if k.startswith("event_")}
        lines.append("events: " + ("  ".join(f"{k} {v}" for k, v in events.items()) or "none"))
        lines.append(
            "prediction vs engine: "
            f"move result {_ratio(c['pred_result_ok'], c['pred_result_bad'])}, "
            f"next position and trail {_ratio(c['state_ok'], c['state_bad'])} "
            f"({c['state_explained_by_kill']} explained by an opponent kill), "
            f"capture size {_ratio(c['gain_ok'], c['gain_off'])}"
        )
        lines.append(
            f"memory: my remembered territory differed from the engine's count on {c['drift_turns']} turns "
            f"(largest gap {self.max_drift} cells)"
        )
        if self.samples.get("belief_hypotheses"):
            hyp, ent = np.asarray(self.samples["belief_hypotheses"]), np.asarray(self.samples.get("belief_entropy", [0.0]))
            lines.append(
                f"belief: hypotheses mean {hyp.mean():.0f} max {hyp.max():.0f}, position entropy mean {ent.mean():.2f} nats, "
                f"rebuilt from scratch {c['belief_reseeds']} times"
            )
        names = [n for n in ("adapter", "belief_move", "belief", "runtime", "policy", "move", "total") if self.timings.get(n)]
        parts = []
        for name in names:
            mean, p95, worst = self.timing_stats(name)
            parts.append(f"{name} {mean:.2f}/{p95:.2f}/{worst:.2f}")
        lines.append("latency ms (mean/p95/max): " + ("   ".join(parts) or "n/a"))
        errors, infos = self.errors(), [a for a in self.anomalies if a.severity == "info"]
        lines.append(f"anomalies: {len(errors)} errors, {len(infos)} info")
        for found in errors[:10]:
            lines.append(f"  {found}")
        if len(errors) > 10:
            lines.append(f"  ... and {len(errors) - 10} more errors")
        by_kind = Counter(a.kind for a in infos)
        if by_kind:
            lines.append("  info: " + ", ".join(f"{k} x{n}" for k, n in sorted(by_kind.items())))
        if self.log_path is not None:
            lines.append(f"per-turn log: {self.log_path}")
        return "\n".join(lines)
