"""Headless check of the run-time layer against the real engine.

Plays the candidate controller (agent.controller.Controller with either the
loop-making probe from agent.probe or the real QStrategy, wired through
agent.runtime) in a full match with no GUI, then compares everything the
runtime believed with the engine's true state:

  * adapter      every observation's position, trail, counts, opponent
                 sighting and window contents against the true board;
  * memory       how closely the remembered map of my own territory matches
                 the true one, turn by turn;
  * events       deaths, opponent kills and captures the runtime inferred
                 against the engine's own death and kill counters;
  * belief       the probability it put on the opponent's true cell, and the
                 mass on its exactly right hidden state (cell, trail, count);
  * prediction   the runtime's per-move predictions (in the diagnostics);
  * latency      time per turn and per stage.

Run it from the repository root:

    python -m candidate.runtime_check --opponent hunter --turns 600 --close --suicide-every 4
    python -m candidate.runtime_check --strategy q --opponent enemy --turns 400

Options: --strategy {probe,q}  what plays: the probe (default) or QStrategy
         --opponent {enemy,basic,random,hunter}   who to play against
         --turns N        turns per player (default 400)
         --player {0,1}   which side the candidate plays (default 0)
         --close          put the respawns 13 cells apart so the players meet
         --suicide-every N  probe only: every Nth excursion, step on my own trail
         --no-hunt        probe only: do not step onto the opponent's trail
         --seed N         probe only: its random seed
         --log-dir DIR    write the per-turn log there
Exit status is 0 when the run is clean, 1 otherwise.
"""

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np

from candidate.agent.config import BOARD_WIDTH, WINDOW_SIZE
from candidate.agent.controller import Controller
from candidate.agent.data.cells import FREE, MY_TERRITORY, MY_TRAIL, OPP_TERRITORY, OPP_TRAIL
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.geometry import bitboard
from candidate.agent.probe import ProbePolicy
from candidate.agent.runtime import TurnView
from candidate.offline import match
from candidate.examples.basic_state_machine import BasicStateMachine
from candidate.examples.random_state_machine import RandomStateMachine
from territorygame.api.agent_controller import AgentController
from territorygame.api.direction import Direction as FrameworkDirection
from territorygame.api.grid_position import GridPosition
from territorygame.api.occupant_view import OccupantView
from territorygame.controller.enemy_state_machine import EnemyStateMachine
from territorygame.helpers.movement_utils import find_cell, is_valid_move, manhattan_distance, next_position


# ---- opponents ----------------------------------------------------------


class Hunter(AgentController):
    """Heads for the nearest visible trail of the other player (stepping onto
    it kills them), otherwise for where it supposes the other player lives.
    Avoids its own trail. Exists so that opponent kills happen."""

    def take_turn(self, game) -> None:
        position = game.get_agent_position()
        target, best = None, None
        for row in game.get_visible_grid():
            for cell in row:
                if cell.occupant == OccupantView.OPPONENT_TRAIL:
                    distance = manhattan_distance(position, cell.position)
                    if best is None or distance < best:
                        target, best = cell.position, distance
        if target is None:
            home = game.get_respawn_position()
            target = GridPosition(game.get_board_width() - 1 - home.x, home.y)

        grid = game.get_visible_grid()
        options = []
        for direction in FrameworkDirection:
            if not is_valid_move(game, direction):
                continue
            dest = next_position(position, direction)
            cell = find_cell(grid, dest)
            if cell is not None and cell.occupant == OccupantView.SELF_TRAIL:
                continue
            options.append((manhattan_distance(dest, target), direction.value, direction))
        if options:
            game.move(min(options)[2])
        else:
            game.move(FrameworkDirection.NORTH)


OPPONENTS = {
    "enemy": EnemyStateMachine,
    "basic": BasicStateMachine,
    "random": RandomStateMachine,
    "hunter": Hunter,
}


# ---- recording what the runtime saw -------------------------------------


class _Spy:
    """Wraps a strategy and keeps what the runtime believed at each turn."""

    def __init__(self, inner, collector=None, player: int = 0) -> None:
        self.inner = inner
        self.collector, self.player = collector, player
        # (observation, remembered own-territory cells, events, remembered trail code under each hidden cell)
        self.records: list[tuple] = []
        self.belief_rows: list[dict] = []

    def reset(self) -> None:
        self.records.clear()
        self.belief_rows.clear()
        self.inner.reset()

    def choose(self, view: TurnView):
        if not view.retry:
            mine = frozenset(np.flatnonzero(view.memory.territory == MY_TERRITORY).tolist())
            hidden = {c: int(view.memory.trail.flat[c]) for c in view.obs.trail_hidden}
            self.records.append((view.obs, mine, view.events, hidden))
            if self.collector is not None and view.belief is not None:
                self.belief_rows.append(_score_belief(view.belief, self.collector.latest, self.player))
        return self.inner.choose(view)

    def on_result(self, view, direction, result) -> None:
        self.inner.on_result(view, direction, result)

    def debug_state(self) -> str:
        return self.inner.debug_state()


def _score_belief(belief, snapshot, player: int) -> dict:
    """How well the belief describes the true opponent in `snapshot`."""
    opp = snapshot.players[1 - player]
    truth_cell = opp.position.y * BOARD_WIDTH + opp.position.x
    p_true = belief.probability_at(truth_cell)
    exact = 0.0
    if belief.n_hypotheses:
        true_trail = bitboard.from_cells(q.y * BOARD_WIDTH + q.x for q in opp.trail)
        match = (
            (belief._pos == truth_cell)
            & (belief._opp_count == opp.territory_count)
            & (belief._trail_bb == true_trail).all(axis=1)
        )
        exact = float(belief._weight[match].sum())
    return {
        "p_true": p_true,
        "in_support": bool(belief.support.reshape(-1)[truth_cell]),
        "exact_mass": exact,
        "top1": belief.most_likely()[0] == truth_cell,
        "seen": belief.sighted_at is not None,
        "n": belief.n_hypotheses,
        "reseeded": any(e.name == "RESEEDED" for e in belief.events),
    }


# ---- the check ----------------------------------------------------------


@dataclass
class CheckResult:
    diagnostics: Diagnostics
    metrics: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    strategy: object = None            # the strategy that played (for its own statistics)

    @property
    def ok(self) -> bool:
        return not self.problems

    def format(self) -> str:
        m = self.metrics
        lines = [self.diagnostics.report(), "", "--- compared with the engine's true state ---"]
        lines.append(
            f"observations checked {m['observations']}: position/trail/count mismatches {m['adapter_mismatches']}, "
            f"visible cells translated wrongly {m['window_wrong']}"
        )
        lines.append(
            f"cells under an agent (trail hidden by the framework): {m['hidden_total']} "
            f"({m['opp_seen']} opponent sightings); the observation's guess was wrong {m['guess_wrong']} times, "
            f"memory's version wrong {m['memory_hidden_wrong']} times"
        )
        lines.append(
            f"events inferred vs engine counters: deaths {m['deaths_inferred']} vs {m['deaths_true']}, "
            f"opponent kills {m['kills_inferred']} vs {m['kills_true']}"
        )
        lines.append(
            f"memory of my territory: exact on {m['memory_exact_turns']}/{m['observations']} turns; "
            f"missing cells mean {m['memory_missing_mean']:.2f} max {m['memory_missing_max']}, "
            f"extra cells mean {m['memory_extra_mean']:.2f} max {m['memory_extra_max']}"
        )
        if "belief_turns" in m:
            lines.append(
                f"belief vs the true opponent: probability on its true cell mean {m['belief_p_true_mean']:.3f} "
                f"(min {m['belief_p_true_min']:.4f}, out of view mean {m['belief_p_true_unseen_mean']:.3f}); "
                f"true cell had non-zero probability on {m['belief_in_support']}/{m['belief_turns']} turns; "
                f"most likely cell was right on {m['belief_top1']}/{m['belief_turns']}"
            )
            lines.append(
                f"  mass on the exactly right hidden state (cell, trail, count) mean {m['belief_exact_mass_mean']:.3f}; "
                f"hypotheses mean {m['belief_hypotheses_mean']:.0f} max {m['belief_hypotheses_max']}; "
                f"rebuilt from scratch {m['belief_reseeds']} times"
            )
        lines.append(
            f"final territory: candidate {m['final_mine']}, opponent {m['final_theirs']}; "
            f"captures {m['captures']}, my deaths {m['deaths_true']}, kills of the opponent {m['kills_true']}"
        )
        stats = getattr(self.strategy, "stats", None)
        if stats:
            lines.append("strategy: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))
            timing = getattr(self.strategy, "timing_ms", {})
            if any(timing.values()):
                lines.append("  per frame (ms, mean): " + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in timing.items() if v))
        lines.append("RESULT: " + ("clean" if self.ok else "PROBLEMS"))
        lines += [f"  - {p}" for p in self.problems]
        return "\n".join(lines)


def run_check(
    opponent: str = "enemy",
    turns: int = 400,
    player: int = 0,
    close: bool = False,
    suicide_every: int | None = None,
    hunt: bool = True,
    seed: int = 1,
    echo=None,
    log_dir=None,
    strategy: str = "probe",
) -> CheckResult:
    diagnostics = Diagnostics(echo=echo, log_dir=log_dir)
    collector = match.Watch(keep=True)          # its `latest` is the board my next turn faces (see _Spy)
    if strategy == "q":
        from candidate.agent.q.strategy import QStrategy

        played = QStrategy()
    elif strategy == "probe":
        played = ProbePolicy(seed=seed, hunt=hunt, suicide_every=suicide_every)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    spy = _Spy(played, collector, player)
    candidate = match.AgentAdapter(Controller(spy, diagnostics))
    controllers = [OPPONENTS[opponent](), OPPONENTS[opponent]()]
    controllers[player] = candidate

    snapshots = match.play(controllers, turns, match.CLOSE if close else match.FAR, watch=collector).snapshots

    result = CheckResult(diagnostics, strategy=played)
    _score(result, spy, snapshots, player, turns)
    diagnostics.close()
    return result


def _score(result: CheckResult, spy: _Spy, snapshots: list, player: int, turns: int) -> None:
    diag, m, problems = result.diagnostics, result.metrics, result.problems
    if any(s.error_message for s in snapshots):
        problems.append("the engine reported a forfeited turn or error: " + next(s.error_message for s in snapshots if s.error_message))
    if len(spy.records) != turns:
        problems.append(f"the controller saw {len(spy.records)} turns, expected {turns}")
        return

    mismatches = window_wrong = opp_seen = hidden_total = guess_wrong = memory_wrong = 0
    missing, extra, exact = [], [], 0
    half = WINDOW_SIZE // 2
    for obs, remembered, _, hidden_memory in spy.records:
        snap = snapshots[2 * obs.turn + player]
        me, opp = snap.players[player], snap.players[1 - player]
        flat = lambda p: p.y * BOARD_WIDTH + p.x  # noqa: E731
        if (
            obs.turns_left != me.remaining_turns
            or obs.my_pos != flat(me.position)
            or [int(c) for c in obs.my_trail] != [flat(p) for p in me.trail]
            or obs.my_count != me.territory_count
            or obs.opp_count != opp.territory_count
        ):
            mismatches += 1

        oy, ox = obs.window_origin
        in_view = abs(opp.position.y - me.position.y) <= half and abs(opp.position.x - me.position.x) <= half
        if (obs.opp_seen_at is not None) != in_view or (in_view and obs.opp_seen_at != flat(opp.position)):
            mismatches += 1
        opp_seen += in_view
        for dy in range(obs.window_territory.shape[0]):
            for dx in range(obs.window_territory.shape[1]):
                board_cell = snap.cells[oy + dy][ox + dx]
                want_terr = MY_TERRITORY if board_cell.territory_owner == me.id else (
                    OPP_TERRITORY if board_cell.territory_owner == opp.id else FREE)
                want_trail = MY_TRAIL if board_cell.trail_owner == me.id else (
                    OPP_TRAIL if board_cell.trail_owner == opp.id else FREE)
                cell_index = (oy + dy) * BOARD_WIDTH + ox + dx
                if obs.window_territory[dy, dx] != want_terr:
                    window_wrong += 1
                if cell_index in hidden_memory:
                    # An agent stands here, so the trail layer is a guess; score it and memory's version.
                    hidden_total += 1
                    guess_wrong += obs.window_trail[dy, dx] != want_trail
                    memory_wrong += hidden_memory[cell_index] != want_trail
                elif obs.window_trail[dy, dx] != want_trail:
                    window_wrong += 1

        truth = frozenset(
            y * BOARD_WIDTH + x
            for y, row in enumerate(snap.cells)
            for x, cell in enumerate(row)
            if cell.territory_owner == me.id
        )
        missing.append(len(truth - remembered))
        extra.append(len(remembered - truth))
        exact += remembered == truth

    last_obs = spy.records[-1][0]
    last = snapshots[2 * last_obs.turn + player]
    events = diag.counts
    deaths_inferred = events["event_SELF_DIED"] + events["event_KILLED_BY_OPPONENT"]
    m.update(
        observations=len(spy.records),
        adapter_mismatches=mismatches,
        window_wrong=window_wrong,
        opp_seen=opp_seen,
        hidden_total=hidden_total,
        guess_wrong=guess_wrong,
        memory_hidden_wrong=memory_wrong,
        deaths_inferred=deaths_inferred,
        deaths_true=last.players[player].death_count,
        kills_inferred=events["event_KILLED_OPPONENT"],
        kills_true=last.players[player].kill_count,
        memory_exact_turns=exact,
        memory_missing_mean=float(np.mean(missing)),
        memory_missing_max=int(max(missing)),
        memory_extra_mean=float(np.mean(extra)),
        memory_extra_max=int(max(extra)),
        final_mine=snapshots[-1].players[player].territory_count,
        final_theirs=snapshots[-1].players[1 - player].territory_count,
        captures=events["move_CAPTURED"],
    )

    rows = spy.belief_rows
    if rows:
        p_true = np.array([r["p_true"] for r in rows])
        seen = np.array([r["seen"] for r in rows])
        m.update(
            belief_turns=len(rows),
            belief_p_true_mean=float(p_true.mean()),
            belief_p_true_min=float(p_true.min()),
            belief_p_true_unseen_mean=float(p_true[~seen].mean()) if (~seen).any() else float("nan"),
            belief_p_true_seen_min=float(p_true[seen].min()) if seen.any() else float("nan"),
            belief_in_support=int(sum(r["in_support"] for r in rows)),
            belief_top1=int(sum(r["top1"] for r in rows)),
            belief_exact_mass_mean=float(np.mean([r["exact_mass"] for r in rows])),
            belief_hypotheses_mean=float(np.mean([r["n"] for r in rows])),
            belief_hypotheses_max=int(max(r["n"] for r in rows)),
            belief_reseeds=int(sum(r["reseeded"] for r in rows)),
            seen_turns=int(seen.sum()),
        )
        if m["belief_in_support"] != len(rows):
            problems.append(f"the true opponent position had zero probability on {len(rows) - m['belief_in_support']} turns")

    if diag.errors():
        problems.append(f"{len(diag.errors())} runtime errors (see the report)")
    if mismatches:
        problems.append(f"{mismatches} observations disagree with the true state")
    if window_wrong:
        problems.append(f"{window_wrong} visible cells were translated wrongly")
    if m["deaths_inferred"] != m["deaths_true"]:
        problems.append(f"inferred {m['deaths_inferred']} deaths, the engine counted {m['deaths_true']}")
    if m["kills_inferred"] != m["kills_true"]:
        problems.append(f"inferred {m['kills_inferred']} opponent kills, the engine counted {m['kills_true']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--opponent", choices=sorted(OPPONENTS), default="enemy")
    parser.add_argument("--turns", type=int, default=400)
    parser.add_argument("--player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--close", action="store_true")
    parser.add_argument("--suicide-every", type=int, default=None)
    parser.add_argument("--no-hunt", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--strategy", choices=("probe", "q"), default="probe")
    args = parser.parse_args(argv)
    result = run_check(
        args.opponent, args.turns, args.player, args.close, args.suicide_every,
        hunt=not args.no_hunt, seed=args.seed, echo=None, log_dir=args.log_dir,
        strategy=args.strategy,
    )
    print(result.format())
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
