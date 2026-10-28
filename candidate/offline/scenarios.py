"""Training games in many situations: where the players start, how long the game is, and games taken over midway.

The real match is one situation (2000 turns, the default respawns), so a network trained only on it sees a narrow
slice of what can happen, and the online runs showed it: it fights well with the enemy in view but expands badly.
Three ways to vary a training game (the gate and the benchmark keep the real situation, which is what is measured):

  spawns     where the two players start. The agent assumes the enemy's respawn is the left-right mirror of its own
             (runtime.py: MatchRules.mirrored) and reads its seat from the side it starts on (the first mover starts on
             the left), so every spawn here is mirrored and player 0 is on the left:
                 far          the default config: (25, 4) and (25, 45)
                 close        (25, 18) and (25, 31), 13 apart
                 very_close   x in 19..22 on a random row: 5 to 11 apart, the 3 x 3 start lands 2 to 8 cells apart
                 random       x in 3..22 on a random row in 3..46
  turns      turns per player, drawn from a weighted list. Short games leave no time to win by killing alone.
  warm-up    for the first `warmup_turns` turns our seat is played by another agent (the enemy machine, the noisy
             enemy, the framework's basic or random agent, or our own stand-in with full behaviour randomness), then our
             agent takes over in whatever position that left. Our agent watches every turn from the start (its memory and
             belief are those of a player who was there), only its moves are someone else's (Handover, Driven), and Q's
             frames are recorded from the take-over on.

A scenario is drawn per game from the game's own seed (online.py), so a plan is reproducible and the draws of the plan
itself do not change.
"""

import random
from dataclasses import replace

from territorygame.api.agent_controller import AgentController
from territorygame.api.game_api import GameApi
from territorygame.api.move_result import MoveResult as FrameworkResult

SPAWN_CHOICES = ("far", "close", "very_close", "random")
WARMUP_CHOICES = ("enemy", "noisy", "basic", "random", "stand_in")
DEFAULT_SPAWNS = "far=2,close=1,very_close=1,random=2"
DEFAULT_TURNS = "2000=2,1000=1,500=1,300=1"
DEFAULT_WARMUP_AGENTS = "enemy,noisy,basic,random,stand_in"
CLOSE = ((25, 18), (25, 31))


def parse_weights(text: str, cast=str) -> list[tuple]:
    """'a=2,b=1' -> [(a, 2), (b, 1)]; a bare name weighs 1."""
    out = []
    for part in text.split(","):
        name, _, weight = part.strip().partition("=")
        out.append((cast(name.strip()), int(weight or 1)))
    return out


def _pick(rng: random.Random, weighted: list[tuple]):
    names = [n for n, w in weighted for _ in range(w)]
    return names[rng.randrange(len(names))]


def spawn(kind: str, rng: random.Random) -> tuple | None:
    """The respawns ((y, x) of player 0, then player 1) of a spawn kind; None is the default config's."""
    if kind == "far":
        return None
    if kind == "close":
        return CLOSE
    if kind == "very_close":
        y, x = rng.randint(5, 44), rng.randint(19, 22)
    elif kind == "random":
        y, x = rng.randint(3, 46), rng.randint(3, 22)
    else:
        raise ValueError(f"unknown spawn {kind!r}; choose from {SPAWN_CHOICES}")
    return ((y, x), (y, 49 - x))


def with_scenario(spec, spawns: list[tuple], turns: list[tuple], warmup_share: float, warmup_agents: list[str],
                  warmup_fraction: tuple[float, float] = (0.1, 0.5)):
    """The spec in a drawn situation: spawn, turns, and maybe a warm-up by another agent (see the module notes)."""
    rng = random.Random(spec.seed + 5)
    kind = _pick(rng, spawns)
    n_turns = int(_pick(rng, turns))
    warmup, agent = 0, None
    if warmup_share > 0 and warmup_agents and rng.random() < warmup_share:
        warmup = int(n_turns * rng.uniform(*warmup_fraction))
        agent = warmup_agents[rng.randrange(len(warmup_agents))]
    return replace(spec, turns=n_turns, close=kind == "close", respawns=spawn(kind, rng), spawn=kind,
                   warmup_turns=warmup, warmup_agent=agent)


# ---- the take-over --------------------------------------------------------------------------------------------------


class _Capture(GameApi):
    """The real game for reading; a move is only noted (and reported as accepted), not made."""

    def __init__(self, inner: GameApi) -> None:
        self._inner = inner
        self.direction = None

    def get_agent_position(self):
        return self._inner.get_agent_position()

    def get_respawn_position(self):
        return self._inner.get_respawn_position()

    def get_owned_territory_cell_count(self):
        return self._inner.get_owned_territory_cell_count()

    def get_opponent_territory_cell_count(self):
        return self._inner.get_opponent_territory_cell_count()

    def get_remaining_turns(self):
        return self._inner.get_remaining_turns()

    def get_active_trail(self):
        return self._inner.get_active_trail()

    def get_visible_grid(self):
        return self._inner.get_visible_grid()

    def get_board_width(self):
        return self._inner.get_board_width()

    def get_board_height(self):
        return self._inner.get_board_height()

    def move(self, direction):
        if self.direction is None:
            self.direction = direction
        return FrameworkResult.MOVED


class Driven:
    """A Strategy that plays a given move while `forced` is set, and is the wrapped strategy otherwise. The wrapped
    one is not told about forced turns: they are not its decisions (nothing is recorded for them)."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.forced = None

    def __getattr__(self, name):                       # stats, close, last, ... are the wrapped strategy's
        return getattr(self.inner, name)

    def reset(self) -> None:
        self.inner.reset()

    def choose(self, view):
        return self.forced if self.forced is not None else self.inner.choose(view)

    def on_result(self, view, direction, result) -> None:
        if self.forced is None:
            self.inner.on_result(view, direction, result)

    def debug_state(self) -> str:
        return "warm-up" if self.forced is not None else self.inner.debug_state()


class Handover(AgentController):
    """Our agent (a Controller whose strategy is a Driven) in a seat another agent plays for the first `turns` turns.
    Each of those turns the other agent decides on the real board (its move is captured, not made), and our agent makes
    that move itself, so its runtime sees and records everything as usual."""

    def __init__(self, ours, driven: Driven, other: AgentController, turns: int) -> None:
        self.ours, self.driven, self.other, self.turns = ours, driven, other, turns
        self.played = 0
        self._left = None

    def take_turn(self, game: GameApi) -> None:
        left = game.get_remaining_turns()
        if left != self._left:                          # a new turn, not the engine asking again after a refusal
            self._left = left
            self.played += 1
        self.driven.forced = None
        if self.played <= self.turns:
            from candidate.agent.adapter import direction_from_framework

            capture = _Capture(game)
            try:
                self.other.take_turn(capture)
            except Exception:  # noqa: BLE001 - a failing warm-up agent just hands over this turn
                capture.direction = None
            if capture.direction is not None:
                self.driven.forced = direction_from_framework(capture.direction)
        self.ours.take_turn(game)

    def get_debug_state(self):
        return self.ours.get_debug_state()


def warmup_agent(name: str, seed: int) -> AgentController:
    """The agent that plays the warm-up: the enemy machine (our exact copy), the noisy enemy, the framework's basic or
    random agent, or our own stand-in with full behaviour randomness and 20% exploration."""
    if name == "enemy":
        from candidate.offline.noisy_enemy import NoisyEnemy

        return NoisyEnemy()
    if name == "noisy":
        from candidate.offline.noisy_enemy import NoisyEnemy

        return NoisyEnemy(randomness=0.8, seed=seed + 11)
    if name in ("basic", "random"):
        from candidate.runtime_check import OPPONENTS

        return OPPONENTS[name]()
    if name == "stand_in":
        from candidate.agent.behavior.config import BehaviorConfig
        from candidate.agent.controller import Controller
        from candidate.agent.diagnostics import Diagnostics
        from candidate.agent.q.heuristic import HeuristicQ
        from candidate.agent.q.strategy import QStrategy
        from candidate.offline import match

        strategy = QStrategy(HeuristicQ(), behavior=BehaviorConfig(randomness=1.0), explore_rate=0.2, seed=seed + 13,
                             build_inputs=False)
        return match.AgentAdapter(Controller(strategy, Diagnostics(echo=None)))
    raise ValueError(f"unknown warm-up agent {name!r}; choose from {WARMUP_CHOICES}")
