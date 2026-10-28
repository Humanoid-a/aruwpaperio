"""NoisyEnemy: a copy of the framework's EnemyStateMachine (territorygame/controller/enemy_state_machine.py) with a
seeded, per-game variation, for training games. The framework file is not modified; this is a separate copy.

The original is deterministic (its random choices come from Random(42)), so every game against it is the same game.
NoisyEnemy(randomness=r, seed=s) draws its own version of the original's habits for each game:

    bite        the side of the box its trail must stay inside (the original's _SAFETY_GRID_SIZE, 7): how far it
                extends before turning. At r = 1, 5, 7, 9 or 11.
    lookahead   how far ahead (the original's _OUT_LOOKAHEAD, 2) it looks for the edge of its land when repositioning.
                At r = 1, 1 to 3.
    vertical    the chance it follows the original's preference for going out north or south (1 in the original)
    aim         the chance it starts an expedition toward where it last saw me, instead of a random way out
    strike      the chance per turn, out on an expedition, that it steps onto a cell of my trail right next to it (a
                kill; the original only attacks my trail on its own land)
    wander      the chance per turn, while repositioning on its land, that it takes a random step on its land instead

At randomness 0 (and no seed) it is the original, move for move: same constants, same Random(42), and none of the
additions draws a random number. Everything it adds keeps the original's safety: its trail still stays inside the box,
and a strike is a step onto my trail, which kills me and cannot hurt it.

One change that is only about speed: the original asks the engine for the whole visible grid (which the engine
recomputes every time) and searches it, for every cell it looks at, about ten times a turn. Here each turn sees the
game through _ThisTurn, which fetches the grid once and finds a cell by its position directly.
"""

import random
from enum import Enum, auto

from territorygame.api.agent_controller import AgentController
from territorygame.api.direction import Direction
from territorygame.api.game_api import GameApi
from territorygame.api.grid_position import GridPosition
from territorygame.api.occupant_view import OccupantView
from territorygame.api.territory_view import TerritoryView
from territorygame.helpers.movement_utils import (
    is_valid_board_move,
    is_within_board,
    manhattan_distance,
    next_position,
    valid_directions,
)

_SAFETY_GRID_SIZE = 7
_OUT_LOOKAHEAD = 2


class _Phase(Enum):
    ATTACK = auto()
    REPOSITION = auto()
    OUT = auto()
    ACROSS = auto()
    BACK = auto()


class _ThisTurn:
    """The game during one of NoisyEnemy's turns: the visible grid and the agent's position fetched once (neither can
    change before the move, which ends the turn), cells found by position. Everything else goes to the game."""

    def __init__(self, game: GameApi) -> None:
        self._game = game
        self._grid = game.get_visible_grid()
        self._position = game.get_agent_position()
        self._cells = {(cell.position.x, cell.position.y): cell for row in self._grid for cell in row}

    def __getattr__(self, name):
        return getattr(self._game, name)

    def get_visible_grid(self):
        return self._grid

    def get_agent_position(self) -> GridPosition:
        return self._position

    def cell(self, position: GridPosition):
        """The visible cell at `position`, or None outside the window (as find_cell)."""
        return self._cells.get((position.x, position.y))


class NoisyEnemy(AgentController):
    def __init__(self, randomness: float = 0.0, seed: int | None = None) -> None:
        self._phase = _Phase.OUT
        self._out_direction: Direction | None = None
        self._steps_out = 0
        self._across_direction: Direction | None = None
        self._reposition_direction: Direction | None = None
        self._random = random.Random(42 if seed is None else seed)        # the original's choices
        self._noise = random.Random(f"noisy-turns:{seed}")                 # the additions' per-turn draws
        self._last_seen: GridPosition | None = None                        # where I (its opponent) was last seen

        r = max(0.0, float(randomness))
        draw = random.Random(f"noisy-habits:{seed}")
        self.randomness = r
        self.bite = _SAFETY_GRID_SIZE + 2 * draw.randint(round(-r), round(2 * r)) if r else _SAFETY_GRID_SIZE
        self.lookahead = max(1, _OUT_LOOKAHEAD + draw.randint(round(-r), round(r))) if r else _OUT_LOOKAHEAD
        self.vertical = 1.0 - draw.uniform(0.0, r) if r else 1.0
        self.aim = draw.uniform(0.0, r) if r else 0.0
        self.strike = draw.uniform(0.0, r) if r else 0.0
        self.wander = draw.uniform(0.0, 0.2 * r) if r else 0.0

    def habits(self) -> dict:
        """This game's draw, for the game's record."""
        return dict(randomness=self.randomness, bite=self.bite, lookahead=self.lookahead, vertical=round(self.vertical, 3),
                    aim=round(self.aim, 3), strike=round(self.strike, 3), wander=round(self.wander, 3))

    def _chance(self, p: float) -> bool:
        """True with probability p; draws nothing when p is 0, so the original's behaviour is untouched."""
        return p > 0.0 and self._noise.random() < p

    def take_turn(self, game: GameApi) -> None:
        game = _ThisTurn(game)
        if self.aim > 0.0:
            self._remember_opponent(game)
        intrusion = self._find_opponent_trail_on_our_territory(game)
        if intrusion is not None:
            self._phase = _Phase.ATTACK
            direction = self._pick_attack(game, intrusion)
        elif not game.get_active_trail():
            direction = self._pick_trail_free(game)
        else:
            direction = self._pick_strike(game) if self.strike > 0.0 else None
            if direction is None:
                direction = self._pick_expedition(game)
        game.move(direction)

    def get_debug_state(self) -> str | None:
        return self._phase.name

    # ---- the additions ----------------------------------------------------------------------------------------

    def _remember_opponent(self, game: GameApi) -> None:
        for row in game.get_visible_grid():
            for cell in row:
                if cell.occupant in (OccupantView.OPPONENT_AGENT, OccupantView.OPPONENT_TRAIL):
                    self._last_seen = cell.position
                    return

    def _pick_strike(self, game: GameApi) -> Direction | None:
        """Out on an expedition: now and then, step onto a cell of the opponent's trail right next to the head."""
        targets = [d for d in self._huntable_directions(game)
                   if self._occupant_at(game, self._destination(game, d)) == OccupantView.OPPONENT_TRAIL]
        if targets and self._chance(self.strike):
            return targets[0]
        return None

    def _out_choice(self, game: GameApi, outside_territory: list[Direction]) -> Direction:
        """The way out of a new expedition: toward the opponent now and then, else the original's choice."""
        if self._last_seen is not None and self._chance(self.aim):
            return min(outside_territory, key=self._distance_to_key(game, self._last_seen))
        candidates = self._prefer_vertical(outside_territory) if self.vertical >= 1.0 or self._chance(self.vertical) \
            else outside_territory
        return self._choose_random(game, candidates)

    # ---- ATTACK / REPOSITION ---------------------------------------------

    def _pick_attack(self, game: GameApi, target: GridPosition) -> Direction:
        """A free kill: crossing their trail sends them back to respawn, no risk to us."""
        return self._choose_best(game, self._huntable_directions(game), self._distance_to_key(game, target))

    def _pick_trail_free(self, game: GameApi) -> Direction:
        """Chooses a move on owned territory or starts OUT when the chosen move leaves it."""
        safe = self._safe_directions(game)
        territory_only = [d for d in safe if self._is_self_territory(game, self._destination(game, d))]

        if (
            self._reposition_direction is not None
            and self._reposition_direction in territory_only
            and self._distance_to_outside_territory(game, self._reposition_direction) <= self.lookahead
        ):
            self._phase = _Phase.REPOSITION
            if territory_only and self._chance(self.wander):
                self._reposition_direction = self._choose_random(game, territory_only)
            return self._reposition_direction

        outside_territory = [d for d in safe if d not in territory_only]
        if outside_territory:
            self._phase = _Phase.OUT
            self._out_direction = (
                self._reposition_direction
                if self._reposition_direction in outside_territory
                else self._out_choice(game, outside_territory)
            )
            self._reposition_direction = None
            self._steps_out = 1
            return self._out_direction

        self._phase = _Phase.REPOSITION
        approaches = self._closest_approach_directions(game, territory_only)
        if approaches:
            self._reposition_direction = self._choose_random(game, approaches)
            return self._reposition_direction
        if self._reposition_direction is not None and self._reposition_direction in territory_only:
            return self._reposition_direction
        self._reposition_direction = self._choose_random(game, safe if not territory_only else territory_only)
        return self._reposition_direction

    # ---- Expedition: OUT / ACROSS / BACK --------------------------------

    def _pick_expedition(self, game: GameApi) -> Direction:
        """Continues whichever part of an active expedition was last selected."""
        if self._phase == _Phase.OUT:
            return self._pick_out(game)
        if self._phase == _Phase.ACROSS:
            return self._pick_across(game)
        return self._pick_back(game)

    def _pick_out(self, game: GameApi) -> Direction:
        next_head = self._destination(game, self._out_direction)
        if (
            self._out_direction in self._safe_directions(game)
            and not self._is_self_territory(game, next_head)
            and self.fits_safety_grid(next_head, game.get_active_trail(), self.bite)
        ):
            self._steps_out += 1
            return self._out_direction
        self._phase = _Phase.ACROSS
        self._across_direction = self._pick_across_direction(game)
        return self._pick_across(game)

    def _pick_across_direction(self, game: GameApi) -> Direction:
        perpendicular = [
            d for d in self.perpendicular_options(self._out_direction)
            if self._can_take_another_across_step(game, d)
        ]
        if not perpendicular:
            return self._out_direction
        return self._choose_random(game, perpendicular)

    def _pick_across(self, game: GameApi) -> Direction:
        if self._can_take_another_across_step(game, self._across_direction):
            return self._across_direction
        self._phase = _Phase.BACK
        return self._pick_back(game)

    def _can_take_another_across_step(self, game: GameApi, direction: Direction) -> bool:
        if direction not in self._safe_directions(game):
            return False
        next_head = self._destination(game, direction)
        if not self.fits_safety_grid(next_head, game.get_active_trail(), self.bite):
            return False
        mirrored = self.mirror_back(next_head, self._out_direction, self._steps_out)
        return self._is_self_territory(game, mirrored)

    # ---- Shared BACK logic ------------------------------------------------

    def _pick_back(self, game: GameApi) -> Direction:
        """Walks opposite out_direction. ACROSS already required that this path lands on owned land."""
        safe = self._safe_directions(game)
        if self._out_direction is None:
            return self._choose_random(game, safe)
        reverse = self.opposite(self._out_direction)
        return reverse if reverse in safe else self._choose_random(game, safe)

    # ---- Board reading -----------------------------------------------------

    def _safe_directions(self, game: GameApi) -> list[Direction]:
        result = []
        for direction in Direction:
            if not is_valid_board_move(
                game.get_agent_position(), direction, game.get_board_width(), game.get_board_height()
            ):
                continue
            occupant = self._occupant_at(game, self._destination(game, direction))
            if occupant not in (OccupantView.SELF_TRAIL, OccupantView.OPPONENT_TRAIL, OccupantView.OPPONENT_AGENT):
                result.append(direction)
        return result

    def _huntable_directions(self, game: GameApi) -> list[Direction]:
        """Like _safe_directions, but allows stepping onto the opponent's trail: that's how a chase ends in a kill."""
        result = []
        for direction in Direction:
            if not is_valid_board_move(
                game.get_agent_position(), direction, game.get_board_width(), game.get_board_height()
            ):
                continue
            occupant = self._occupant_at(game, self._destination(game, direction))
            if occupant not in (OccupantView.SELF_TRAIL, OccupantView.OPPONENT_AGENT):
                result.append(direction)
        return result

    @staticmethod
    def _occupant_at(game: GameApi, position: GridPosition) -> OccupantView:
        cell = game.cell(position)
        return OccupantView.EMPTY if cell is None else cell.occupant

    @staticmethod
    def _destination(game: GameApi, direction: Direction) -> GridPosition:
        return next_position(game.get_agent_position(), direction)

    def _closest_approach_directions(self, game: GameApi, candidates: list[Direction]) -> list[Direction]:
        closest: list[Direction] = []
        closest_distance = self.lookahead + 1
        for direction in candidates:
            distance = self._distance_to_outside_territory(game, direction)
            if distance < closest_distance:
                closest = []
                closest_distance = distance
            if distance == closest_distance and distance <= self.lookahead:
                closest.append(direction)
        return self._prefer_vertical(closest)

    def _distance_to_outside_territory(self, game: GameApi, direction: Direction) -> int:
        position = game.get_agent_position()
        for distance in range(1, self.lookahead + 1):
            position = next_position(position, direction)
            if not is_within_board(position, game.get_board_width(), game.get_board_height()):
                break
            cell = game.cell(position)
            if cell is None:
                break
            if cell.territory != TerritoryView.SELF:
                return distance
        return self.lookahead + 1

    @staticmethod
    def _prefer_vertical(candidates: list[Direction]) -> list[Direction]:
        vertical = [d for d in candidates if NoisyEnemy.is_vertical(d)]
        return vertical if vertical else candidates

    @staticmethod
    def _is_self_territory(game: GameApi, position: GridPosition) -> bool:
        """False for cells outside the visible window."""
        cell = game.cell(position)
        return cell is not None and cell.territory == TerritoryView.SELF

    def _choose_best(self, game: GameApi, candidates: list[Direction], ranking) -> Direction:
        if not candidates:
            return self._fallback(game)
        return min(candidates, key=ranking)

    def _choose_random(self, game: GameApi, candidates: list[Direction]) -> Direction:
        if not candidates:
            return self._fallback(game)
        return candidates[self._random.randrange(len(candidates))]

    @staticmethod
    def _distance_to_key(game: GameApi, target: GridPosition):
        return lambda direction: manhattan_distance(NoisyEnemy._destination(game, direction), target)

    @staticmethod
    def _find_opponent_trail_on_our_territory(game: GameApi) -> GridPosition | None:
        """Nearest visible cell where the opponent's trail is crossing land that's ours: an intrusion worth punishing."""
        from_position = game.get_agent_position()
        best: GridPosition | None = None
        best_distance: int | None = None
        for row in game.get_visible_grid():
            for cell in row:
                if cell.territory == TerritoryView.SELF and cell.occupant == OccupantView.OPPONENT_TRAIL:
                    distance = manhattan_distance(from_position, cell.position)
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best = cell.position
        return best

    @staticmethod
    def _fallback(game: GameApi) -> Direction:
        valid = valid_directions(game)
        return valid[0] if valid else Direction.NORTH

    # ---- Pure helpers (no GameApi) ---------------------------------------------

    @staticmethod
    def chebyshev_distance(a: GridPosition, b: GridPosition) -> int:
        return max(abs(a.x - b.x), abs(a.y - b.y))

    @staticmethod
    def fits_safety_grid(head: GridPosition, trail: list[GridPosition], grid_size: int) -> bool:
        """Every cell in trail must stay within grid_size // 2 of head. Vacuously true for an empty trail."""
        half = grid_size // 2
        return all(NoisyEnemy.chebyshev_distance(cell, head) <= half for cell in trail)

    @staticmethod
    def opposite(direction: Direction) -> Direction:
        return {
            Direction.NORTH: Direction.SOUTH,
            Direction.SOUTH: Direction.NORTH,
            Direction.EAST: Direction.WEST,
            Direction.WEST: Direction.EAST,
        }[direction]

    @staticmethod
    def perpendicular_options(direction: Direction) -> list[Direction]:
        if direction in (Direction.NORTH, Direction.SOUTH):
            return [Direction.EAST, Direction.WEST]
        return [Direction.NORTH, Direction.SOUTH]

    @staticmethod
    def mirror_back(position: GridPosition, out_direction: Direction, steps: int) -> GridPosition:
        """Walks steps cells in the reverse of out_direction from position."""
        result = position
        reverse = NoisyEnemy.opposite(out_direction)
        for _ in range(steps):
            result = next_position(result, reverse)
        return result

    @staticmethod
    def is_vertical(direction: Direction) -> bool:
        return direction in (Direction.NORTH, Direction.SOUTH)
