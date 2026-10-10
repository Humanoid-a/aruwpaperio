"""A scripted opponent that really takes my land, which none of the stock opponents does.

It leaves its own territory, walks a rectangle round my start land and comes back onto its
territory, closing a loop that encloses (and so captures) whatever of mine lies inside. With
the close respawns of runtime_check (me at (25, 18), it at (25, 31)) that is my start land.
Used to check that the belief copes with an opponent that captures from me.
"""

from territorygame.api.agent_controller import AgentController
from territorygame.api.direction import Direction
from territorygame.helpers.movement_utils import is_valid_move

# waypoints as (y, x): out of its land (rows 24-26, cols 30-32), round mine (rows 24-26, cols 17-19), and home
LAP = [(24, 30), (22, 30), (22, 14), (28, 14), (28, 30), (26, 30)]


class Raider(AgentController):
    def __init__(self) -> None:
        self._next = 0

    def take_turn(self, game) -> None:
        pos = game.get_agent_position()
        for _ in range(len(LAP)):
            if (pos.y, pos.x) == LAP[self._next % len(LAP)]:
                self._next += 1
            else:
                break
        ty, tx = LAP[self._next % len(LAP)]
        wanted = []
        if pos.x != tx:
            wanted.append(Direction.EAST if tx > pos.x else Direction.WEST)
        if pos.y != ty:
            wanted.append(Direction.SOUTH if ty > pos.y else Direction.NORTH)
        for direction in wanted + list(Direction):
            if is_valid_move(game, direction):
                game.move(direction)
                return
        game.move(Direction.NORTH)
