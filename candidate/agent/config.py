"""Match constants shared by every module under candidate/agent.

These mirror territorygame/resources/game-config.properties (50 x 50 board,
11 x 11 window, 3 x 3 starting territory). The framework is read-only for us,
so nothing here is read from it; the adapter must check the values it is
handed against these and fail loudly on a mismatch.

Bitboards store one board row per int64 word, so a row must fit below the
sign bit: BOARD_WIDTH <= 62. The assert below keeps that true if the board
size is ever changed.
"""

BOARD_HEIGHT = 50
BOARD_WIDTH = 50
CELL_COUNT = BOARD_HEIGHT * BOARD_WIDTH

WINDOW_SIZE = 11
STARTING_TERRITORY_SIZE = 3          # side length of the square start around each respawn
STARTING_TERRITORY_COUNT = 9

assert BOARD_WIDTH <= 62, "a bitboard row must fit in a non-negative int64 word"
