"""What deaths, captures and hidden cells do to the MemoryMap."""

import numpy as np

from candidate.agent.data.cells import FREE, MY_TERRITORY, MY_TRAIL, OPP_TERRITORY, OPP_TRAIL, UNKNOWN
from candidate.agent.data.memory import NEVER_SEEN, MemoryMap
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.grid import flat
from candidate.agent.rules.match import MatchRules, starting_cells
from candidate.tests.builders import make_observation, square

RULES = MatchRules.from_respawns(flat(25, 4), flat(25, 45))
MY_START = RULES.me.start_cells
OPP_START = RULES.opp.start_cells


def populated() -> MemoryMap:
    """A memory that has seen: my start plus extra land, my trail, enemy land and an enemy trail."""
    memory = MemoryMap.new()
    territory = {c: MY_TERRITORY for c in square(25, 4)}
    territory.update({c: MY_TERRITORY for c in square(21, 4)})
    territory.update({(23, 8): OPP_TERRITORY, (23, 9): OPP_TERRITORY})
    memory.update(
        make_observation(
            pos=(25, 7),
            turn=3,
            territory=territory,
            trail={(22, 8): OPP_TRAIL},
            my_trail=[(25, 6), (25, 7)],
        )
    )
    return memory


def cells(memory, layer, code):
    return set(np.flatnonzero(getattr(memory, layer) == code).tolist())


# ---- reset_my_land ------------------------------------------------------


def test_my_death_leaves_only_my_start_land_and_no_trail_of_mine():
    memory = populated()
    assert len(cells(memory, "territory", MY_TERRITORY)) == 18
    memory.reset_my_land(MY_START, turn=9)
    assert cells(memory, "territory", MY_TERRITORY) == set(MY_START)
    assert cells(memory, "trail", MY_TRAIL) == set()
    memory.validate()


def test_my_death_does_not_touch_what_is_remembered_of_the_opponent():
    memory = populated()
    memory.reset_my_land(MY_START, turn=9)
    assert cells(memory, "territory", OPP_TERRITORY) == {flat(23, 8), flat(23, 9)}
    assert cells(memory, "trail", OPP_TRAIL) == {flat(22, 8)}


def test_the_cells_a_death_changes_are_stamped_with_that_turn_and_the_rest_are_not():
    memory = populated()
    memory.reset_my_land(MY_START, turn=9)
    assert memory.last_seen[21, 4] == 9    # extra land of mine, now gone
    assert memory.last_seen[25, 6] == 9    # a trail cell of mine, now gone
    assert memory.last_seen[23, 8] == 3    # enemy land: not touched, keeps its old stamp


def test_a_death_makes_start_cells_known_even_if_never_seen():
    memory = MemoryMap.new()
    memory.reset_my_land(MY_START, turn=5)
    assert cells(memory, "territory", MY_TERRITORY) == set(MY_START)
    assert (memory.last_seen[memory.territory == MY_TERRITORY] == 5).all()
    assert memory.territory[0, 0] == UNKNOWN and memory.last_seen[0, 0] == NEVER_SEEN
    memory.validate()


# ---- reset_opponent_land ------------------------------------------------


def test_the_opponents_death_leaves_only_their_start_land_and_no_trail_of_theirs():
    memory = populated()
    memory.reset_opponent_land(OPP_START, turn=9)
    assert cells(memory, "territory", OPP_TERRITORY) == set(OPP_START)
    assert cells(memory, "trail", OPP_TRAIL) == set()
    assert len(cells(memory, "territory", MY_TERRITORY)) == 18  # mine is untouched
    memory.validate()


# ---- apply_capture ------------------------------------------------------


def test_a_capture_turns_the_trail_and_the_enclosed_region_into_my_land():
    memory = populated()
    region = bitboard.from_cells([flat(26, 6), flat(26, 7)])
    memory.apply_capture([flat(25, 6), flat(25, 7)], region, turn=10)
    for cell in [(25, 6), (25, 7), (26, 6), (26, 7)]:
        assert memory.territory[cell] == MY_TERRITORY and memory.trail[cell] == FREE
        assert memory.last_seen[cell] == 10
    assert cells(memory, "trail", MY_TRAIL) == set()
    assert len(cells(memory, "territory", MY_TERRITORY)) == 18 + 4
    memory.validate()


def test_a_capture_takes_enclosed_enemy_land_and_reaches_cells_never_seen():
    memory = populated()
    far = flat(40, 40)
    region = bitboard.from_cells([flat(23, 8), far])
    memory.apply_capture([flat(25, 6)], region, turn=11)
    assert memory.territory[23, 8] == MY_TERRITORY            # was the opponent's
    assert memory.territory[40, 40] == MY_TERRITORY and memory.trail[40, 40] == FREE and memory.last_seen[40, 40] == 11
    memory.validate()


# ---- the trail under an agent is hidden --------------------------------


def test_the_trail_remembered_under_my_agent_survives_a_window_that_cannot_show_it():
    memory = MemoryMap.new()
    memory.update(make_observation(pos=(25, 4), turn=1, trail={(24, 4): OPP_TRAIL}, opp_at=(23, 5)))
    assert memory.trail[24, 4] == OPP_TRAIL
    # I am now standing on (24, 4); the framework shows me, not the trail beneath.
    later = make_observation(pos=(24, 4), turn=2, respawn=(24, 4), trail_hidden=(flat(24, 4),))
    memory.update(later)
    assert memory.trail[24, 4] == OPP_TRAIL
    assert memory.agrees_with(later)


def test_with_nothing_remembered_under_my_agent_the_observations_guess_is_used():
    memory = MemoryMap.new()
    memory.update(make_observation(pos=(24, 4), turn=2, respawn=(24, 4), trail_hidden=(flat(24, 4),)))
    assert memory.trail[24, 4] == FREE


def test_under_the_opponent_the_guess_replaces_old_memory():
    memory = MemoryMap.new()
    memory.update(make_observation(pos=(25, 4), turn=1))
    assert memory.trail[24, 6] == FREE
    # The opponent has since stepped onto (24, 6), off its own land, laying a trail there.
    seen = make_observation(pos=(25, 4), turn=2, trail={(24, 6): OPP_TRAIL}, opp_at=(24, 6), trail_hidden=(flat(24, 6),))
    memory.update(seen)
    assert memory.trail[24, 6] == OPP_TRAIL


def test_agrees_with_ignores_the_hidden_trail_cell_only():
    memory = MemoryMap.new()
    memory.update(make_observation(pos=(25, 4), turn=1, trail={(24, 4): OPP_TRAIL}, opp_at=(23, 5)))
    later = make_observation(pos=(24, 4), turn=2, respawn=(24, 4), trail_hidden=(flat(24, 4),))
    memory.update(later)
    memory.trail[26, 4] = OPP_TRAIL      # a visible cell that now disagrees
    assert not memory.agrees_with(later)
