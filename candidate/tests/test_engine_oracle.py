"""Checks our data structures against the real engine as ground truth."""

import numpy as np
import pytest

from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH
from candidate.agent.data.cells import FREE, MY_TERRITORY, MY_TRAIL, OPP_TERRITORY, OPP_TRAIL
from candidate.agent.data.memory import MemoryMap
from candidate.agent.geometry import bitboard
from candidate.examples.basic_state_machine import BasicStateMachine
from candidate.examples.random_state_machine import RandomStateMachine
from candidate.tests.engine_support import (
    Marcher,
    Pacer,
    Recording,
    play_match,
    world_from_snapshot,
)
from territorygame.api.move_result import MoveResult
from territorygame.controller.enemy_state_machine import EnemyStateMachine

MATCH_TURNS = 250
MATCHUPS = {
    "enemy_vs_basic": lambda: [EnemyStateMachine(), BasicStateMachine()],
    "basic_vs_random": lambda: [BasicStateMachine(), RandomStateMachine()],
    "random_vs_enemy": lambda: [RandomStateMachine(), EnemyStateMachine()],
}
CROSSING_TURNS = 60


@pytest.fixture(scope="module")
def matches() -> dict[str, list]:
    return {name: play_match(make(), MATCH_TURNS) for name, make in MATCHUPS.items()}


@pytest.fixture(scope="module")
def crossing() -> list:
    """Player 0 marches through player 1's territory; player 1 paces at home."""
    return play_match([Marcher(), Pacer()], CROSSING_TURNS)


# ---- WorldState ---------------------------------------------------------


@pytest.mark.parametrize("name", MATCHUPS)
def test_every_engine_state_is_a_valid_world_from_both_perspectives(matches, name):
    for snapshot in matches[name]:
        assert snapshot.error_message is None
        for me in (0, 1):
            world = world_from_snapshot(snapshot, me)
            world.validate()


@pytest.mark.parametrize("name", MATCHUPS)
def test_bitboard_counts_match_the_engines_territory_counts(matches, name):
    for snapshot in matches[name][:: 5]:
        for me in (0, 1):
            world = world_from_snapshot(snapshot, me)
            bbs = world.bitboards()
            assert all(bitboard.is_valid(b) for b in bbs)
            assert bitboard.popcount(bbs.my_territory) == snapshot.players[me].territory_count
            assert bitboard.popcount(bbs.opp_territory) == snapshot.players[1 - me].territory_count
            assert bitboard.popcount(bbs.my_trail) == len(snapshot.players[me].trail)
            assert bitboard.popcount(bbs.opp_trail) == len(snapshot.players[1 - me].trail)


def test_the_random_matches_cover_captures_and_deaths(matches):
    results = {s.last_move_result for snapshots in matches.values() for s in snapshots}
    assert MoveResult.CAPTURED in results and MoveResult.DIED in results


def test_a_trail_lying_on_the_other_players_land_is_a_real_engine_state(crossing):
    over_enemy_land = 0
    for snapshot in crossing:
        for me in (0, 1):
            world = world_from_snapshot(snapshot, me)
            world.validate()
            mine_on_theirs = (world.trail == MY_TRAIL) & (world.territory == OPP_TERRITORY)
            theirs_on_mine = (world.trail == OPP_TRAIL) & (world.territory == MY_TERRITORY)
            if me == 0:
                over_enemy_land += int(mine_on_theirs.any())
            # The single-code view cannot say which land is underneath.
            for y, x in zip(*np.nonzero(mine_on_theirs | theirs_on_mine)):
                assert world.cells[y, x] in (MY_TRAIL, OPP_TRAIL)
                assert world.territory[y, x] in (MY_TERRITORY, OPP_TERRITORY)
    assert over_enemy_land > 0, "the marcher never crossed the enemy's territory"


def test_while_crossing_the_head_is_the_last_trail_cell(crossing):
    marched = [s for s in crossing if s.players[0].trail]
    assert marched
    for snapshot in marched:
        trail = snapshot.players[0].trail
        assert trail[-1] == snapshot.players[0].position


# ---- Observation and MemoryMap -----------------------------------------


@pytest.fixture(scope="module")
def observed_crossing():
    """The crossing match again, with each controller's observations recorded."""
    sink: list = []
    controllers = [
        Recording(Marcher(), 0, CROSSING_TURNS, sink),
        Recording(Pacer(), 1, CROSSING_TURNS, sink),
    ]
    return play_match(controllers, CROSSING_TURNS), sink


@pytest.fixture(scope="module")
def observed_enemy_vs_basic():
    sink: list = []
    controllers = [
        Recording(EnemyStateMachine(), 0, MATCH_TURNS, sink),
        Recording(BasicStateMachine(), 1, MATCH_TURNS, sink),
    ]
    return play_match(controllers, MATCH_TURNS), sink


def _check_observations_against_truth(snapshots, sink):
    """Compare every recorded Observation with the engine's true board.
    Returns (observations checked, times the opponent was in view, opponent
    cells whose trail-layer guess differed from the truth)."""
    assert len(sink) == len(snapshots) - 1
    memories = {0: MemoryMap.new(), 1: MemoryMap.new()}
    in_view = guessed_wrong = 0
    for j, (player, obs) in enumerate(sink):
        snapshot = snapshots[j]  # the state this player faced
        me = snapshot.players[player]
        assert me.remaining_turns == obs.turns_left and obs.my_pos == me.position.y * BOARD_WIDTH + me.position.x
        truth = world_from_snapshot(snapshot, player)

        rows, cols = obs.window_slices
        assert obs.my_count == truth.my_count and obs.opp_count == truth.opp_count
        assert list(obs.my_trail) == truth.my_trail
        assert np.array_equal(obs.window_territory, truth.territory[rows, cols])  # territory is always visible

        trail_window = truth.trail[rows, cols]
        differing = np.argwhere(obs.window_trail != trail_window)
        hidden = obs.hidden_mask
        if obs.opp_seen_at is not None:
            in_view += 1
            assert obs.opp_seen_at == truth.opp_pos
            assert obs.opp_seen_at in obs.trail_hidden
        else:
            assert not obs.covers(truth.opp_pos)
        for dy, dx in differing:
            # Only a cell an agent stands on, whose trail the framework hides, may differ.
            assert hidden[dy, dx], "a visible cell was translated wrongly"
            guessed_wrong += 1

        # Window geometry: centred on the agent and clipped to the board.
        y, x = divmod(truth.my_pos, BOARD_WIDTH)
        assert obs.window_origin == (max(0, y - 5), max(0, x - 5))
        assert obs.window_territory.shape == (min(BOARD_HEIGHT - 1, y + 5) - obs.window_origin[0] + 1,
                                              min(BOARD_WIDTH - 1, x + 5) - obs.window_origin[1] + 1)

        memories[player].update(obs)
        assert memories[player].agrees_with(obs)
        memories[player].validate()
    return len(sink), in_view, guessed_wrong


def test_observations_match_the_true_board_while_crossing(observed_crossing):
    snapshots, sink = observed_crossing
    checked, in_view, _ = _check_observations_against_truth(snapshots, sink)
    assert checked == 2 * CROSSING_TURNS
    assert in_view > 0, "the two agents never saw each other"


def test_observations_match_the_true_board_in_a_longer_match(observed_enemy_vs_basic):
    snapshots, sink = observed_enemy_vs_basic
    checked, _, _ = _check_observations_against_truth(snapshots, sink)
    assert checked == 2 * MATCH_TURNS


def test_memory_after_a_match_agrees_with_truth_inside_the_last_window(observed_crossing):
    snapshots, sink = observed_crossing
    memory = MemoryMap.new()
    last_player0 = [obs for player, obs in sink if player == 0][-1]
    for player, obs in sink:
        if player == 0:
            memory.update(obs)
    truth = world_from_snapshot(snapshots[len(sink) - 2], 0)  # state faced on player 0's last turn
    rows, cols = last_player0.window_slices
    assert np.array_equal(memory.territory[rows, cols], truth.territory[rows, cols])
    # Seen land outside the window may have changed since; memory just holds the last sighting.
    assert memory.known_mask().sum() > last_player0.window_territory.size
    assert (memory.territory[~memory.known_mask()] == -1).all()
    assert FREE == 0
