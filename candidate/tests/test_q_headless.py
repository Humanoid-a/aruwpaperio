"""A direct run of the game with no display: the real GameWindow, its own loop and the GUI's own controller registry,
driven by posted clicks. Checks that the right agents play and that the Candidate Controller records the game exactly
as the engine played it."""

import pytest

pytest.importorskip("pygame")

from candidate.agent.q.record import load_record  # noqa: E402
from candidate.candidate_controller import CandidateController  # noqa: E402
from candidate.offline.headless import run_window, verify_record  # noqa: E402
from territorygame.controller.enemy_state_machine import EnemyStateMachine  # noqa: E402


@pytest.mark.parametrize("seat", [0, 1])
def test_a_direct_run_plays_the_right_agents_and_records_every_frame_exactly(seat, tmp_path):
    result = run_window(turns=60, seat=seat, record_dir=tmp_path)
    slots = result["controllers"]
    assert isinstance(slots[seat], CandidateController) and slots[seat] is result["controller"]
    assert isinstance(slots[1 - seat], EnemyStateMachine)
    truth = result["truth"]
    assert len(truth) == 2 * 60 + 1 and truth[-1]["over"]                 # the fresh board, then one per turn
    assert result["record"] is not None and result["record"].parent == tmp_path
    assert verify_record(result["record"], truth, seat) == []
    meta, arrays = load_record(result["record"])
    assert meta["complete"] and meta["frames"] == 60 and meta["label"] == "gui" and meta["q_model"] == "HeuristicQ"
    final = truth[-1]
    assert final["counts"][seat] >= 9                                       # it played: kept or grew its land


def test_verify_record_catches_a_record_that_does_not_match_the_game(tmp_path):
    result = run_window(turns=30, seat=0, record_dir=tmp_path)
    wrong = [dict(board, counts=[c + 1 for c in board["counts"]]) for board in result["truth"]]
    problems = verify_record(result["record"], wrong, 0)
    assert problems and "recorded counts" in problems[0]


def test_an_unknown_opponent_is_refused(tmp_path):
    with pytest.raises(ValueError):
        run_window(turns=10, opponent="Nobody", record_dir=tmp_path)
