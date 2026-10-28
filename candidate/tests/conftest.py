import pytest

from candidate.tests.rules_oracle import play_random_game

FUZZ_EPISODES = 24
FUZZ_STEPS = 250


@pytest.fixture(scope="session")
def fuzz_records() -> list:
    """Moves from randomly played games on the framework's real rules, as
    (world_before, rules, side, direction, real_outcome). Generated once."""
    records = []
    for seed in range(FUZZ_EPISODES):
        records.extend(play_random_game(seed, FUZZ_STEPS))
    return records


@pytest.fixture(autouse=True)
def _gui_controller_plays_the_stand_in(monkeypatch):
    """The GUI's controller may be set to play a local, git-ignored weights file (candidate_controller.DEFAULT_WEIGHTS,
    or $CANDIDATE_Q_WEIGHTS). Tests must not depend on such a file: they see the hand-set stand-in unless a test sets
    weights itself."""
    import candidate.candidate_controller as cc

    monkeypatch.setattr(cc, "DEFAULT_WEIGHTS", None)
    monkeypatch.delenv(cc.WEIGHTS_VARIABLE, raising=False)
    monkeypatch.setattr(cc, "DEFAULT_K_WEIGHTS", None)
    monkeypatch.delenv(cc.K_WEIGHTS_VARIABLE, raising=False)
