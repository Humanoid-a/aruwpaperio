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
