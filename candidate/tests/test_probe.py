"""The probe policy, alone and driven through the run-time layer."""


from candidate.agent.data.types import Direction, MoveResult, Side
from candidate.agent.diagnostics import Diagnostics
from candidate.agent.geometry.grid import flat
from candidate.agent.probe import Phase, ProbePolicy
from candidate.agent.rules.match import MatchRules
from candidate.agent.runtime import AgentRuntime, Event
from candidate.tests.builders import put_trail
from candidate.tests.live import LiveGame

N, E, S, W = Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST
RULES = MatchRules.from_respawns(flat(25, 4), flat(25, 45))


def pace(live):
    live.opponent_moves(E if live.world().opp_pos == flat(25, 45) else W)


def drive(live, runtime, policy, turns):
    states = []
    for _ in range(turns):
        view = live.begin(runtime)
        if Event.MATCH_STARTED in view.events:
            policy.reset()
        direction = policy.choose(view)
        result = runtime.move(live.me, direction)
        policy.on_result(view, direction, result)
        states.append(policy.debug_state())
        pace(live)
    return states


def test_choosing_does_not_advance_the_plan():
    runtime = AgentRuntime(Diagnostics())
    live = LiveGame(RULES.initial_world(), RULES)
    policy = ProbePolicy(seed=3)
    view = live.begin(runtime)
    first = policy.choose(view)
    state = policy.debug_state()
    assert [policy.choose(view) for _ in range(5)] == [first] * 5 and policy.debug_state() == state


def test_it_leaves_home_makes_loops_and_closes_them():
    runtime = AgentRuntime(Diagnostics())
    live = LiveGame(RULES.initial_world(), RULES)
    policy = ProbePolicy(seed=1, hunt=False)
    states = drive(live, runtime, policy, 200)
    phases = {s.split()[0] for s in states}
    assert {"HOME", "OUT", "ACROSS", "BACK"} <= phases
    assert policy.captures >= 8 and policy.deaths == 0
    assert runtime.diag.errors() == []


def test_it_goes_home_when_it_finds_itself_exposed_without_a_plan():
    world = RULES.initial_world()
    put_trail(world, Side.ME, [(25, 6), (25, 7), (26, 7), (26, 6)])
    runtime = AgentRuntime(Diagnostics())
    view = LiveGame(world, RULES).begin(runtime)
    policy = ProbePolicy(hunt=False)
    assert policy.choose(view) == W and policy.debug_state() == "BACK"


def test_it_steps_onto_an_adjacent_opponent_trail_to_cut_it():
    world = RULES.initial_world()
    put_trail(world, Side.ME, [(25, 6), (25, 7)])
    put_trail(world, Side.OPP, [(25, 8), (24, 8)])
    view = LiveGame(world, RULES).begin(AgentRuntime(Diagnostics()))
    assert ProbePolicy(hunt=True).choose(view) == E


def test_with_suicide_enabled_it_steps_onto_its_own_trail_when_due():
    world = RULES.initial_world()
    put_trail(world, Side.ME, [(25, 6), (25, 7), (26, 7), (26, 6)])
    view = LiveGame(world, RULES).begin(AgentRuntime(Diagnostics()))
    policy = ProbePolicy(hunt=False, suicide_every=1)
    policy.excursions = 1
    assert policy.choose(view) == N  # (25, 6) is its own trail
    quiet = ProbePolicy(hunt=False, suicide_every=None)
    quiet.excursions = 1
    assert quiet.choose(view) == W


def test_a_refused_move_does_not_advance_the_plan_but_an_accepted_one_does():
    runtime = AgentRuntime(Diagnostics())
    view = LiveGame(RULES.initial_world(), RULES).begin(runtime)
    policy = ProbePolicy()
    policy._phase, policy._direction, policy._left = Phase.OUT, E, 3
    policy.on_result(view, E, MoveResult.INVALID)
    assert policy._left == 3
    policy.on_result(view, E, MoveResult.NORMAL)
    assert policy._left == 2
    policy.on_result(view, E, MoveResult.CAPTURED)
    assert policy._phase == Phase.HOME and policy.captures == 1 and policy.excursions == 1


def test_reset_starts_a_new_match():
    policy = ProbePolicy(seed=2)
    policy.captures, policy.deaths, policy.excursions, policy._phase = 5, 2, 7, Phase.BACK
    policy.reset()
    assert (policy.captures, policy.deaths, policy.excursions, policy._phase) == (0, 0, 0, Phase.HOME)


def test_the_same_seed_makes_the_same_loops():
    runs = []
    for _ in range(2):
        runtime = AgentRuntime(Diagnostics())
        live = LiveGame(RULES.initial_world(), RULES)
        runs.append(drive(live, runtime, ProbePolicy(seed=9, hunt=False), 60))
    assert runs[0] == runs[1]
