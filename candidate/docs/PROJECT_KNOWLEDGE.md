# Project Knowledge

Game facts and lifecycle below were verified against the question PDF, source, and tests on 2026-10-06. The design sections were reconciled with [`method1.0.pdf`](../../method1.0.pdf) on 2026-10-08. See [`INDEX.md`](INDEX.md) for which document wins on a conflict.

## Purpose and deliverables

The task is to design and implement a state-machine-based controller for one agent in a two-player territory-capture game. The design reference is [`method1.0.pdf`](../../method1.0.pdf); see "Design reference: method1.0" below. Everything we write stays inside `candidate/`. The submission must include:

- controller code;
- a concise state-machine design;
- state responsibilities and important transitions; and
- reasoning that the candidate can explain and defend.

Review emphasizes design reasoning, implementation quality, extensibility, and testability. Game score is useful feedback, not the only criterion. AI assistance is allowed, but the candidate is expected to understand every submitted choice.

## Rules and configured match

- Board: 50 x 50.
- Players: exactly two in the current rules and state implementation.
- Starting territory: a 3 x 3 square around each respawn point.
- Visibility: an 11 x 11 window centered on the agent and clipped at board edges.
- Turn budget: 2,000 successful moves per player.
- Movement: one cardinal cell (`NORTH`, `SOUTH`, `EAST`, or `WEST`).
- Invalid moves: off-board or onto the opponent's current position. They do not consume a turn and the controller is called again.
- Trail: leaving owned territory creates or extends an exposed trail.
- Capture: re-entering owned territory with a non-empty trail converts the trail and newly enclosed cells to owned territory.
- Self collision: stepping on one's own trail causes death.
- Attack: stepping on the opponent's trail respawns the opponent and completes the mover's turn.
- Death: clears all of the dead player's current territory and active trail, restores starting territory, and respawns the agent.
- Winner: the player with more current territory after both turn budgets reach zero; equal territory is a draw.

Default respawns are `(4, 25)` and `(45, 25)`. All defaults live in [`territorygame/resources/game-config.properties`](../../territorygame/resources/game-config.properties).

## Candidate boundary

The submission entry point is [`candidate/candidate_controller.py`](../candidate_controller.py). Supporting candidate-owned modules may be added under `candidate/`; framework code should not need modification.

The framework keeps a controller instance alive across turns, so instance fields are the controller's persistent memory. On each call to `take_turn(game)`, the controller reads a player-scoped `GameApi` and calls `game.move(direction)`.

The candidate can observe exactly:

- own position and respawn position;
- own complete active trail, ordered oldest to newest;
- both territory counts;
- own remaining turns;
- board dimensions;
- the current visible cells, including relative `SELF`/`OPPONENT` territory and occupants; and
- the `MoveResult` returned by its own attempted move.

The candidate cannot access the authoritative full board through `GameApi`. Returned collections are defensive copies.

Provided helpers cover position arithmetic, bounds, Manhattan distance, visible-cell lookup, mechanically valid directions, and a last-seen `ObservedBoard`. Mechanical validity does **not** reject a move onto the controller's own trail.

## Exact turn and move lifecycle

The implementation is sequential, not simultaneous:

```text
active player selected
  -> reset the API's per-turn move guard
  -> call controller
  -> INVALID: call it again, up to 20 attempts
  -> successful result: consume one mover turn
  -> publish a full observer snapshot
  -> advance to the next player who still has turns
```

If a controller raises an exception or never makes a successful move within 20 attempts, that player forfeits one turn. The engine records `INVALID` plus an error message and continues the match.

For a requested move, [`MoveResolver`](../../territorygame/rules/move_resolver.py) applies this order:

1. Reject an off-board destination or the opponent's current cell as `INVALID`.
2. If the destination contains the mover's trail, respawn the mover, count a death, consume the turn, and return `DIED`.
3. Move the agent to the destination.
4. If the destination is owned territory and the mover has an active trail, apply capture.
5. If the destination held the opponent's trail, respawn the opponent and update kill/death counts.
6. If capture occurred, consume the turn and return `CAPTURED`.
7. Otherwise, add a mover trail cell when the destination is not mover-owned territory, consume the turn, and return `MOVED`.

Consequences worth remembering:

- Entering owned territory with no active trail is an ordinary `MOVED` result.
- A kill normally returns `MOVED`; there is no separate kill result.
- Capture takes precedence in the returned result if capture and an opponent-trail cut happen on the same destination.
- Killing the opponent does not consume one of the opponent's turns.
- Capture runs before opponent respawn on a same-move capture/kill; respawn then restores the opponent's starting territory.

## Capture and respawn details

Capture converts the active trail first, then uses an edge flood-fill to find cells newly enclosed by the capturer's territory. It records regions already enclosed before converting the trail, so pre-existing holes are not captured merely because a later unrelated trail closes.

Respawn clears all current territory belonging to the dead player, restores that player's configured starting cells, and normally places the agent at its respawn position. If occupied by the opponent, the nearest unoccupied starting cell is used; if necessary, the nearest free board cell is used.

## Runtime and offline experimentation

The GUI is launched with:

```powershell
python -m territorygame.main
```

The engine can run without pygame by constructing `GameEngine`, attaching a `GameObserver`, and calling `reset`, `start`, or `step`; the tests demonstrate this. Observer snapshots contain the full board, both positions, trails, counts, and debug labels, so framework-side offline logging can see authoritative state without exposing it to the candidate controller.

There is not yet a purpose-built synchronous batch runner, trajectory writer, or engine shutdown method. Each `GameEngine` owns a daemon worker thread. A serious multi-match experiment harness should address lifecycle and throughput rather than blindly creating thousands of engines. The framework is read-only for us, so any such harness lives in `candidate/offline/` and works around these limits from the outside.

## Randomness and reproducibility

The configuration contains one controller seed per player slot. The GUI passes those seeds to controller factories, but the current factories ignore the argument and the example/reference controllers use hard-coded seeds (`42` or `7`). Therefore editing `controller.seed.*` currently does not change their behavior, and `EnemyStateMachine` has no seed argument at all (it hard-codes `random.Random(42)`). The framework is read-only for us, so seeded or varied-opponent experiments have to be arranged from candidate code, for example with a subclass that sets the controller's random source.

## Design reference: method1.0

The pipeline is: observation -> memory map -> belief over the hidden opponent -> per-turn snapshot (distance maps, safety margin, kill windows) -> explicit state machine -> safety filter -> move. Design principles from the spec:

- The state machine is explicit, with a declarative prioritized transition table of named, individually testable guards. Learned models feed guards and plan choice; they never replace transitions.
- Safety is worst-case and hard-coded. A learned score can never override it.
- Expensive geometry (flood fill, reachability) is computed exactly; models only learn weights on top of it.

method1.0's development process (its milestone order, per-milestone gates, and build guidance) is deliberately not followed. What to build, and in what order, comes from the user's instructions.

Keep these terms separate:

- **World state** (`WorldState`): the complete true board, held by the framework. The controller never sees it; self-play logging reads it for ground truth.
- **Observation**: one turn's player-scoped API values and visible window.
- **Memory / belief**: the `MemoryMap` of last-seen cells plus weighted hypotheses about the hidden opponent.
- **Action-state / controller state**: the behavior being executed: SAFE, EXPAND, RETURN, or HUNT.

## Reconciling method1.0 with the framework

method1.0 says its framework assumptions came from the booklet alone. Checked against this repo (2026-10-08):

| Topic | method1.0 assumes | This repo | Resolution |
| --- | --- | --- | --- |
| Dependencies | NumPy, SciPy, and PyTorch available in-process | `README.md` says `pip install -r requirements.txt` (pygame, pytest) is all the setup needed. `requirements.txt` has an uncommitted edit adding `numpy` and `pytorch`; the PyPI package is named `torch`. SciPy is not listed. | Unconfirmed whether the reviewers' environment allows these. The spec points to robomstr@uw.edu (the booklet's contact). Anything beyond the standard library is a risk until confirmed, and the spec itself requires a working CPU-only path. |
| Result names | `MoveResult.NORMAL` | `MoveResult.MOVED` | The adapter maps framework values to the agent's own `MoveResult`. |
| Coordinates | `(y, x)` cells, flat index `50*y + x` | `GridPosition(x, y)`, visible grid indexed `[y][x]` | Convert once, in the adapter. |
| Cell codes | One int8 code per cell: 0 free, 1 my territory, 2 my trail, 3 opponent territory, 4 opponent trail | `BoardCell` allows a territory owner and a *different* player's trail on the same cell (a trail crosses free space or enemy land). Confirmed against the engine: a marching agent lays its trail across the other player's territory. One code per cell would lose the land under a trail. | `WorldState`, `MemoryMap`, and `Observation` keep two layers (territory, trail) using the same code numbers. The single-code view (`combine`) is lossy and for display only. |
| Head and trail | A hypothesis's `pos` is never a bit of its own trail | The engine sets the position, then appends the same cell, so `get_active_trail()` ends at the agent's cell whenever it is non-empty. Confirmed against the engine in every state of the test matches. | Our types, including the belief's hypotheses, store the trail with the head included and check it (`pos` is in the trail exactly when the trail is not empty). Because every step outside its land adds one trail cell, a hypothesis' `trail_len` equals method1.0's `turns_out`, so that field is not stored. |
| What agents hide | The window as cell codes | A cell holding an agent is reported as the agent, so any trail beneath it is invisible. Territory is always visible. This matters in practice: a player that respawns onto a cell the other player's trail crosses stands on that trail unseen (9 times in one 600-turn match against a hunting opponent). | `Observation.trail_hidden` lists every such cell and its trail layer there is a guess. Under my own agent the memory keeps what it knew (right in all 9 cases); under the opponent the guess replaces old memory, because the opponent lays a trail wherever it steps off its land. Opponent-cell guess: certain that there is no trail on its own land, usual that there is one elsewhere, wrong only for an empty trail on enemy land after a capture around it. Not recoverable: a trail laid across a cell while it was out of view stays unknown until seen. |
| Being killed by the opponent | Open question: how it is reported | `GameApi` has no event channel. `MoveResult` is only the return value of the controller's own `move()`. | Inferred in `AgentRuntime`: my last move left a trail, and now the trail is empty and I stand on my respawn cell (or the alternate cell). Verified against the engine's own death counters (46 of 46 and 25 of 25 deaths, with a hunting opponent). A death at exactly 9 cells does not change the count, so the count is not used. |
| Exceptions in `take_turn` | Not mentioned | `TurnManager` catches any exception from `take_turn`, records an error, and forfeits that turn (the turn is lost). | `Controller` never raises: every failure is recorded and answered with a move, preferring the way home. |
| Swapping or resetting the controller | A controller lives for one match | The GUI builds a fresh controller on Reset, and `set_controller` can swap one in mid-match without resetting the board. | `AgentRuntime` detects a new match (more turns left than before, or a repeat call with a different board) and starts fresh; joining mid-match works, with memory knowing only what it has seen. |
| Territory map | Open question: full map or only counts | Both counts plus the visible window only (11 x 11, clipped at edges) | Keep a `MemoryMap`; remembered cells outside the window are a prior, never certain evidence. |
| Opponent spawn | Known or mirrors mine | The opponent's respawn is not exposed by `GameApi`. The default config (`(4, 25)`, `(45, 25)`) mirrors x as `49 - x`. | The belief may start from the mirrored spawn; it should not depend on that. |
| Turn order | Exactly one opponent move between my calls | Consistent with the lifecycle above while both players have turns; a forfeit still consumes a turn. | Keep the +/-1 in margins. Opponent invalid retries are invisible to the controller. |
| Per-move time limit | Open question | None stated in the booklet or `CANDIDATE_GUIDE.md` | Unconfirmed; budget hypotheses conservatively. |
| Capture semantics | Enclosed means not reachable from the board edge | The engine uses an edge flood-fill but excludes regions already enclosed before the trail converts. What happens to an agent inside a captured region is not documented here. | Verify the fast and reference transitions against the engine as an oracle (see `test_territory_resolver.py`, `test_move_resolver.py`). |
| Headless self-play | Drives the engine headless and reads true state | No synchronous batch runner; each `GameEngine` owns a daemon thread; configured seeds are ignored (see above). | A self-play runner in `candidate/offline/` built on `GameEngine` plus a `GameObserver`, handling thread lifecycle and seeds from candidate code only. |
| Code placement | `agent/` and `offline/` wherever importable | `CANDIDATE_GUIDE.md`: nothing outside `candidate/` should change; the user has also ruled out any edit outside it | Everything goes under `candidate/`: `agent/`, `offline/`, `tests/`, `docs/` (see `CODEBASE_INDEX.md`). |

## Current implementation status

- Built (2026-10-08 to 2026-10-10), all in `candidate/agent/` and `candidate/offline/` with about 700 tests in `candidate/tests/` (see [`ARCHITECTURE_GUIDE.md`](ARCHITECTURE_GUIDE.md) for a guided tour): the data structures (`config`, `data/{types,cells,observation,world,memory,territory,hypotheses,residual,belief_view}`), the geometry (`geometry/{grid,bitboard,fill,distance,zobrist}`), the move rules (`rules/{match,reference,transition}`), the run-time layer (`adapter`, `prediction`, `runtime`, `diagnostics`, `controller`, and the loop-making `probe`), the belief over the hidden opponent, one module per step of a frame (`belief/{config,motion,state,land,generate,cut,weigh,settle,after_move,reseed,publish,filter}`), the action-states and chooser (`behavior/*`), and the Q model, its inputs, network and strategy (`q/*`, see `Q_BUILD_PLAN.md`), plus `candidate/runtime_check.py`. See `CODEBASE_INDEX.md`.
- Move rules are verified three ways: the reference and the fast bitboard version agree with the framework's real `MoveResolver` on hand-built scenarios, on every move of randomly played games (about 6,000 per test run, 60,000 in a one-off larger run, with zero mismatches), and on every turn of full engine matches. See "Move rules as implemented" below.
- `candidate_controller.py` is a thin wrapper around `agent.controller.Controller` and now plays `agent.q.strategy.QStrategy`: the six action-states (EXPAND, EXPAND2, RETURN, HUNT, TRACK TARGET, PATROL) and the chooser with its hard rules, in `agent/behavior/` (which knows nothing of Q), and the Q model in `agent/q/`. The Q model's training pipeline is built (steps 7-10 of `Q_BUILD_PLAN.md`): recording (in the GUI too), headless collection with exact replay, the data handler, the trainer (GPU when it works), one PyTorch network for training and play, evaluation, and online training by epochs with randomised opponents and a gate. K has a learned version too (`belief/neural.py`, trained every online epoch). The controller plays online15's epoch 25 by default, its Q network and its trained K (`q_training/online/online15/epoch_025/`); with PyTorch or the file missing it falls back to the hand-set stand-in. The probe is kept as an explicit strategy for testing the runtime.
- Not yet exercised against the engine: the empty-trail-on-enemy-land case for the opponent-cell guess (covered only by a hand-built `WorldState` test). Also unverified: how the pipeline copes when the opponent's real respawn is not the mirror of mine (the code assumes the mirror, which the default config satisfies).

## Belief over the hidden opponent

`AgentRuntime` owns a `BeliefFilter` and hands every strategy `TurnView.belief`, a `BeliefView` (position probabilities, support, entropy, trail marginal on demand, exposure probability, events). It follows method1.0's filter, with these differences, each forced by something verified in the code:

- **The order of a frame.** Observe: (1) generate every move of every hypothesis, resolved by the game rules; (2) cut what the rules and the observation make impossible; (3) apply K to what is left and give each child its weight; (4) settle: merge, normalise, prune into the residual, re-base on memory, rehydrate. Then the agent moves, and (5) at the end of the frame, once the engine has accepted the move, it is applied to every hypothesis (`BeliefFilter.after_my_move`, called by `AgentRuntime.move`). So the hypotheses always describe the world right after my latest move and before the opponent's reply, and nothing is applied at the start of a frame (an earlier version applied it at the start of the next frame; the two are the same on every state, which was checked frame for frame, but this one needs no `last move` handed over and cannot be forgotten). Applying my move only touches the states when it did something: a capture takes the enclosed cells from the opponent's land and count and gives them to my land; my death leaves me my start land; cutting its trail collapses the belief to the one certain state; an ordinary step changes and copies nothing. It does not raise: the move is already made, and a failure only makes the belief rebuild itself from the next observation.
- **K comes after the cuts, over all the legal moves.** Cutting first means K is run only for the parents that still have a child (when the opponent is in view or the counts are selective, a handful of hypotheses instead of thousands). But K is still shown all four moves of those parents, and the probabilities are over all the moves the rules allow, not only the survivors: a parent whose other moves were ruled out by the evidence is as unlikely as K says its remaining move is, not certain. (Normalising over the survivors would erase exactly the information the evidence carries.)
- **Hypotheses and residual.** A hypothesis is one complete guess about the opponent (position, trail, territory, count) and about my own land (see below). Probability whose history has been thrown away lives in a per-cell residual, so pruning never changes the probability of any position.
- **The per-position budget** (`BeliefConfig.budget`, default `ProportionalBudget(total=2000, min_keep=1, max_keep=1500)`): each position keeps `ceil(total x probability)` hypotheses, at least 1 and at most 1500; the rest go to that position's residual. Because every occupied cell keeps at least one, the stored count can exceed `total` (observed mean 150 to 3,000, maximum about 4,000; hard cap `capacity=8000`).
- **Territory as a difference from memory**, re-expressed against the memory after every window update, so differences stay small and never go stale. This is done for both sides: `opp_land_id` is the opponent's land under the hypothesis, `my_land_id` is mine (a second `TerritoryDiffs`, `BeliefState.my_diffs`, against the cells memory believes are mine). My land has to be per hypothesis because what the opponent did decides what it took from me, and for cells out of view memory cannot know. My own move changes it for every hypothesis (my capture adds the cells, my death leaves my start land, cutting its trail returns its start land out of mine); a hypothesis' own capture removes the cells it encloses from its view of my land, its death removes its start land, and killing me leaves me only my start land.
- **Counts are exact evidence, both of them.** The engine reports the opponent's territory count every turn, so a hypothesis whose count after the move differs is removed (checked on every hypothesis every frame; 0 mismatches in about 9 million hypotheses over live matches). My count is checked as well: the engine's count for me after my move, minus what this hypothesis' move takes from its view of my land, must equal the count now reported. That needs the hypothesis' view of my land to add up to the engine's count to begin with (`my_exact`); a hypothesis whose view does not is not judged by it, since the cells it would take cannot be trusted. It is applied after my own captures too (an earlier version skipped those turns). When the opponent kills me my whole territory is gone and my count is the start size for every hypothesis, so there is nothing to tell them apart by. Counts are also what force the bookkeeping to be exact. The cell sets are a lower bound for the opponent's land: its count includes land I have never seen, so for a hypothesis born from a guess its cell set is smaller than its count.
- **K never rules a hypothesis out.** `move_probabilities` keeps every legal move above zero; only the rules and observations remove hypotheses. K is the abstract `MotionModel`; the default is `FeatureMotionModel`, a weighted sum of seven labelled features (F1 own trail, F2 my trail, F3 capture, F4 going home, F5 chasing my trail in view, F6 momentum, F7 exploring), weights in `MotionWeights`. They were fitted by hand to the one-step log-likelihood of the real opponent's move (true state as the hypothesis): enemy state machine -1.35 uniform -> -0.87, basic -1.38 -> -1.23, random -1.38 -> -1.42 (a random walker has no tendencies to learn). `ConstantMotionModel` remains as the baseline.
- **The start state and the first look.** A match starts with one certain hypothesis (`belief.state.start_state`): the opponent on its respawn, its 3x3 start land, no trail, count 9; my own land is my 3x3 start land, so every hypothesis starts with the right view of both territories; and I am on my respawn with count 9. The world state is seeded to match: `AgentRuntime` marks both start lands in memory at once, the opponent's too although it is out of view, so memory and the belief agree about it from the first frame (and a guessed history placed on that land is correctly 'at home'). Measured against not seeding, over whole matches the mass on the exactly right hidden state went from 0.046 to 0.073 against the random walker and 0.021 to 0.024 against the basic one, and within 0.003 elsewhere except hunter-as-seat-1 with deliberate deaths, which went from 0.438 to 0.427. Seeding and the belief's start state apply only if the first observation looks like a start (`is_fresh_start`: on my respawn, no trail, both counts 9); a controller switched on mid-match gets neither, and the belief is built from that observation. At my first observation the opponent has made no move if I move first and exactly one if it does, and the API does not say which seat I have. The seat is inferred from the respawn side (the default config gives the first mover the left respawn, the same kind of assumption as `MatchRules.mirrored`) and held as a belief, not a certainty: `seat_confidence=0.95`, so the expected order gets 0.95 and the other 0.05 (`first_update_stay` overrides it). Measured against the old fixed 50/50 hedge over 300-turn matches: probability on the true cell 0.0038 -> 0.0069 and exact-state mass 0.0024 -> 0.0048 (enemy, seat 0); small gains in the other cases.
- **Added to method1.0 because of what the live checks found:**
  - *Respawn transition for the residual.* A death teleports the opponent to its respawn cell, which a step-by-step walk can never reach; without it the true position got probability zero (33 turns in one match). A share of the residual (`residual_respawn_rate`, 2%) moves to the respawn cell each turn.
  - *Rehydration.* It runs at the end of each frame (in `settle.rebase`), so the states are final when the next frame starts; a frame that rebuilt the belief is left plain until the next one. After enough unseen turns the exact history has been pruned away and one unexplained capture then deletes every remaining hypothesis (all 7,456 children contradicted at once in one match). When hypotheses hold under 20% of the belief, half the residual is turned back into hypotheses with minimal guessed histories (`rehydrate_below`, `rehydrate_fraction`); wrong guesses are removed by the next evidence and cost nothing, because the rest of the residual keeps each position's probability.
  - *Contradiction means exactly zero.* The floor for rebuilding is `1e-300`, not `1e-12`: a purposeful opponent is a 1e-22 event for a random walk, which is still a valid posterior once normalised. Rebuilding (from the sighting, or a reachable diamond if unseen) happens only when no mass is left.
- **Limits.** With K constant the belief models the opponent as a random walker, so an opponent that walks purposefully while out of view is very improbable to it; position probability is then diffuse rather than wrong. History-level evidence cannot be applied to the residual, which is why rehydration exists. Against an opponent that captures my land with a loop the belief cannot follow (the scripted raider in `tests/raider.py` walks a 60-step rectangle), every child is cut by the opponent's count, the belief falls back on the residual and rehydrated guesses, and those guesses cannot know which of my cells were lost; they are marked inexact and not judged by my count until my land is exact again. Before that first loss every hypothesis' view of my land matched the engine's count in every frame.

Results over whole matches against the real engine (`python -m candidate.runtime_check`, 400 turns each):

| Scenario | True position ever at zero probability | Rebuilds | Probability on the true cell when in view | Belief update (mean) |
| --- | --- | --- | --- | --- |
| hunter, close, either side | never | 0 | 1.000 | 5 ms |
| basic, close | never (was 33 turns before the fixes) | 0 | 1.000 | 43 ms |
| random, close | never | 0 | 1.000 | 29 ms |
| enemy, close or far | never | 0 | not in view | 18 to 28 ms |
| hunter, far | never (15 rebuilds before the fixes) | 0 | 1.000 | 14 ms |

The belief update is the largest single cost per turn and grows with the number of hypotheses. Profiling showed the first version spent 80% of its time hashing territory differences one at a time; hashing them in one vectorised call made it about three times faster, and rewriting only the differences in use (step 0 and rebase) roughly halved it again (basic-close 43 to 16 ms, random-close 29 to 11 ms, hunter-close 4.6 to 2.3 ms mean).

## Run-time layer: what the live checks showed

`python -m candidate.runtime_check` plays the candidate in real matches and compares everything it believed with the engine's true state (see `CODEBASE_INDEX.md`, "Running and checking"). Results from the runs made while building it:

- **Translation.** Over thousands of observations, position, trail, territory counts, opponent sightings and every visible cell matched the engine exactly, except cells under an agent (see the table row above), which are flagged instead of guessed silently.
- **Inference.** Own deaths plus opponent kills inferred by the runtime equalled the engine's death counters exactly (46 of 46 in a 600-turn match with deliberate deaths against a hunter; 25 of 25 as player 1). Kills of the opponent equalled the engine's kill counter (33 of 33 and 20 of 20).
- **Prediction.** Every move's result was predicted correctly before it was made (600 of 600), the next position and trail matched every turn that was not an opponent kill, and the size of every capture was predicted exactly (31 of 31).
- **Memory.** The remembered map of my own territory was exact on every turn, including after deaths and captures that happened out of view.
- **Bug found by the tests, fixed.** `Observation.last_result` was always `None` (built before the previous turn was reconciled). The framework does not report it, so a strategy relying on it would have seen nothing.
- **Latency.** Adapter about 0.9 ms, runtime bookkeeping about 0.03 ms, probe policy about 0.3 ms, total about 2 to 3 ms per turn. The largest single cost is the framework's own move resolution, up to about 15 ms on a capture turn, which we do not control.
- **Not observable, so not fixed.** Land the opponent captures out of view is only noticed as a drop in my territory count (`TERRITORY_LOST`); which cells were lost is unknown to memory until they are seen again (a hypothesis that simulated the capture does know, see the belief section). A trail laid across a cell while it was out of view is likewise unknown until seen.

## Move rules as implemented

`MoveResolver` is the single place the board changes during a turn, so the rules below are the engine's, reproduced and checked rather than assumed. Details that are easy to get wrong:

- **Pre-read destination.** The destination's territory owner is read before any capture or kill and is what decides whether the cell is appended to the mover's trail. A mover who steps onto its own land never gets a trail cell there, even if a kill happens on the same step.
- **Capture rule.** Only the capturer's territory blocks the flood from the edge, and the trail counts as a wall only once converted. Cells enclosed *before* the trail closed are not captured, so an enemy island inside an old ring of mine stays theirs. Enemy territory inside the new loop flips. An enemy trail or agent inside the loop stays where it is (an enemy trail then lies on my land).
- **Capture and kill on one step.** Capture runs first, then the victim is respawned, and respawn gives the victim its starting land back from whoever holds it now. A kill has no result code; the step reports CAPTURED or NORMAL plus a separate killed flag.
- **Respawn.** Clears the trail and all territory, restores the starting land (taking those cells from the other player), and places the agent on its respawn cell, or the alternate cell if the other agent stands there. The alternate is a constant per side: the nearest other start cell, ties to the smaller `(y, x)`.
- **Invariants that held in every real state checked** (about 60,000 random moves plus full matches): the trail's last cell is the agent's cell, no trail lies on its owner's territory, trails are simple adjacent paths, and territories are disjoint. An agent can stand on enemy land with an empty trail, and can respawn onto a cell that carries the other player's trail.
- **Not modelled.** Turn counts, kill and death counts, and forfeits. The rules operate on the board only; `WorldState` has no turns-left.
- `BasicStateMachine` is deliberately weak and demonstrates only two states and `MoveResult` handling.
- `EnemyStateMachine` is the benchmark opponent. It uses `ATTACK`, `REPOSITION`, `OUT`, `ACROSS`, and `BACK` phases to make small rectangular captures inside a 7 x 7 safety grid around its head, and it attacks only when the opponent's trail is visible on its own territory.
- The framework has broad unit coverage for API types, domain state, visibility, rules, engine behavior, helpers, GUI snapshot handling, and the reference opponent.
