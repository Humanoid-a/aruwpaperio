# Codebase Index

This is a task-oriented map of the code. For which document to trust, see [`INDEX.md`](INDEX.md); for game facts, lifecycle details, and the method1.0 reconciliation, see [`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md).

## Candidate-owned code

| File | Responsibility |
| --- | --- |
| [`candidate/candidate_controller.py`](../candidate_controller.py) | Submission entry point, a thin wrapper that builds `agent.controller.Controller` (the GUI's "Candidate Controller"). It plugs in `agent.q.strategy.QStrategy` (the five action-states with the stand-in for Q); pass another strategy object, for example `ProbePolicy(seed=1)`, to replace it. |
| [`candidate/runtime_check.py`](../runtime_check.py) | Headless script (`python -m candidate.runtime_check`) that plays the candidate in a real match and scores the run-time layer against the engine's true state. See "Running and checking" below. |
| [`candidate/examples/basic_state_machine.py`](../examples/basic_state_machine.py) | Minimal two-state example showing persistent state, transitions, and `MoveResult` handling. |
| [`candidate/examples/random_state_machine.py`](../examples/random_state_machine.py) | Stateless random baseline among mechanically valid moves. |

Everything we add lives under `candidate/`: strategy modules in `candidate/agent/`, tests in `candidate/tests/`, notes in `candidate/docs/`. Nothing outside `candidate/` is modified (see [`INDEX.md`](INDEX.md)).

## Our code under `candidate/agent/`

[`method1.0.pdf`](../../method1.0.pdf) sketches an `agent/` package plus offline tooling. Per [`CANDIDATE_GUIDE.md`](../../CANDIDATE_GUIDE.md) and the working rules, all of it goes under `candidate/`. Folder names are the spec's suggestions, and the order of work follows the user's instructions rather than the spec's build order. Status: **built** or *planned*.

### Built: data structures, geometry, and move rules

| File | Responsibility |
| --- | --- |
| [`agent/config.py`](../agent/config.py) | Board and window constants (50 x 50, 11 x 11, 9 starting cells). Mirrors `game-config.properties`; the adapter must check the framework's values against it. |
| [`agent/data/types.py`](../agent/data/types.py) | Our own `Direction` (N, E, S, W = 0..3) and `MoveResult` (`NORMAL`, `CAPTURED`, `DIED`, `INVALID`). |
| [`agent/data/cells.py`](../agent/data/cells.py) | Cell codes (-1 unknown, 0 free, 1 my territory, 2 my trail, 3 opp territory, 4 opp trail), the two-layer cell model, `combine()` (lossy single-code view), `layer_bitboards()`, and layer validation. |
| [`agent/data/observation.py`](../agent/data/observation.py) | `Observation`: one turn's input, validated on construction, arrays read-only. `trail_hidden` lists the cells under an agent, whose trail layer is a guess because the framework hides it. |
| [`agent/data/world.py`](../agent/data/world.py) | `WorldState`: the true board (two layers, positions, ordered trails, counts) with `problems()` / `validate()`, `copy()`, and `bitboards()`. |
| [`agent/data/memory.py`](../agent/data/memory.py) | `MemoryMap`: last-seen layers plus a last-seen turn per cell; `update(obs)` copies the window (keeping what it knew about the trail under my own agent). `reset_my_land`, `reset_opponent_land` and `apply_capture` apply what the rules guarantee about cells that are out of view. |
| [`agent/geometry/grid.py`](../agent/geometry/grid.py) | Flat indexing `c = 50*y + x`, N/E/S/W offsets, `step()` that does not wrap across row ends, `neighbors()`, `manhattan()`. |
| [`agent/geometry/bitboard.py`](../agent/geometry/bitboard.py) | Row-word bitboards: one int64 word per row, bit `x` of word `y` is cell `(y, x)`; `(50,)` per board, `(N, 50)` per batch. Conversion to and from masks and cell lists, cell set/clear/test, `complement()`, `popcount()`, `is_valid()`. |
| [`agent/geometry/fill.py`](../agent/geometry/fill.py) | `dilate()`, `flood()`, `holes()` (regions a set fully surrounds), and `capture_region()` (the engine's capture rule, including "already-enclosed regions are not captured"). All work on a board or a batch. |
| [`agent/geometry/distance.py`](../agent/geometry/distance.py) | `bfs_distance()` by dilation, `shortest_path()` (deterministic ties, works from a blocked start such as an exposed head), and `taxicab_distance()` (separable two-sweep transform, batched over candidate paths; replaces SciPy). Unreachable is 9999. |
| [`agent/geometry/zobrist.py`](../agent/geometry/zobrist.py) | Seeded 64-bit keys per (role, cell); `hash_board()` by byte-table gathers (batched), `hash_state()`, incremental XOR updates. |
| [`agent/rules/match.py`](../agent/rules/match.py) | `MatchRules` / `RoleRules`: each side's respawn cell, starting-territory bitboard, and the alternate respawn cell used when the other agent stands on it. `mirrored()` assumes the opponent respawns at the left-right mirror. `initial_world()`. |
| [`agent/rules/reference.py`](../agent/rules/reference.py) | `apply_move(world, rules, mover, direction)` on a dense `WorldState`: slow, plain-Python mirror of the engine's move resolution, with its own BFS. Returns the new world (ordered trails and counts included), the `MoveResult`, and whether the mover cut the other's trail. |
| [`agent/rules/transition.py`](../agent/rules/transition.py) | `BitWorld` (N boards as bitboards plus positions), `classify_moves()` (all four moves per board with bit tests only), and `step()` (one move per board for a whole batch, flooding only the boards that capture). |
| [`agent/adapter.py`](../agent/adapter.py) | The only module here that imports the framework. Converts the framework's `GameApi` into our `Observation` (`observation_from_game`), maps `Direction` / `MoveResult`, converts cells, `send_move`, a framework-only `fallback_move`, and `check_board`. Stateless. |
| [`agent/prediction.py`](../agent/prediction.py) | `predict_move()`: what the engine will do with a move, from my own point of view (result, kill, new position and trail, expected capture size). Every move is predicted before it is made and compared with the engine's answer. |
| [`agent/runtime.py`](../agent/runtime.py) | `AgentRuntime`, the API a strategy talks to: `begin_turn(game) -> TurnView`, `move(game, direction) -> MoveResult`. Handles retries after INVALID, new-match detection, memory upkeep across deaths and captures, event inference, and prediction checks. `TurnView` carries the observation, memory, events, legal and safe moves, `home_path`, `predict()`, and the opponent's last sighting. |
| [`agent/diagnostics.py`](../agent/diagnostics.py) | `Diagnostics`: counters, timings, anomalies (error or info), the per-turn JSONL log, progress lines and the final report. Optional; the runtime works without any output attached. |
| [`agent/probe.py`](../agent/probe.py) | `ProbePolicy`: a deliberately simple loop-maker (out, across, back by the shortest route) that gives the runtime captures to check. Optionally steps onto adjacent opponent trails and deliberately dies every Nth excursion, to exercise kill and death inference. Not a strategy; no longer the GUI default. |
| [`agent/q/config.py`](../agent/q/config.py), [`types.py`](../agent/q/types.py) | Every constant of the Q model and the states in one place (`QConfig`, `Q`), and the `State` enum in the order of the network's five outputs (EXPAND, RETURN, HUNT, TRACK, PATROL). |
| [`agent/q/votes.py`](../agent/q/votes.py), [`snapshot.py`](../agent/q/snapshot.py) | What the hypotheses say per cell (enemy land, enemy trail, my land; the trail sum in chunks, only non-zero row words) and the per-frame `Snapshot`: possibility mask, enemy arrival times, D1-D8 with the raw D5/D7 kept for labels. D5/D6 are the best over the 8 tie-break orders of equally short paths, so they do not depend on the board's orientation. |
| [`agent/q/features.py`](../agent/q/features.py), [`grid.py`](../agent/q/grid.py), [`confidence.py`](../agent/q/confidence.py), [`inputs.py`](../agent/q/inputs.py) | Q's three inputs: the 22 scalars, the 7x21x21 egocentric grid (plus `rotate()` for augmentation), the 8-channel confidence board pooled to 8x10x10 (value and confidence kept apart; reach region; the residual's coin flip), and `build_inputs`. All pure. |
| [`agent/q/maps.py`](../agent/q/maps.py), [`paths.py`](../agent/q/paths.py), [`frame.py`](../agent/q/frame.py), [`exits.py`](../agent/q/exits.py) | Box filter and neighbour shifts; path following, capture gain (the rival's cells counting double) and `ORDERS`; the per-frame bundle `Frame` with the expedition's remembered direction (`Hints`); PATROL's exit score (room, safety, information, walk). |
| [`agent/q/movers.py`](../agent/q/movers.py), [`chooser.py`](../agent/q/chooser.py), [`safety.py`](../agent/q/safety.py) | The five states' own logic and validity masks (RETURN scores its direct and sideways-then-home paths by gain and margin); the chooser (death -> PATROL, thin margin -> RETURN, masks, switching cost h, larger RETURN->EXPAND cost); the safety look-ahead `margin_after`. |
| [`agent/q/heuristic.py`](../agent/q/heuristic.py), [`strategy.py`](../agent/q/strategy.py) | `HeuristicQ`, the hand-set stand-in for Q (same interface the NumPy network will have); `QStrategy`, the `Strategy` that ties it together: snapshot, inputs, values, masks, hard rules, the chosen state's move, a guard on the final move, a `FrameRecord` of the inputs exactly as Q saw them. |
| [`agent/controller.py`](../agent/controller.py) | `Controller`: one turn is `begin_turn` -> `strategy.choose` -> `move`. Never raises (an exception in `take_turn` would forfeit the turn): failures are recorded and answered with a move, preferring the way home. Defines the `Strategy` interface. |
| [`agent/data/territory.py`](../agent/data/territory.py) | `TerritoryDiffs`: a player's land as a small, deduplicated difference from the cells memory believes are theirs (`base & ~removed[id] | added[id]`). Used for both lands, the opponent's and mine. |
| [`agent/data/hypotheses.py`](../agent/data/hypotheses.py) | `HypothesisStore`, one hypothesis per row of parallel arrays: the opponent's position, last direction, trail and its length, its land (`opp_land_id`) and its size (`opp_count`), **my land under this hypothesis (`my_land_id`)**, event flags, weight, hash. `merge_key` and `problems` (every invariant). |
| [`agent/data/residual.py`](../agent/data/residual.py) | `ResidualGrid`: probability whose history has been thrown away; `spread()` moves it one step, `teleport()` sends part of it to the respawn cell. |
| [`agent/data/belief_view.py`](../agent/data/belief_view.py) | What the belief publishes each turn: `BeliefView` (position probabilities, support, entropy, trail marginal, exposure, count distribution), `BeliefSummary` (what is stored and what the frame did) and `BeliefEvent`. |
| [`agent/belief/config.py`](../agent/belief/config.py) | `BeliefConfig` and the **per-position budget**: `BudgetPolicy` (abstract) and `ProportionalBudget(total, min_keep, max_keep)`, which decides how many hypotheses each position may keep. |
| [`agent/belief/motion.py`](../agent/belief/motion.py) | The opponent motion model K: abstract `MotionModel.score(SuccessorBatch) -> (n, 4)`, `FeatureMotionModel` (the default: seven labelled features F1-F7 weighted by `MotionWeights`), the baseline `ConstantMotionModel` (every move scores the same; prints a description of each batch), and `move_probabilities()` (softmax over legal moves mixed with the floor epsilon, so no legal move is ever impossible). |
| [`agent/belief/state.py`](../agent/belief/state.py) | The data the filter owns (`BeliefState`: hypotheses, residual, both difference tables and their bases, `MyState`, flags) and the **correct starting state**: `start_state(rules)` (one certain hypothesis, both lands equal to the start lands) and `is_fresh_start(rules, obs)`. Also `BeliefStats` and `invalidate`. |
| [`agent/belief/land.py`](../agent/belief/land.py) | `rewrite_opp_land` / `rewrite_my_land`: change a land under every hypothesis at once and re-express it against a new base, rewriting only the differences in use. |
| [`agent/belief/generate.py`](../agent/belief/generate.py) | **Step 1.** `classify()` resolves all four moves of every hypothesis with bit tests (own-trail death, cut of my trail, loop capture and its size, my land taken), nothing copied; `build_children()` builds only the chosen (hypothesis, move) pairs; `unmoved_children()` for the first look; `advance_residual()`. `Situation` is what is true while the opponent moves. |
| [`agent/belief/cut.py`](../agent/belief/cut.py) | **Step 2.** What the rules and the observation make impossible: `cut_moves` (position seen or not, whether I was killed), `cut_children` (the opponent's count, the visible window, my count), `gate_residual`. Evidence only ever removes. |
| [`agent/belief/weigh.py`](../agent/belief/weigh.py) | **Step 3.** `assign_weights`: runs K for the parents that still have a child, softmax over *all* their legal moves, child weight = parent x P(move). `first_look_stay`: the prior on who moves first. |
| [`agent/belief/settle.py`](../agent/belief/settle.py) | **Step 4.** `settle` (merge duplicates, normalise, prune each position to its budget into the residual), `rebase` (re-express both lands on the new memory; rebuild if everything was contradicted), `rehydrate`. |
| [`agent/belief/after_move.py`](../agent/belief/after_move.py) | **Step 5, at the end of the frame.** `MyMove` and `apply_my_move`: my capture, my death or my cut of its trail applied to every hypothesis; leaves `MyState` where my move put me. Nothing is touched after an ordinary step. |
| [`agent/belief/reseed.py`](../agent/belief/reseed.py) | `reseed`: rebuild the belief from one observation when everything was contradicted. |
| [`agent/belief/publish.py`](../agent/belief/publish.py) | `publish` / `summarise`: the read-only `BeliefView` and its `BeliefSummary`. |
| [`agent/belief/filter.py`](../agent/belief/filter.py) | `BeliefFilter`, which only calls the steps in order: `begin_match(rules, obs)`, then each frame `observe(obs, memory, killed_by_opponent)` (steps 1-4, publishes `view`) and, once the engine has accepted the agent's move, `after_my_move(move)` (step 5). |

**A frame of the belief** (what the modules above do, in order):

```
begin_match(rules, obs)         once: one certain hypothesis, both lands = the start lands, memory seeded with them
--- each of my turns ---
observe(obs, memory, killed)    1. generate  every move of every hypothesis, resolved by the game rules
                                2. cut       what the rules and this observation make impossible
                                3. weigh     K: P(move | hypothesis); child weight = parent weight x P
                                4. settle    merge, normalise, prune into the residual, re-base on memory, rehydrate
   ... the agent chooses and makes its move ...
after_my_move(move)             5. apply my move to every hypothesis (end of the frame)
```

The hypotheses always describe the world right after my latest move and before the opponent's reply, which is the world the opponent moves in; nothing needs applying at the start of a frame.

Dependency direction: `config` <- `geometry` <- `data` <- `rules`, with `data/types.py` importing nothing. `data/__init__.py` is deliberately empty so importing `types` never pulls in the rest.

Speed (this machine, NumPy 2.5; indicative): `step()` takes about 1 µs per board in a batch of 1,000+ when few boards capture, and 13 µs per board when every board closes a loop. The reference takes about 10 µs for an ordinary move and 4 ms for a capture. A single-board `step()` costs about 0.03 ms (NumPy call overhead), 0.26 ms if it captures, so batch where possible.

### Planned

| Path (under `candidate/`) | Responsibility |
| --- | --- |
| `agent/data/` (rest) | `Snapshot` (distance maps, safety margin, kill windows), `Plan`, candidate batch, and log record types. |
| `agent/fsm/` | States (SAFE, EXPAND, RETURN, HUNT), the declarative transition table with named guards, and the final safety filter. |
| `agent/planning/` | Candidate plan generation, plan features, and Q scoring with hysteresis. |
| `agent/models/` | V, Q, and K models. |
| `offline/` | Self-play, dataset assembly, training, and evaluation. |

Only `adapter.py`, `candidate_controller.py` and `offline/selfplay.py` should import framework internals, so every other module can be tested without the framework. Tests may import the framework, because they are the oracle.

### Our tests (`candidate/tests/`)

Run with the project interpreter from the repository root; `-p no:cacheprovider` and `PYTHONDONTWRITEBYTECODE=1` keep a run from writing outside `candidate/`.

| File | What it pins down |
| --- | --- |
| `test_grid.py`, `test_bitboard.py` | Flat-index and bit layout, no wrap across rows, no stray bits past column 49, round trips, `popcount`, `complement` versus bare `~`. |
| `test_cells.py` | Code numbering, the two-layer model, lossy `combine()`, layer validation. |
| `test_observation.py` | Window clipping, head-last trail, field consistency, rejected inputs, read-only arrays. |
| `test_world.py` | Every `WorldState` invariant, one failing case each. |
| `test_memory.py` | Update semantics, stamps, staleness, bitboards, invariant checks. |
| `test_engine_oracle.py` (with `engine_support.py`, `builders.py`) | Real engine matches converted to `WorldState` from both perspectives and validated, bitboard counts equal the engine's counts, a scripted match that puts a trail on the other player's land, and `Observation` / `MemoryMap` compared with the true board. |
| `test_fill.py`, `test_distance.py`, `test_zobrist.py` | Flood fill, holes, and capture region against an independent plain BFS on random boards; hand-built capture cases (open loop, pre-existing hole, sealed channel); distance maps and shortest paths against a reference BFS; hashes against brute-force XOR. |
| `test_rules_match.py` | Starting territory clipping, alternate respawn tie-breaks, mirrored opponent, overlap rejection, initial world. |
| `test_reference_rules.py` | About 25 hand-built scenarios (movement, invalid moves, death, capture with enemy land and trail inside, pre-existing holes, cuts, capture-and-cut, displaced respawns), each run through the reference, the fast rules, and the framework's real `MoveResolver`, which must all agree. |
| `test_transition.py` | `BitWorld` conversion and invariants, `classify_moves()` against what `step()` then does, batch independence, malformed input, edge columns, a loose speed check. |
| `test_rules_fuzz.py` (with `rules_oracle.py`, `conftest.py`) | About 6,000 moves of randomly played games on the real rules (24 games x 250 moves): reference and fast rules reproduce the real outcome every time, every real intermediate state satisfies the `WorldState` invariants, untaken moves also agree, and whole mixed batches match the reference. |
| `test_rules_engine_oracle.py` | Full engine matches with the players placed close together: after every real turn the board equals what the reference and fast rules predict from the previous board and the logged move, from either player's perspective. |
| `test_adapter.py` | Direction, result and cell mappings; observations built from the real `GameApi` on hand-made boards (window clipping, the opponent's trail guess, my own cell hidden, a respawn onto the opponent's trail); `fallback_move`; `check_board`. |
| `test_memory_events.py` | What my death, the opponent's death and a capture do to the memory map (including cells never seen), and the hidden-cell rule: keep what was known under my own agent, take the guess under the opponent. |
| `test_prediction.py` | Predictions for all four moves checked against the real engine on 350 random positions (over 1,000 checks), plus hand cases for capture size, death, displaced respawn, invalid moves and kills. |
| `test_runtime.py` | The runtime driven turn by turn on the real rules: matches, retries, a refused destination, a reset mid-session, forfeits, captures (including cells out of view), self-death, being killed, killing, land lost to the opponent, home path, safe and legal moves, last sighting. |
| `test_probe.py` | The probe policy: choosing does not advance the plan, loops are made and closed, exposed-without-a-plan goes home, adjacent opponent trails are cut, deliberate suicide, determinism. |
| `test_q_snapshot.py` | D1-D8 on hand-built boards (D7 = D1 - D4, D8 = D2 - D3, no trail gives no gain), the possibility mask and its probability floor, known enemy trails, the hypothesis votes (chunk size does not matter). |
| `test_q_inputs_basic.py` | The 22 scalars (order, scaling, clipping, one-hot state), invariant under rotating and mirroring the board; the grid (channels, whole trail in the crop, off-board cells zero) and that it rotates with the board. |
| `test_q_confidence.py` | The brief's acceptance checks for the confidence map: certain belief gives confidence 1, all residual gives enemy-trail confidence 0 over the reach region, visible cells have confidence 1, position channels sum to 1 after pooling, pooling commutes with rotation; the vote arithmetic on small beliefs. |
| `test_q_states.py` | Each state's validity and move on small boards: PATROL never leaves my land and paces at its best exit, the exit terms one at a time, EXPAND straight on and never back, RETURN's candidate paths and scores (close enemy picks the direct path, far enemy a wider loop), HUNT, TRACK (visible and unseen). |
| `test_q_chooser.py` | Masks, the switching cost, the larger RETURN->EXPAND cost, the hard rules and their order, values within h never change the state. |
| `test_q_safety.py` | The look-ahead margin, including the property that the predicted margin is never better than what the next frame delivers, whatever the enemy does. |
| `test_q_strategy.py` | `QStrategy` through its interface: hard rules, the look-ahead, retries after a refused move, the guard, hints, the recorded inputs, a masked state is never taken, whole runs through the real controller. |
| `test_q_e2e.py` | Whole engine matches with the stand-in: clean and without deaths against every opponent in both seats, the look-ahead's measured effect, expansion/capture/hunting happen, per-frame cost, and the confidence map checked against the engine's true board. |
| `test_controller.py` | The controller never raises (bad strategy, nonsense directions, a broken adapter) and still moves, the way-home fallback, report printed once, the per-turn log, the debug label, and the GUI registry's `CandidateController`. |
| `test_diagnostics.py` | Counting, timing statistics, anomaly echo cap, report text, one report per match, the log directory. |
| `test_runtime_e2e.py` | Whole engine matches through `runtime_check`: against the Enemy State Machine, a hunting opponent (as either player) with deliberate deaths, and Random / Basic opponents; plus the exact `CandidateController` the GUI builds playing a match, and a latency budget. |
| `test_belief_description.py` | The four `[belief state]` lines per turn: present on every turn, no cell list, off unless asked for, on in the GUI controller; the summary behind them agrees with the engine's counts, its frame numbers are per frame (they add up to the totals), and an empty belief is described without failing. |
| `test_belief_territory.py` | Both lands per hypothesis: `my_land_id` in the store and the merge key, the gate on my count (a hypothesis with the right view survives, one with an inexact view is not judged), guessed histories inheriting a view, and whole engine matches checking every hypothesis' count for both sides every frame, including against the scripted raider that captures my land. |
| `test_match_start.py` | The start of a match: memory knows both start lands (even the opponent's, out of view) and nothing invented elsewhere, agrees with the board's initial world, the belief starts certain in either seat, a second match starts fresh, and a controller started mid-match does not pretend it is the start. |
| `test_hypotheses.py` | `TerritoryDiffs` (id 0, deduplication, growth, content keys), every `HypothesisStore` invariant with a failing case each, `ResidualGrid` spreading against a slow reference (mass conserved, never into my cell, corners, parity, teleport), and the `BeliefView` queries. |
| `test_belief_motion.py` | `move_probabilities` (rows sum to 1, illegal moves zero, the floor, sharpness, overflow), the abstract model and the constant model, the budget policy (proportional, minimum, maximum, pluggable) and config validation. |
| `test_belief_generate.py` | Step 1 against `rules.transition` (itself checked against the engine) on 1,000+ children from real play, with the opponent's land expressed against a deliberately wrong base: positions, trails, both lands, counts, my losses, kill flags. Plus capture, death, displaced respawn, illegal moves, batches, and children that know their parent and start with no weight. |
| `test_belief_filter.py` | The filter step by step on small scenes: the start state and who moves first, position evidence, did-not-die versus killed-me, impossible counts, visible trail and land, the gate on my count, **K applied after the cuts (only to parents that still have a child; probabilities over all legal moves)**, rebuilds, pruning, merging, the residual, rehydration, a plugged-in model, normalisation, determinism, capacity. |
| `test_belief_runtime.py` | The belief inside the runtime on real rules: a belief every turn, the truth always possible while the opponent walks in and out of view, my kill collapsing it, the opponent's kill of me, a new match, diagnostics, a custom filter, a skipped turn. |
| `test_belief_after_move.py` | Step 5: my capture, my death and my cut of its trail applied to every hypothesis and both lands; where it leaves me and with what count; an ordinary step changes and copies nothing; and in the runtime it is called once per accepted move, **after** the move and not at the start of the frame, not for a refused move, and a failure inside it cannot make the move look failed. |
| `test_belief_display.py` | `BeliefView.top_positions` (ordering, ties, only possible cells), the printed line (turn, in view or not, `(x,y)` cells, small probabilities keep precision), one line per turn and none on a repeated call, the legend once per match, the setting, and ordering before the end-of-match report. |
| `test_belief_e2e.py` (with `belief_support.py`, `belief_rig.py`, `raider.py`) | Whole engine matches through `runtime_check`: the opponent's true position always had non-zero probability and was certain whenever in view, against a hunter (both sides, with deaths), a loop-making opponent (a scenario that once zeroed the truth), a wanderer and a distant careful one; plus a time budget. The helpers build belief states from known worlds, drive the filter like the runtime, and script an opponent that captures my land. |

## Running and checking

**In the GUI** (`python -m territorygame.main`, run from the repository root): click a player's controller label until it reads "Candidate Controller", then Reset and Start. The console that launched the GUI shows, on every one of the candidate's turns, a line with the five cells the opponent is most likely on and their probabilities (`[belief] turn   10 | out of view | (45,25) 2.4%  (48,27) 1.4% ...`, cells as `(x,y)`; a legend line precedes the first), a progress line every 250 turns, and a full report when the match ends (or when Reset starts a new one); a per-turn JSON-lines log is written to `candidate/runtime_logs/` (git-ignored). The player card shows the current action-state (`EXPAND`, `RETURN`, `HUNT`, `TRACK`, `PATROL`, with ` !death`, ` !safety` or ` !lookahead` when a hard rule decided) and, if any error was recorded, `| err N`. `python -m candidate.runtime_check --strategy q ...` plays the same controller headlessly against the engine and prints its statistics.

**Headless, scored against the engine's true state:**

```powershell
python -m candidate.runtime_check --opponent hunter --turns 600 --close --suicide-every 4
```

To change or silence the per-turn line, construct `CandidateController(show_belief=N)` (0 is off; the GUI registry uses the default of 5), or `Controller(..., show_belief=N)` in `agent/controller.py`. Four more `[belief state]` lines per turn describe the whole belief (`belief_description` in `agent/controller.py`, built from `BeliefView.summary`, a `BeliefSummary` computed once per frame in `belief.publish.summarise`): how many hypotheses and how much of the belief they hold against the residual; the opponent's count on the hypotheses against the engine's, trail exposure, and how complete their land cells are; the engine's count for me and how many views of my land the hypotheses hold and how much of the mass agrees with it; and what this frame did (successors considered, cut by its count, by what I can see, by my count, merged, pruned, rehydrated). `CandidateController(describe_belief=False)` turns them off (on by default in the GUI controller, off in `Controller`).

Options: `--opponent {enemy,basic,random,hunter}`, `--turns N`, `--player {0,1}`, `--close` (respawns 13 cells apart so the players meet), `--suicide-every N`, `--no-hunt`, `--seed`, `--log-dir DIR`. It prints the runtime report followed by a comparison with the engine: observations (position, trail, counts, window cells), the cells hidden under agents, inferred deaths and kills against the engine's own counters, memory accuracy turn by turn, and the belief: the probability it put on the opponent's true cell, whether that cell ever had zero probability, how often its most likely cell was right, the mass on the exactly right hidden state (cell, trail, count), hypotheses kept, and rebuilds. Exit status 0 means clean.

**What a clean report means:** zero errors; "move result" and "next position and trail" predictions all exact (apart from the ones explained by an opponent kill); inferred deaths and kills equal to the engine's; memory of my territory exact on every turn; the opponent's true position never given zero probability by the belief, and probability 1 whenever it is in view. Info-level items (such as the opponent capturing some of my land out of view) are normal.

## Candidate-facing API

| File | Responsibility |
| --- | --- |
| [`territorygame/api/agent_controller.py`](../../territorygame/api/agent_controller.py) | `take_turn` contract and optional GUI debug label. |
| [`territorygame/api/game_api.py`](../../territorygame/api/game_api.py) | All information/actions exposed to a controller. |
| [`territorygame/api/direction.py`](../../territorygame/api/direction.py) | Four cardinal actions. |
| [`territorygame/api/grid_position.py`](../../territorygame/api/grid_position.py) | Immutable absolute board coordinate. |
| [`territorygame/api/visible_cell.py`](../../territorygame/api/visible_cell.py) | Immutable visible cell with independent occupant and territory layers. |
| [`territorygame/api/occupant_view.py`](../../territorygame/api/occupant_view.py) | Relative agent/trail occupant values. |
| [`territorygame/api/territory_view.py`](../../territorygame/api/territory_view.py) | Relative territory ownership values. |
| [`territorygame/api/move_result.py`](../../territorygame/api/move_result.py) | `MOVED`, `CAPTURED`, `DIED`, and `INVALID`. |

## Helpers available to candidate code

| File | Responsibility |
| --- | --- |
| [`territorygame/helpers/movement_utils.py`](../../territorygame/helpers/movement_utils.py) | Position arithmetic, board bounds, mechanical validity, Manhattan distance, visible lookup, and random direction. |
| [`territorygame/helpers/observed_board.py`](../../territorygame/helpers/observed_board.py) | Last-observed cell memory for the full board; it does not infer unseen changes or record when a cell was seen, so method1.0's `MemoryMap` (cell codes plus last-seen turn) is a separate type. |

## Engine and authoritative state

| File | Responsibility |
| --- | --- |
| [`territorygame/engine/game_engine.py`](../../territorygame/engine/game_engine.py) | Match construction, background worker, start/pause/step/reset, observer publication, and full snapshots. |
| [`territorygame/engine/turn_manager.py`](../../territorygame/engine/turn_manager.py) | Sequential scheduling, invalid retries, exception/attempt forfeits, and active-player advancement. |
| [`territorygame/engine/game_api_impl.py`](../../territorygame/engine/game_api_impl.py) | Player-scoped facade and one-successful-move-per-turn guard. |
| [`territorygame/engine/game_snapshot.py`](../../territorygame/engine/game_snapshot.py) | Immutable full-state observer DTOs used by the GUI and suitable for framework-side logging. |
| [`territorygame/engine/game_observer.py`](../../territorygame/engine/game_observer.py) | Observer interface called after each turn. |
| [`territorygame/domain/game_state.py`](../../territorygame/domain/game_state.py) | Mutable authoritative match state, remaining turns, active player, and counts. |
| [`territorygame/domain/board.py`](../../territorygame/domain/board.py) | Territory/trail ownership grid and territory queries. |
| [`territorygame/domain/agent.py`](../../territorygame/domain/agent.py) | Position, respawn point, and ordered active trail. |
| [`territorygame/domain/player.py`](../../territorygame/domain/player.py) | Player identity, agent, and immutable starting-territory copy. |
| [`territorygame/domain/game_config.py`](../../territorygame/domain/game_config.py) | Configuration loading and validation. |

## Rules and visibility

| File | Responsibility |
| --- | --- |
| [`territorygame/rules/move_resolver.py`](../../territorygame/rules/move_resolver.py) | Exact validation, collision, move, capture, kill, trail, and turn-consumption order. |
| [`territorygame/rules/territory_resolver.py`](../../territorygame/rules/territory_resolver.py) | Trail conversion and edge flood-fill enclosure capture. |
| [`territorygame/rules/respawn_service.py`](../../territorygame/rules/respawn_service.py) | Trail/territory clearing, starting-territory restoration, and safe respawn placement. |
| [`territorygame/visibility/visibility_service.py`](../../territorygame/visibility/visibility_service.py) | Clipped view construction and internal-to-relative ownership translation. |

## Controllers, configuration, and UI

| File | Responsibility |
| --- | --- |
| [`territorygame/controller/enemy_state_machine.py`](../../territorygame/controller/enemy_state_machine.py) | Stronger reference opponent and benchmark behavior. |
| [`territorygame/controller/available_controllers.py`](../../territorygame/controller/available_controllers.py) | GUI controller registry/factories. |
| [`territorygame/resources/game-config.properties`](../../territorygame/resources/game-config.properties) | Board, visibility, turns, respawns, seeds, delay, and retry cap. |
| [`territorygame/gui/game_window.py`](../../territorygame/gui/game_window.py) | pygame controls, controller selection, engine wiring, and snapshot ingestion. |
| [`territorygame/gui/board_renderer.py`](../../territorygame/gui/board_renderer.py) | Snapshot rendering only; no game rules. |
| [`territorygame/gui/snapshot_history.py`](../../territorygame/gui/snapshot_history.py) | GUI history navigation. |
| [`territorygame/main.py`](../../territorygame/main.py) | GUI entry point. |

## Test map

The root `tests/` tree is framework-owned and read-only for us; our own tests go in `candidate/tests/`. The `tests/territorygame/` tree mirrors the production packages. High-value references are:

| Test | What it pins down |
| --- | --- |
| [`tests/territorygame/rules/test_move_resolver.py`](../../tests/territorygame/rules/test_move_resolver.py) | Invalid moves, self-death, cuts, captures, and move-order edge cases. |
| [`tests/territorygame/rules/test_territory_resolver.py`](../../tests/territorygame/rules/test_territory_resolver.py) | Enclosure and pre-existing-hole semantics. |
| [`tests/territorygame/rules/test_respawn_service.py`](../../tests/territorygame/rules/test_respawn_service.py) | Reset and occupied-respawn fallback. |
| [`tests/territorygame/engine/test_turn_manager.py`](../../tests/territorygame/engine/test_turn_manager.py) | Retry, forfeit, exception, and scheduling behavior. |
| [`tests/territorygame/engine/test_game_api_impl.py`](../../tests/territorygame/engine/test_game_api_impl.py) | Candidate facade and one-move guard. |
| [`tests/territorygame/visibility/test_visibility_service.py`](../../tests/territorygame/visibility/test_visibility_service.py) | Window clipping and relative view layers. |
| [`tests/territorygame/controller/test_enemy_state_machine.py`](../../tests/territorygame/controller/test_enemy_state_machine.py) | Reference opponent phases and tactical choices. |
| [`tests/candidate/examples/test_basic_state_machine.py`](../../tests/candidate/examples/test_basic_state_machine.py) | Minimal headless match pattern using `GameEngine` and an observer queue. |

Run all tests with:

```powershell
pytest
```

## Where to look for common tasks

Only the first few rows involve editing, and only inside `candidate/`. Everything under `territorygame/` and the root `tests/` is read-only for us.

| Goal | Start here |
| --- | --- |
| Implement the submission controller | `candidate/candidate_controller.py` as a thin wrapper, with the logic in `candidate/agent/` (see the planned layout above). |
| Add controller unit tests | `candidate/tests/`; use a small `GameApi` stub, or hand-built boards for pure functions, for deterministic state/action tests. |
| Understand why a move produced a result | `move_resolver.py`, then its tests. |
| Check a reference transition or flood fill against the engine | Drive `GameEngine` with random moves and compare boards and counts; `territory_resolver.py` and its tests define capture semantics. |
| Add remembered observations | `agent/data/memory.py` (`MemoryMap` with last-seen turns); `observed_board.py` was only a starting reference. |
| Check the live pipeline against the engine | `python -m candidate.runtime_check ...` (see "Running and checking"). |
| Build headless match experiments | `game_engine.py`, `game_observer.py`, and the basic FSM integration test; this is the basis for `offline/selfplay.py`. |
| Log authoritative trajectories | Consume `GameSnapshot` from a framework-side observer (self-play only); do not expose it through `GameApi`. |
| Know the default match parameters | `game-config.properties` (read-only for us): board size, turns, respawns, seeds. |
| Change how many hypotheses are kept per position | `BeliefConfig.budget` in `agent/belief/config.py`: `ProportionalBudget(total, min_keep, max_keep)`, or subclass `BudgetPolicy`. Pass the config to `BeliefFilter`, and the filter to `AgentRuntime(belief=...)`. |
| Plug in a real opponent motion model | Subclass `MotionModel.score(batch)` in `agent/belief/motion.py` (the batch carries the consequence flags of each move), or just change the numbers in `MotionWeights` / add a feature to `FeatureMotionModel.features` and pass it as `BeliefFilter(motion=...)`. |
| Make random experiments reproducible | Framework factories ignore the configured seeds and `EnemyStateMachine` hard-codes `random.Random(42)` with no seed argument, so this has to be handled in candidate code (for example a subclass that sets the random source). |
| See the controller in the GUI | `CandidateController` is already registered in the picker as "Candidate Controller" in `available_controllers.py` (read-only). |
