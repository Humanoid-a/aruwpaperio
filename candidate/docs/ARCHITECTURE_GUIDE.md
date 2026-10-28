# Architecture Guide

A guided tour of the candidate controller: what the approach is, how the code is organised, what each important part
does and why it was built that way, how it is verified, and what is still weak. Everything here lives under
`candidate/`; the framework (`territorygame/`) is never modified.

Contents

1. [The approach in one page](#1-the-approach-in-one-page)
2. [The structure of the code](#2-the-structure-of-the-code)
3. [One turn, end to end](#3-one-turn-end-to-end)
4. [The important parts](#4-the-important-parts)
5. [The training system](#5-the-training-system)
6. [The less important parts, briefly](#6-the-less-important-parts-briefly)
7. [How we know it works](#7-how-we-know-it-works)
8. [Results, limits and what comes next](#8-results-limits-and-what-comes-next)
9. [Questions to expect, with short answers](#9-questions-to-expect-with-short-answers)

---

## 1. The approach in one page

**The game.** Two agents on a 50 x 50 board, 2,000 moves each. Leaving your territory lays a trail; returning closes
it and captures the trail plus everything it encloses. Stepping on your own trail kills you; stepping on the
opponent's trail kills it. Death wipes your territory back to the 3 x 3 start. You see only an 11 x 11 window around
yourself, plus both territory counts. Most territory at the end wins.

**The core difficulty** is partial observability: the opponent is usually out of view, yet everything that matters
(is it about to cut my trail? is its trail exposed so I can kill it?) depends on where it is. And the game punishes
one mistake (a death) far more than it rewards any one good move.

**The design**, in one line:

```
observation -> memory -> belief over the hidden opponent -> snapshot of the situation
            -> Q scores six action-states -> chooser (hard safety rules can overrule Q) -> the state's own move
            -> look-ahead safety check and guard -> move
```

Five ideas carry the design:

1. **Know what you can know.** A run-time layer translates the framework's API into our own types once, remembers
   every cell it has seen (`MemoryMap`), and infers what the API never says directly (that I was killed, that the
   opponent captured my land out of view). It predicts the result of every move before making it and checks the
   prediction against the engine.
2. **Believe precisely about what you cannot see.** A particle-style belief filter keeps thousands of complete
   hypotheses about the hidden opponent (position, trail, territory, count) and updates them with the exact game
   rules. Territory counts reported by the API are used as hard evidence, which is very selective.
3. **An explicit state machine.** Six action-states (EXPAND, EXPAND2, RETURN, HUNT, TRACK TARGET, PATROL), each with its own
   deterministic logic and validity test. The states decide *how* to move; they never learn.
4. **A learned chooser, under hard safety rules.** A value function Q estimates, for each state, the score difference
   it leads to over the next 50 turns. The controller switches to the best valid state with a switching cost
   (hysteresis), but hard rules that Q cannot outvote handle safety: after a death go PATROL; with a thin safety
   margin go RETURN; and never make a move that would leave the worst-case margin negative.
5. **A closed training loop.** Every game (headless or in the GUI) records Q's inputs exactly as seen; labels come
   from what happened next; a PyTorch network is trained on them and plugged back in. An online loop repeats this by
   epochs against randomised opponents.

The main deliberate trade-off: **safety is worst-case and hard-coded, value is learned.** A learned model may decide
*which* reasonable thing to do; it may never decide to take an avoidable death.

---

## 2. The structure of the code

```
candidate/
  candidate_controller.py   the framework's entry point ("Candidate Controller" in the GUI); thin: builds the stack
  runtime_check.py          plays a headless match and scores everything we believed against the engine's truth
  agent/                    everything that runs during a match
    config.py                 board constants (50 x 50, 11 x 11 view, 9 start cells)
    adapter.py                the ONLY module that speaks framework types: GameApi -> Observation, Direction <-> ours
    prediction.py             what the engine will do with a move (result, kill, new trail, capture size)
    runtime.py                AgentRuntime: memory upkeep, event inference, retries, prediction checks, the belief
    controller.py             Controller: one turn = begin_turn -> strategy.choose -> move; never raises
    diagnostics.py            counters, timings, anomalies, per-turn log, end-of-match report
    probe.py                  a simple loop-making test strategy (exercises the runtime; not the real strategy)
    geometry/                 bitboards, flood fill, distances, Zobrist hashing           (pure, fast NumPy)
    data/                     our types: Observation, WorldState, MemoryMap, hypotheses, residual, BeliefView
    rules/                    the game rules re-implemented: reference (slow, clear) and batched bitboard version
    belief/                   the belief filter, one module per step of a frame
    behavior/                 the six action-states, the snapshot and world map they read, the chooser, the safety rules
    q/                        the Q model: its inputs, the stand-in and the network, recording, and QStrategy
  offline/                  training and simulation tools; never imported during a match
  tests/                    ~700 tests, many using the real engine as the oracle
  docs/                     this guide, the build plan and decisions (Q_BUILD_PLAN.md), the code index, game facts
  q_training/               recorded games, datasets, weights, online runs (git-ignored except the shipped networks)
```

**Size** (lines of Python): geometry 490, data 1,085, rules 520, belief 1,584, behavior 1,206, q 993, the top-level
agent modules 1,253, offline 2,000, tests about 9,000.

**Dependency direction** (each layer imports only the ones to its left):

```
config <- geometry <- data <- rules <- belief <- runtime <- behavior <- q <- controller / candidate_controller
                                                                               offline (tools, outside the match)
```

Two boundaries matter most:

- **Framework boundary.** Only `adapter.py` (and the entry point, `runtime_check.py` and `offline/`) touch framework
  types. Everything else works on our own arrays and enums, so it can be tested on hand-built boards without an engine.
- **Behaviour / Q boundary.** `behavior/` knows nothing of Q; `q/` imports `behavior/`, never the reverse. The states
  can be tested and played with any six numbers (the stand-in, the network, fixed numbers in a test).

---

## 3. One turn, end to end

The engine calls `CandidateController.take_turn(game)`; it delegates to `agent.controller.Controller.take_turn`:

1. **`runtime.begin_turn(game)` -> `TurnView`** (`agent/runtime.py`)
   - `adapter.observation_from_game(game)` builds an `Observation`: my position and trail, both counts, turns left,
     the visible window as two layers (territory, trail), the opponent if seen.
   - It notices a **retry** (the engine calls again after an INVALID move in the same turn) and a **new match**
     (more turns left than before: the GUI's Reset), and starts fresh if so.
   - It infers **events** since my last turn: `SELF_DIED`, `KILLED_BY_OPPONENT` (my trail vanished and I am at my
     respawn), `CAPTURED`, `KILLED_OPPONENT`, `TERRITORY_LOST` (my count fell without a death), `FORFEITED_TURN`.
   - It updates the **memory** (`MemoryMap`) with the window, and applies what the rules guarantee for cells out of
     view (a death resets my land; a capture adds the enclosed cells).
   - It runs the **belief**: `BeliefFilter.observe(obs, memory, killed)` (generate, cut, weigh, settle; section 4.3).
   - The `TurnView` carries all of it, plus derived facts computed on first use: legal moves, safe moves (legal and
     not onto my own trail), the shortest way home, `predict(direction)`.
2. **`strategy.choose(view)`** (`agent/q/strategy.py`, `QStrategy`)
   - `build_snapshot` (behavior): distance maps, the way home, D1..D8 and the safety margin.
   - `build_inputs` (q): the 23 scalars, the 7 x 21 x 21 grid, the 10 x 10 x 10 probability map.
   - `q_model.values(frame, inputs)`: six numbers (the stand-in, or the network).
   - `moves.valid_states(frame)`: which states make sense now (the mask).
   - `chooser.choose(...)`: hard rules, mask, switching cost -> the state.
   - `moves.move_for(state, frame)`: that state's own move.
   - the **look-ahead** (`safety.margin_after`) may replace it with RETURN's (or PATROL's) move; the **guard** makes
     sure the final move is legal and not onto my own trail.
   - a `FrameRecord` (and, if recording, a row in the game record).
3. **`runtime.move(game, direction)`**: predicts the result, sends the move, compares the engine's answer with the
   prediction (a disagreement is logged as an error), and on success applies my move to every hypothesis of the
   belief (`after_my_move`). INVALID means the engine will call again; the strategy is asked again with the refused
   cell excluded, and keeps its state.
4. **`strategy.on_result(...)`**: remembers the direction an expedition set out in (EXPAND keeps going that way, and
   EXPAND2's "sideways" is perpendicular to it).

If anything raises, `Controller` records it and answers with a fallback move (preferring the way home): an exception
inside `take_turn` would make the engine forfeit the turn.

---

## 4. The important parts

### 4.1 The adapter and the observation (`agent/adapter.py`, `agent/data/observation.py`)

The only translation point. Coordinates become flat indices `c = 50*y + x`; the window becomes two int8 layers
(territory and trail, because a trail can lie on the other player's land and one code per cell would lose that);
framework `MoveResult.MOVED` becomes our `NORMAL`. Two subtleties it handles:

- **Cells under an agent hide their trail.** The API reports the agent, not the trail beneath it. `trail_hidden`
  lists such cells: under my agent the memory keeps what it knew, under the opponent it guesses (it lays a trail
  wherever it is off its own land).
- **The API has no event channel.** I am never told I was killed; the runtime infers it (my trail vanished and I
  stand on my respawn), verified against the engine's death counters 46/46 and 25/25.

### 4.2 Memory (`agent/data/memory.py`)

`MemoryMap`: the last-seen territory and trail layer of every cell, and the turn it was last seen. Remembered cells
are a prior, never certain evidence. The runtime keeps it right across events the window cannot show: after my
death my land is exactly my start square; after my capture the enclosed cells are mine even out of view. My own
territory in memory was exact on every turn of every checked match.

### 4.3 The belief over the hidden opponent (`agent/belief/`, data in `agent/data/hypotheses.py`, `residual.py`, `territory.py`)

**What it represents.** A weighted set of *hypotheses*. Each is one complete guess about the opponent: its cell, last
direction, full trail, its territory and count, and also **my own territory as the opponent's actions would have left
it** (it may have captured my cells out of view). Probability whose detailed history was thrown away lives in a
per-cell **residual**, so pruning never changes the probability of any position.

**A frame** (`belief/filter.py` only calls the steps in order):

```
observe(obs, memory, killed)    1. generate  every move of every hypothesis, resolved with the real rules
                                2. cut       remove what the rules and this observation make impossible
                                3. weigh     K: probability of each move; child weight = parent x P(move)
                                4. settle    merge duplicates, normalise, prune into the residual, re-base, rehydrate
   ... my strategy chooses, the engine accepts my move ...
after_my_move(move)             5. apply my move to every hypothesis (my capture, my death, my kill of it)
```

- **Generate** (`generate.py`) classifies all four moves of every hypothesis with bit tests on bitboards (own-trail
  death, cutting my trail, closing a loop and the capture's size), then builds only the children that survive.
- **Cut** (`cut.py`) is the evidence, and it only ever removes:
  - if I see the opponent, only children on that cell survive; if I do not see it, none inside my window survive;
  - every visible cell must agree (a child that put a trail where I see none is gone);
  - **the opponent's territory count is reported every turn**: a child whose count differs is removed. This is
    very selective (it tells which captures happened, and how large);
  - my own count, after my move, must match what the child's capture would have taken from me;
  - whether I was killed (did a child cut my trail?).
- **Weigh** (`weigh.py`) runs the motion model K only for parents that still have a child, and normalises over *all*
  legal moves, not just the survivors: if the evidence ruled out three of a parent's moves, the remaining child is
  as unlikely as K says, not certain. K (`motion.py`) is `FeatureMotionModel`: a softmax over seven labelled features
  (own trail, my trail, capture, going home, chasing my trail, momentum, exploring), weights fitted by hand to real
  opponents' moves. K never makes a legal move impossible; only rules and evidence remove hypotheses. A deeper K
  (`neural.py`) replaces the weighted sum with a small neural network over 17 features computed for every hypothesis
  on its own (its distance home, the kill and safety margins both ways, whether it sees me, its gain if it went home...);
  it is trained every online epoch on the real moves of the enemy machines and of our own agent (`offline/k_train.py`),
  and the hand-set K remains the fallback.
- **Settle** (`settle.py`): merges identical hypotheses (Zobrist hash keys), normalises, and keeps a per-position
  budget (`ProportionalBudget`: about 2,000 in total, at least 1 and at most 1,500 per cell); the rest of each
  cell's mass goes to the residual. If hypotheses hold under 20% of the mass, part of the residual is turned back
  into hypotheses with guessed histories (**rehydration**), because one unexplained capture would otherwise contradict
  everything. A share (2%) of the residual moves to the respawn cell each turn (the opponent may have died out of
  view; a step-by-step walk can never reach the respawn).
- **Territory as differences** (`data/territory.py`): storing a 50-word land per hypothesis is wasteful when almost
  all agree with memory, so a hypothesis stores an id into a deduplicated table of `(added, removed)` differences from
  memory's view, re-expressed after every window update so the table stays small.

**What it publishes** (`publish.py` -> `BeliefView`): the position probability of every cell, the support, entropy,
the hypotheses' detail (weights, trails, lands) for the snapshot and the probability map, and a summary for the
console.

**Why a particle filter and not a heat map?** Because what matters is not only *where* the opponent is but *what it
has*: an exposed trail (a kill chance, and a threat to my trail), its territory (to explain count changes). Those
are properties of a history, which a per-cell probability cannot hold. The counts make histories cheap to check.

**Measured:** the opponent's true cell never had zero probability over whole matches against every opponent, and was
certain whenever in view; the true hidden state (cell, trail, count) typically holds a few percent of the mass while
out of view. The belief is the most expensive part of a turn (tens of ms with ~1,500 hypotheses).

**Its weak spot, measured** (`offline/motion_fit.py`, decision 30 of the build plan): out of view, against
EnemyStateMachine, the probability on the true cell averages about 0.004, ten times a uniform guess, but its mean
log-probability is no better than a uniform guess: the belief is often confidently wrong. Fitting K to 1.4 million
recorded opponent moves made K predict purposeful opponents' moves better, yet did not improve the belief against
them; the limit is calibration out of view, not K's one-step accuracy.

### 4.4 The snapshot (`agent/behavior/snapshot.py`)

Built once per frame, read by the states and by Q's inputs. Pure function of the `TurnView`. It first builds the world
map (the belief as per-cell probabilities, full resolution), then reads D1..D8 from it, keeping only the cells at
least this likely: the enemy's position >= 1e-4, its trail >= 0.5, land (mine or its) >= 0.5 (`BehaviorConfig`).

| | Meaning | How (on the world map, above the thresholds) |
| --- | --- | --- |
| D1 | enemy's distance to my trail | taxicab from the cells it is likely on to my nearest trail cell |
| D2 | my distance to the enemy's trail | shortest path avoiding my trail, to the nearest likely enemy trail cell |
| D3 | enemy's distance to its land | from its seen cell, else averaged over the likely cells (renormalised), to its likely land |
| D4 | my distance home | shortest path avoiding my trail, to my likely land |
| D5 | my gain if I return now | flood fill along that way home; its likely land counts double (it swings the score twice) |
| D6 | enemy's gain if it returns now | the same for the enemy: its likely trail, from its most likely cell |
| D7 | margin | D1 - D4: how many turns the enemy is late if it races for my trail while I go home |
| D8 | kill race | D2 - D3: negative means I can reach its trail before it gets home |

**The hard safety rules do not use these.** A threshold on the enemy's position drops its true cell on many turns
(measured out of view: 12% of turns below 1e-6, 28% below 1e-4, half the turns against the hunter), and a safety rule
must not be wrong exactly then. So the snapshot also keeps the worst case under its own names: `possible` (every
cell with any probability), `enemy_arrival`, `safety_d1` and `safety_margin` (the forced RETURN and the look-ahead use
these), and the way home over the land memory holds (`home_path`, which RETURN takes).

D5/D6 are the best gain over the eight tie-break orders of equally short paths, so they do not change if the board
is rotated (a test checks it). Home targets exclude the cell the visible enemy stands on (it can block me).

### 4.5 The six action-states (`agent/behavior/`)

Each state is one module with `valid(frame)` and `move(frame)`. Every state recomputes its target every frame, so it
changes its mind exactly when the best choice changes. Each has one simple job; *when* to do which is Q's question.
A capture is a loop: EXPAND goes out, EXPAND2 turns, RETURN closes.

| State | Its job | Valid when |
| --- | --- | --- |
| **EXPAND** (`expand.py`) | Go out: walk to the **nearest** exit of my land, step out on its roomiest side, then straight on in that direction. Never turns. | inside: an exit exists; outside: the cell straight ahead is free (not my land, not my trail) |
| **EXPAND2** (`expand2.py`) | The turn: sideways, perpendicular to the way out, on the side with more free room, and that side is kept until the trail is home. It makes the loop enclose area instead of a thin line. | I have a trail and can step to that side |
| **RETURN** (`return_home.py`) | Close the loop where it gains the most: every cell of my land it could close on (up to 8 steps beyond the nearest), the shortest path to each, scored by gain (enemy cells double) and the worst-case margin against the enemy; the highest gain with at least 1 turn to spare, else the safest. | I have a trail |
| **HUNT** (`hunt.py`) | Cut the enemy's trail before it is home. For each of my moves, the share of the belief whose trail I could then reach before that enemy reaches its land (per hypothesis: its own trail, cell and land); take the move with the best share. Stepping on the trail kills the enemy. | the best share is at least 40% (`hunt_min_chance`; measured: below it the true trail could almost never be cut) |
| **TRACK TARGET** (`track.py`) | Find the enemy: head for the cell whose 11 x 11 view would cover the most of its position probability on the world map, discounted by 0.97 per step of walking (no reach limit). | the enemy is out of view |
| **PATROL** (`patrol.py`) | Reposition inside my borders: walk to the best exit (`exits.py`) and pace there. The safe state; where a death sends me. | no trail out |

**PATROL's exit score** (`exits.py`) for every boundary cell of my land: room (free-cell density just outside) +
safety (the enemy's worst-case arrival time) + information (how stale the cells beyond are) - walking distance, each
scaled to 0..1 and weighted in the config. PATROL picks *where* to stand; EXPAND then leaves from the nearest edge.

**The world map** (`worldmap.py`, built once per frame into the snapshot) is the belief as per-cell probabilities over
the whole world: ownership and trail layers, confidence, positions. HUNT and TRACK read it, and Q's probability map is
it, pooled.

### 4.6 Choosing the state, and staying safe (`agent/behavior/chooser.py`, `safety.py`; `agent/q/strategy.py`)

In order (the first two are the brief's hard rules, Q cannot outvote them):

1. After a death -> **PATROL** at the respawn.
2. If I have a trail and the margin D7 <= 1 -> **RETURN**.
3. Remove the invalid states (a masked state is never chosen).
4. Switch to the best remaining state only if its value beats the current one by **h = 2 cells** (4 from RETURN back
   to EXPAND or EXPAND2, which lengthens the trail). Without this hysteresis, two near-equal values flip the state every frame
   and the agent dithers. If the current state became invalid, switch at once.

Then two more layers, both in `QStrategy`:

- **The look-ahead** (`safety.margin_after`), added after measuring that rule 2 acts too late: one step straight at an
  approaching enemy can cost three turns of margin (it closes one, my new trail cell is one nearer, my way home one
  longer), so a margin of 4 becomes 1 and then -2 before rule 2 fires. Measured: the stand-in died 19 times in 400
  turns against a hunting opponent. `margin_after` computes the worst-case margin the *next* frame would show if I made
  the move, counting the cells of my way home as exposed too (the enemy can block me by standing on them; this was a
  second bug found in play). Any non-RETURN move that would leave it below 0 is replaced by RETURN's move (or PATROL's
  without a trail). Deaths against the hunter: 19 -> 0. A move that kills the enemy is not automatically safe: the enemy
  reappears at once on its respawn and moves next, so a kill is scored from there, and kills are checked even for
  RETURN (a third bug found in play: a forced step home cut its trail right beside its respawn).
- **The guard**: the final move must be legal and not onto my own trail; if a state produced none, take the safe move
  that gets home soonest.

### 4.7 The Q model: what it predicts (`agent/q/`, `offline/dataset.py`)

**Q(state)** = the expected change in score difference (my cells - enemy cells) over my next 100 turns if I take that
state now, in cells. Concretely, the label for frame t is

```
y_t = (score_diff[t+100] - score_diff[t])  +  D5[t+100] * [D7[t+100] > 0]
```

and in online training (bootstrapping, the default) the second term is replaced by the current network's value of the
state taken at t+100, so a label looks about 200 turns ahead: a longer loop, which pays off later, is credited.

The second term credits a loop still open at the end of the window, but only if it was safe then (an open, unsafe
loop is worth nothing). Near the end of the game the window stops at the end: y = final_diff - score_diff[t]. Only the
state actually taken gets a gradient (the others' outcomes were not observed).

**Inputs** (all pure functions of the frame):

| Input | Shape | Content |
| --- | --- | --- |
| scalars (`features.py`) | 23 | trail length, both counts and their difference, D1..D8 (scaled, D7/D8 clipped to +-25), turns left, enemy visible, enemy trail known, turns since sighting, how spread the belief is, the current state (one-hot) |
| grid (`grid.py`) | 7 x 21 x 21 | egocentric crop around me: on board, visible now, my land, my whole trail, visible enemy land, trail and agent |
| probability map (`probmap.py`) | 10 x 10 x 10 | the whole belief as a full world state, pooled in 5 x 5 blocks: ownership (free, mine, enemy) and trail (none, mine, enemy) as probability layers, a confidence per group, the enemy's position probability, my position |

The probability map gives each cell two groups of three **mutually exclusive layers that sum to 1** (ownership: free,
mine, enemy; trail: none, my trail, enemy trail), so 0.45 mine / 0.45 enemy / 0.10 free reads as a coin flip between
two owners, not as free. A **confidence** per group (1 - entropy / log 3) survives pooling: a block half certainly mine
and half certainly free averages to the same layers as a block of 50/50 guesses, and only the confidence tells them
apart. Residual mass (no known history) follows memory by halves inside the region the enemy could have reached since
it was last seen, and memory outside it; my trail is exact.

**The network** (`network.py`, PyTorch, 79,478 parameters): three branches (grid: 4 convolutions -> 64; scalars:
dense 64; probability map: 3 convolutions -> 32) concatenated into a head (64 -> 6, one value per state). A switched-off branch outputs zeros of
its usual width, so the head's weights carry over when a branch is added (the brief's staged training).

**The stand-in** (`heuristic.py`, `HeuristicQ`): six hand-set formulas from the snapshot, playing a rectangle
(EXPAND and EXPAND2 share a value that grows with margin and shrinks with the trail, plus a bonus for the one whose leg
it is: out for the first 5 cells, then the turn; RETURN worth D5 minus a penalty when unsafe, so it wins once the loop
is worth closing; HUNT a fixed prize when the kill race is won; TRACK small, more when the enemy is unseen; PATROL
small). It plays before any network exists, collects the first
data, and is the baseline every network must beat. The controller uses it by default.

**`QStrategy`** (`strategy.py`) is where the two layers meet; also the brief's **exploration** (on 3% of frames a
random valid state is held for 1-8 frames, so Q learns about states it would not choose) and the hook for recording.

---

## 5. The training system

The recorder is in `agent/q/record.py`; everything else is in `candidate/offline/`.

```
play (headless engine, many processes) --record--> raw/*.npz + sidecar tapes
   --dataset.py--> labels, game weights --train.py--> network.pt --> next epoch plays with it (online.py)
```

- **Recording** (`q/record.py`, `GameRecorder`): one file per game, written by the controller itself in the GUI and
  headless alike: every frame's inputs *exactly as Q saw them* (the grid bit-packed, the probability map in float16),
  values, masks, the state before and after, forced/explored flags, the accepted move, and the label ingredients
  (counts, raw D5/D7). It never raises into the game. Headless games also store a **sidecar** with both players' move
  tapes: the engine is deterministic given the moves, so a game can be replayed exactly.
- **Stale data** (`q/version.py`, `offline/replay.py`): `inputs_version()` fingerprints the source of every module
  that decides the inputs, plus the config values they read. A record made by older code is replayed from its tapes
  to regenerate its inputs with the current code, instead of being thrown away.
- **Dataset** (`offline/dataset.py`): labels as in 4.7; every game weighs the same (a long game must not dominate);
  optional *bootstrapped* labels (the open-loop term replaced by a previous network's value, which credits slow
  payoffs such as TRACK's information).
- **Trainer** (`offline/train.py`):
  - squared error on the taken state only; Adam, learning rate 1e-3, batch 256, weight decay 1e-4;
  - split **by game**, never by frame (frames of one game are strongly correlated); early stopping on validation;
  - augmentation: random quarter turns and mirroring of the grid and the probability map together (the scalars are
    invariant by construction);
  - labels are divided by their standard deviation for training and the factor is folded back into the last layer,
    so the exported network outputs cells;
  - **staged** build from scratch: scalars alone, then + grid, then + probability map, each kept only if it beats a
    *control* trained for the same extra budget without it (a naive comparison favours the later stage just
    because it trained longer: this was a real false positive);
  - **finetune**: an existing network trained further, keeping the old weights if no epoch beats them.
- **Online loop** (`offline/online.py`): per epoch, games (60 at first, growing to 200) with the current best network,
  the *player* (unrandomised, 3% exploration, recorded; half of them with the *learner*), against randomised
  opponents; then the learner is fine-tuned on the last three epochs' games, with all three branches; then a *gate*:
  the learner plays 16 fixed real-belief games and replaces the player (`best.pt`) only if it does at least as well.
  The first player is the hand-set stand-in, so a network is only used once it beats it, game by game, by more
  than the noise (a paired comparison on the same games). Labels bootstrap from the player (a target network).
  Training games are played in many situations (`offline/scenarios.py`): far, close, very close and random spawns,
  games of 2000 down to 300 turns, and games where another agent plays our seat for a while before ours takes over.
  Resumable.
- **Most games skip the belief** (`offline/oracle.py`): in all but 20% of training games (`QConfig.belief_share`) our
  agents are handed the engine's true board as a certain belief, so the filter does not run: a game is 5-6 times
  faster and Q's inputs carry no belief noise (the probability map is then one-hot; the belief games teach the
  uncertain case). The catch: with the truth the agent also *plays* better informed and bolder than it can in a real
  match, so each epoch reports the belief games' results separately; evaluation always uses the real belief. With
  `--self-oracle-share` (0 by default), some belief games against our own agent give only that opponent the truth: one
  belief filter instead of two, but that opponent then wins nearly all of them. The costliest games are started first.
- **Reading a run** (`offline/report.py`): `curves` draws an online run's win rates (belief vs true-world games, per
  opponent), territory, kills, deaths, state shares and Q's and K's training; `benchmark` plays the stand-in and
  chosen checkpoints on the same fixed games with the real belief, the clean measure of what training did.
- Opponents: a third of the games each (by default) against `NoisyEnemy`, against our own agent with randomness,
  and against `enemy_copy`, our copy of EnemyStateMachine with no randomness at all (the benchmark opponent
  itself, move for move). The first two differ from game to game through:
  - `BehaviorConfig.randomness` (`behavior/style.py`): our own agent with per-game varied constants, exit
    preferences and occasional sideways jogs, **safety only ever drawn more cautious**; 0 in a real match;
  - `NoisyEnemy` (`offline/noisy_enemy.py`): a copy of the framework's EnemyStateMachine with per-game habits (box
    size 5-11, look-ahead, aiming at me, striking my trail next to it); at randomness 0 it equals the original move
    for move (tested).
- **Simulation** (`offline/match.py`, `collect.py`): the real engine, headless, with no turn delay, one game per worker
  process. A subclass skips the engine's per-turn full-board copy when nobody watches. Threads do not help (the work
  holds Python's global lock: 4 threads gave 1.7x, 4 processes 3.6x); processes scale to about 16-20 on this CPU. A
  2000-turn game takes about 71 s against NoisyEnemy, 151 s against our own agent (two belief filters).
- **Evaluation** (`offline/evaluate.py`): variants played on the *same* games (paired), so the comparison's noise is
  the difference per game, not the spread between games.

---

## 6. The less important parts, briefly

- **`geometry/`**: bitboards (one int64 word per row: set operations on whole boards are a handful of NumPy ops;
  batches are `(N, 50)`), flood fill and the capture rule, BFS distances by dilation, taxicab distance transforms,
  Zobrist hashing. The fast substrate the belief and the snapshot rely on.
- **`rules/`**: the game rules re-implemented twice. `reference.py` is slow and plain (the readable truth);
  `transition.py` steps a whole batch of boards with bit operations (what the belief uses). Both are checked against
  the framework's own `MoveResolver`. `match.py` holds respawn cells and start lands.
- **`data/`**: our own types. `WorldState` (a full true board, for tests), `cells.py` (cell codes), `types.py`
  (`Direction`, `MoveResult`), `belief_view.py` (what the belief publishes).
- **`prediction.py`**: predicts every move's result before it is made; a mismatch with the engine is logged as an
  error. It caught real bugs, and it is how the run-time layer proves itself in live matches.
- **`diagnostics.py`**: counters, latency statistics, anomalies, a per-turn JSON-lines log (`runtime_logs/`), the
  end-of-match report. Optional; nothing depends on it for correctness.
- **`probe.py`**: a deliberately simple loop-maker that can die on purpose and cut adjacent trails, to exercise the
  runtime's death and kill inference. A test tool, not a strategy.
- **`runtime_check.py`**: plays a headless match and scores observations, memory, inferred events, predictions and
  the belief against the engine's true board. Also holds the opponents table (enemy, basic, random, a scripted
  hunter).
- **`offline/headless.py`**: runs the real GUI window with a dummy display, driven by posted mouse clicks, and checks
  that the GUI's recorded game matches the engine frame by frame. **`offline/locations.py`**: where training files go.
- **`examples/`**: the framework's starter example controllers (basic, random), used as opponents.
- **`behavior/maps.py`, `paths.py`, `votes.py`, `common.py`, `frame.py`**: small helpers (box filter, path following
  and capture gain, the hypotheses' per-cell votes, shared state helpers, the per-frame bundle).
- **Configs**: `behavior/config.py` (`BehaviorConfig`: switching, safety, RETURN, TRACK, exits, randomness) and
  `q/config.py` (`QConfig`: input scaling, label window, exploration, the stand-in's weights). One per layer, so
  neither layer has to know the other's numbers.

---

## 7. How we know it works

The strongest evidence is that **the real engine is used as the oracle wherever possible**:

- **Rules.** The reference and batched rules agree with the framework's `MoveResolver` on ~25 hand-built scenarios,
  on every move of randomly played games (~6,000 per test run, 60,000 once, zero mismatches), and on every turn of
  full engine matches.
- **Run-time layer.** In live matches: every move's result predicted correctly (600/600), inferred deaths and kills
  equal the engine's counters, memory of my territory exact every turn.
- **Belief.** The true opponent cell never had zero probability, and was certain whenever in view.
- **States and safety.** Unit tests on hand-built boards for each state, the chooser and the look-ahead (including the
  property that the predicted margin is never better than what the next frame actually delivers); whole matches with
  0 deaths against every opponent, both seats.
- **Recording.** A game recorded in the real GUI window matches the engine's boards on every frame; the recorded values
  equal the network's output recomputed from the recorded inputs (to 1e-6); a replay from tapes regenerates every
  array byte for byte.
- **Refactors.** Every refactor and speed-up was checked with a *play fingerprint*: three fixed matches hashed frame by
  frame (inputs, values, masks, states, moves) must be byte-identical before and after.
- **The whole loop** (online training -> weights -> the GUI's controller playing them -> its record -> a dataset ->
  training) was run end to end with nothing mocked.

---

## 8. Results, limits and what comes next

**Results with the hand-set stand-in** (96 full games, round 0b): it beats the basic, random and hunting opponents in
every game (e.g. 566 vs 9, 327 vs 9) with almost no deaths (2 in 96 games), but **loses to EnemyStateMachine in 46 of
48** (about 200 vs 1,300 cells): the enemy makes safe rectangular bites all game, while our worst-case safety rules
keep our loops small.

**The submitted network** is online15's epoch 25 (`q_training/online/online15/epoch_025/`: Q `weights.pt` and the
trained K `k.npz`), the controller's `DEFAULT_WEIGHTS` and `DEFAULT_K_WEIGHTS`. On its gate (16 real-belief games
against EnemyStateMachine copies and NoisyEnemy, hand-set K) it won 14 of 16, mean territory difference +347 +- 72
(enemy copies: 4 of 4, +504). The run's player at the time (epoch 4, `best.pt`) won 12 of 16, +416 +- 149; the paired
difference, -68 +- 181, is a tie, so the gate did not replace the player. online15's newest network (epoch 64,
`latest.pt`) won 14 of 16 too, at +226 +- 63. Both are kept in git; every other checkpoint is not.

**Limits, honestly:**

- **Worst-case safety is conservative.** D1 assumes the enemy could be on *any* cell it might be on, which after some
  unseen turns is much of the board, so the margin is small and RETURN fires early. A probability floor on the
  possibility mask trades safety for territory (measured: much more territory against EnemyStateMachine, but 5-22
  deaths per match against the hunter). This is exactly the risk-taking a learned Q should eventually earn.
- **Q can only choose among states, not overrule safety**, so a trained Q alone cannot fix the conservatism above.
- **The first trained network (round 0) was weak** (R^2 0.27 on held-out games; it used only the scalars branch) and
  is not the default. The online loop improved on it (below).
- **The belief assumes the opponent's respawn mirrors mine** (true in the default config) and models it with a
  hand-fitted motion model.
- **Cost**: the belief is tens of ms per turn; fine for the game, the main cost of simulation.

**Next steps, in order of expected value:** tune the possibility-mask floor against the risk it adds (measured per
opponent); a paired comparison of the submitted network with the trained K against the hand-set K (the gate measured
only the hand-set one); bootstrapped labels for slow payoffs.

---

## 9. Questions to expect, with short answers

**Why a state machine plus a learned Q, rather than end-to-end reinforcement learning?**
The states encode the game's structure (expand, close, hunt, scout, wait) and are exactly testable; the hard part to
hand-tune is *when* to switch, which is what Q learns. Safety stays in code where a test can prove it. The learning
problem becomes a six-way value regression with plenty of data, instead of credit assignment over 2,000 moves.

**Why are the safety rules not learned?**
A death costs the whole territory, and the data has few of them, so a learned model would underweight them. Worst-case
rules are provable: the look-ahead test checks that the predicted margin is never better than what really happens next.

**What is the safety margin exactly?**
D7 = D1 - D4: the enemy's worst-case time to reach any cell of my trail, minus my time to get home. Positive means I
close the loop before it could arrive. The look-ahead computes it for the position after a move, counting the cells of
the way home as exposed too.

**Why the switching cost?**
Hysteresis. Two nearly equal values would otherwise swap the state every frame and the agent would dither between two
plans. It costs 2 cells of expected value to switch, 4 from RETURN back to EXPAND because that lengthens the trail.

**How do you handle not seeing the opponent?**
A belief filter over complete hypotheses, updated with the exact rules; the opponent's territory count (reported every
turn) is hard evidence that removes impossible histories; leftover probability is kept in a residual so nothing is
ever assigned zero by pruning. It feeds the snapshot (worst-case distances use every cell the enemy might be on) and
the probability map.

**How do you know the belief is right?**
Against the engine's true state over whole matches: the true cell never had zero probability and was certain when in
view; every hypothesis' counts were checked against the engine every frame.

**What does Q's output mean, and why that label?**
Cells of score difference over the next 50 of my turns, if I take that state now. 50 turns covers 99% of excursions
(measured), so most consequences of a choice land inside the window; an open loop at the end is credited only if safe.

**Why split by game, and why the control in staged training?**
Frames of one game are correlated; splitting by frame leaks the answer into validation. The control: a later stage
trains longer, so it would look better even with a useless branch; it was a real false positive here.

**Why PyTorch at play time when the brief asked for NumPy?**
Decided with the author: one network for training and play removes a re-implementation and its parity risk. To keep the
submission robust, PyTorch is imported only when weights are configured, and the controller falls back to the stand-in
if PyTorch or the file is missing.

**How did you make simulation faster?**
Profiled first: two thirds of a game was the belief, a quarter the engine's bookkeeping. Fixed the hot spots (bit
counting with `np.bitwise_count`, deduplication by hashing instead of sorting rows, skipping the engine's per-turn board
copy when nobody watches): 68 s -> 17 s for a profiled game, every move unchanged. Parallelism is by process, one game
each; threads do not help because the work holds Python's global lock.

**Where would you take it next?**
Earn back the territory the conservative safety costs (a calibrated possibility floor, measured per opponent), keep
training Q online against stronger opponents, and close the gap between how the neural K is trained and how it plays.

**If you had to explain one design rule of the code?**
Each layer only knows the one below: framework types stop at the adapter, behaviour knows nothing of Q, training code
is never imported during a match. That is what made each part testable on its own, and every refactor checkable.
