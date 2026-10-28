# Q model and action-states: build plan

Source: `Q_Model.pdf` (the brief, "Q Model & Action-States — Implementation Brief", Oct 9, 2026). This
document turns the brief into a build order, records every place where the brief was ambiguous or
inconsistent and what was decided, and tracks progress. The brief stays the authority for *what* Q is;
this file is the authority for *how it is built here*. Status is at the bottom.

## 1. What is being built

Q is the controller's decision model. Every frame it scores five action-states (EXPAND, RETURN, HUNT,
TRACK TARGET, PATROL); the controller switches to the best valid one, with a switching cost, and a hard
safety rule that Q cannot outvote. Each state has its own deterministic logic that produces the move.

Three separable systems, built in this order so each can be tested without the next:

| System | Nature | Verified by |
| --- | --- | --- |
| **A. Inputs**: snapshot (distance maps, margins), 22 scalars, 7x21x21 grid, 10x10x10 probability map | Pure functions of belief, memory and observation | Hand-built boards; the brief's acceptance list |
| **B. States and chooser** (`agent/behavior`): five states, validity masks, switching cost, hard rules | Deterministic logic, no learning | Unit tests, then whole matches with a stand-in for Q |
| **C. The network**: data collection, labels, training, play | One PyTorch module for training and play | Calibration, win rate |

Constraints: the framework is untouched; NumPy is allowed in the submission (confirmed by the author); SciPy is not
used at all (a box filter is an integral image). The brief also asks that only NumPy run during a match, with the
trained network re-implemented in NumPy; **the author decided otherwise**: the network is one PyTorch module used for
training and for play (decision 26). Every input builder is a pure function.

## 2. Mapping to what already exists

| The brief needs | Provided by |
| --- | --- |
| Possibility mask, position marginal | `BeliefView.support`, `BeliefView.position` |
| Weighted hypotheses, residual mass | the belief filter; **new** `BeliefView.detail` (`BeliefDetail`) exposes weights, trails, both lands (per distinct land) and the residual grid |
| Distances, paths, flood-fill gain | `geometry.distance` (`bfs_distance`, `shortest_path`, `taxicab_distance`), `geometry.fill.capture_region` |
| Memory map, last-seen ages, sightings | `MemoryMap`, `TurnView.opp_last_seen` |
| Playing a match headlessly, scoring | `runtime_check`, `GameEngine` |
| Everything else | new: the snapshot, the states and the chooser under `candidate/agent/behavior/`; the inputs, network, recording and the strategy under `candidate/agent/q/`; training under `candidate/offline/` |

## 3. Module layout: `candidate/agent/behavior/` and `candidate/agent/q/`

Two packages, one per layer (decision 27). `behavior` is the action-states: what the board says, how each state
moves, when it is valid, and the rules that override any scoring. It knows nothing of Q. `q` is the value function:
its inputs, the stand-in and the network, recording for training, and `QStrategy`, which joins the two. Imports go
one way only, `q -> behavior`.

```
candidate/agent/behavior/
  states.py      State enum (EXPAND, RETURN, HUNT, TRACK, PATROL = output order of the network)
  config.py      every constant of the action-states (BehaviorConfig / BEHAVIOR)
  snapshot.py    per-frame distance maps, margins, D1..D8 (pure); read by the states and by Q's inputs
  votes.py       hypothesis votes per cell: enemy land, enemy trail, my land (chunked)
  paths.py       path following, capture gain, the 8 tie-break orders
  maps.py        box filter (integral image) and neighbour shifts, plain NumPy
  frame.py       Frame (view + snapshot + remembered expedition direction + config) used by the states
  exits.py       PATROL's exit score
  patrol.py      PATROL       \
  expand.py      EXPAND        |  one module per state, each with valid(frame) and move(frame)
  return_home.py RETURN        |  (RETURN's candidate paths and their scores are in return_home.py)
  hunt.py        HUNT          |
  track.py       TRACK TARGET /
  common.py      helpers the states share (first, toward, safe_dest, in_land, pace)
  moves.py       the table: valid_states(frame) (the masks) and move_for(state, frame)
  chooser.py     masks, switching cost, the two hard rules
  safety.py      the margin a move would leave (look-ahead)
  style.py       randomness for training games (BehaviorConfig.randomness, 0 in a real match; decision 28)

candidate/agent/q/
  config.py      every constant of the Q model (QConfig / Q): input scaling, labels, exploration, the stand-in
  features.py    the 22 scalars
  grid.py        7 x 21 x 21 egocentric grid (+ rotate() for augmentation)
  probmap.py     10 x 50 x 50 probability map of the whole world, pooled to 10 x 10 x 10 (Q_Probability_Map_Input.md)
  inputs.py      the three inputs for a frame, together
  heuristic.py   a stand-in for Q (hand-set values) so the controller plays before any model exists
  network.py     the Q network (QNet, PyTorch), its file format, pick_device(), predict(), and NetworkQ, the trained Q as
                 the strategy uses it (imported only when a trained network is asked for)
  record.py      GameRecorder: one file per game, every frame's inputs, values, masks, states, move, label ingredients
  version.py     inputs_version(): a fingerprint of the code and settings that decide the inputs (stale-data check)
  strategy.py    QStrategy: the Strategy plugged into agent.controller.Controller (and the GUI's Candidate Controller):
                 snapshot -> inputs -> Q's values -> behavior's masks and chooser -> the state's move -> look-ahead and
                 guard; also the brief's exploration and the hook for the recorder
```

Training code lives in `candidate/offline/` and is never imported during a match:

```
locations.py   where the files go (candidate/q_training/: raw, datasets, weights, logs, smoke)
match.py       the one place that runs a headless engine match (collection, replay, runtime_check and the engine
               tests all use it); Taped / Replayer: move tapes and exact replay
collect.py     many recorded games in parallel (game record + sidecar with both tapes and the engine's result)
replay.py      regenerate a game's inputs with the current code from its tapes (for stale records)
headless.py    a direct run of the real GameWindow with no display, driven by posted clicks; verify_record()
dataset.py     the data handler: labels (and bootstrapped labels), building, saving, loading, summaries; CLI
train.py       the trainer: staged training with controls, augmentation, early stopping, calibration, export; CLI
evaluate.py    paired A/B evaluation of Q variants
online.py      online training: per epoch, a batch of games with the current network, then training on them (decision 28)
noisy_enemy.py NoisyEnemy: a copy of the framework's EnemyStateMachine with seeded per-game habits (decision 28)
```

Files: `candidate/q_training/raw/<run>/` (recorded games; `raw/gui/` is where GUI games are recorded),
`datasets/<name>.npz` (training sets), `weights/<name>.pt` + `.json` (trained networks and their reports), `logs/`,
`smoke/`, `online/<run>/`. raw, datasets, logs, smoke and online are git-ignored, except the networks that ship
(`q_training/.gitignore` lists them): the submitted controller loads online15's epoch 25.

## 4. Decisions on points where the brief is ambiguous or inconsistent

Each of these is also a constant or a documented choice in the code.

1. **Possibility mask: resolved by measurement, default kept at the brief's literal reading.** The belief's
   support is every cell with non-zero probability, which after some unseen turns is a large part of the board,
   so the worst-case margin D1 is small and RETURN fires early. A probability floor
   (`BehaviorConfig.possible_min_probability`) trades safety for territory. Measured with the stand-in, 400 turns:
   against EnemyStateMachine a floor of 1e-4 gives territory 642 vs 456 (floor 0: 133 vs 456), but against the
   hunter it gives 5-22 deaths per match (floor 0: 0 deaths in every run). The floor is exactly the
   risk-taking the trained Q is meant to learn, so the baseline keeps 0.
2. **Forced RETURN (rule 2) is gated on trail length > 0**, because RETURN is invalid without a trail and
   there is no margin to protect otherwise.
3. **Labels use raw D5 and D7** (cells and turns), not the scaled features. Logs store the raw values.
4. **"Enemy trail known (probability > 0.5)"**: the probability is hypothesis mass over total mass (residual
   mass contributes nothing to a trail), so the test is conservative. A visible enemy trail cell is always known.
5. **Margin = (minimum over my exposed cells of the enemy's worst-case arrival) - closing time**, exactly as
   written. It is conservative (a cell not laid yet is not exposed yet); a tighter version is a later change.
   The one-turn question (brief, open question 1) is answered by the engine's turn order: players alternate,
   one move each, so exactly one enemy move happens between my frames except at the first look (the belief
   already handles that). Margins use the brief's definitions with `BehaviorConfig.margin_offset = 0` as the single
   place to shift them.
6. **PATROL's exit score** is only named in the brief. Defined here as `w_room*room + w_safety*safety +
   w_info*info - w_dist*walk`, each term scaled to 0..1 (room: free-cell density around the cell just outside;
   safety: worst-case enemy distance / 50; info: age of the cells beyond / 100; walk: steps from me inside my
   territory / 50). Weights are in the config.
7. **"50 turns"** in the label means 50 of my own frames.
8. **Rotation/flip augmentation** transforms the grid and the probability map together. 50 is divisible by 5, so
   pooling commutes with rotation; a test checks it.
9. **Logging size.** The brief says "about 1.6 KB per frame"; that is the confidence map alone. Grid (3,087 B),
   scalars (88 B) and map (1,600 B) are about 4.8 KB. The grid is bit-packed (about 390 B), giving about 2.1 KB.
10. **HUNT validity** also requires a known enemy trail cell to be reachable (otherwise there is no move).
11. **"Sideways" in RETURN** means perpendicular to the expedition's outward direction (remembered while the
    trail is out), not to the last step.
12. **Unseen enemy, never sighted**: the reach region is the diamond around the enemy's respawn with radius =
    turns played, since the enemy was known to start there.
13. **Safety rule with a look-ahead (a fix to the brief's rule 2).** The brief's hard rule looks at the margin of now
    and sends the controller to RETURN when it is at or below 1. But a step straight toward an approaching enemy
    costs up to three turns of margin (the enemy closes one, my new trail cell is one nearer to it, my way home is
    one longer), so a margin of 4 becomes 1 and then -2 before the rule can act. Measured: with the rule as
    written the stand-in died 19 times in 400 turns against the hunter at the close respawn, every time this way.
    `safety.margin_after` computes the margin the next frame would show if a move were made (worst case), and the
    strategy refuses any non-RETURN move that would leave it below `safety_floor` (0): it goes home instead, or
    stays inside my land if there is no trail. This is a hard rule too (Q cannot outvote it), the brief's rule is
    kept, and `safety_lookahead=False` switches the addition off. Deaths against the hunter: 19 -> 0, and in all
    16 runs of the matrix.
14. **D5 and D6 must not depend on the board's orientation.** The gain of "my shortest path home" depends on which
    of several equally short paths is followed, so a rotated board gave a different D5, against the brief's
    acceptance check that rotating leaves every scalar unchanged. They are now the best gain over the eight
    tie-break orders (the rotations of N,E,S,W and of its mirror image), a set closed under rotation and
    mirroring, which makes them exactly invariant (tested for 5 rotations/mirrors).
15. **`Direction.NORTH` is 0**, so `x or y` on directions silently treats north as "no move". All direction
    choices in the states use explicit `None` tests (a bug of this kind was found and fixed by the PATROL test).
16. **The first model needs data.** Round 0 is collected with `heuristic.py` (a hand-set stand-in for Q) plus
    the brief's exploration; later rounds use the trained model.
17. **Recording: the inputs as seen, plus the moves.** The controller writes every frame's inputs exactly as Q saw
    them (the brief's rule), with values, masks, the state before and after the decision, whether a hard rule or
    exploration decided, the accepted move, and the label ingredients (both counts, raw D5 and D7). Headless games also
    keep both players' move tapes and the engine's result in a sidecar. The engine is deterministic given the moves, so
    a tape replays the game exactly, and running the current pipeline on the replay regenerates the inputs as today's
    code would have seen them in that very game (tested: byte for byte identical when the code has not changed). Every
    record carries `inputs_version`; stale records are replayed instead of thrown away. Size: about 0.5 MB per 2000-turn
    game compressed.
18. **GUI games are recorded too**, by the Candidate Controller itself, into `q_training/raw/gui/` (the controller as the
    GUI builds it records; code that brings its own Diagnostics does not unless asked). They have no tape, so the final
    difference of their label window is the one at my last frame.
19. **Exploration**: 3% of frames start a random valid state, held 1-8 frames (uniform) or until invalid; hard rules end
    it. Seeded per game, so a collection is reproducible. Off by default (only collection turns it on).
20. **Labels are trained scaled**: divided by their standard deviation on the training games, folded back into the last
    layer at export; the exported network outputs cells and every reported loss is in cells squared.
21. **Staged training against a control.** A later stage trains longer than the one before it, so its lower loss can
    come from the extra training rather than the new branch (seen: with a noiseless target, a grid branch fed only zeros
    "won" by 14%). Each new branch is therefore compared with the current configuration trained on for the same budget
    from the same weights, and kept only if it beats that by 2%.
22. **Blocking, a fix found in the data.** The first collection showed one game with 23 deaths against the hunter, all
    one cycle. Two causes: (a) the way home counted a land cell the visible enemy stood on as a target, so D4 said 1
    while the real way home was 3; (b) the look-ahead ignored that the enemy can block my way home by standing on it,
    not only kill me on my trail. Fixes: home targets exclude the enemy's cell; the look-ahead and RETURN's path scores
    treat the cells of the way home, the entry land cell included, as cells the enemy must not reach first. Replaying
    round 0's hunter games with the fix: deaths 29 -> 1, mean territory 301 -> 327. Regression tests rebuild the scene.
23. **Devices.** Training and play use the GPU only if PyTorch can really run a kernel on it (`pick_device` runs one),
    else the CPU. The first installed build (cu126) had no kernels for the RTX 5060 (sm_120) and reported CUDA as
    available anyway, which is why the check runs a kernel; the cu130 build works. For one frame at a time the CPU is
    as fast or faster (about 0.1 ms against 0.25 ms on the GPU), so collection workers, 16 processes at once, run the
    network on the CPU rather than each opening a CUDA context on an 8 GB card.
24. *(withdrawn: there was a parity check between the NumPy and PyTorch forward passes; with one PyTorch network for
    training and play there is nothing to compare.)*
25. **Bootstrapped labels** (the brief's later improvement) are implemented as an option of the data handler: the
    open-loop credit at the end of the window is replaced by a previous round's value of the state taken there.
26. **The network is PyTorch in play too (the author's decision, against the brief's NumPy-only rule).** One module,
    `agent/q/network.py` (`QNet`, built from `nn.Sequential` branches), is trained and played; the NumPy re-implementation,
    its PyTorch twin and the parity tool were removed. To keep the submitted controller robust: nothing that runs without
    a trained network imports PyTorch; the Candidate Controller imports it only when weights are configured and falls back
    to the stand-in if PyTorch or the file is missing; the device is chosen by `pick_device` (decision 23). The round-0
    weights were converted to the new format and give the old network's values to within float32 rounding (1.3e-4 on
    values up to 693, on 480 saved real frames).
27. **The action-states and Q are separate packages, with a config each (the author's decision; the brief asks for one
    configuration file).** `agent/behavior` holds the states, their validity, the snapshot they read, the chooser and
    the safety rules; `agent/q` holds Q's inputs, the models, recording and `QStrategy`. `behavior` never imports `q`,
    so the states run and are tested with any five values. Each layer owns its constants: `BehaviorConfig` (switching,
    safety, what counts as known or possible, RETURN, TRACK, exits) and `QConfig` (input scaling, labels, exploration,
    the stand-in's weights). The two numbers both layers used are now one per layer: the scalars divide distances by
    `QConfig.scale_distance` (50) and turns since sighting by `QConfig.scale_since_sighting` (100), where they used the
    snapshot's caps of the same values. The split was checked to change nothing: three fixed 300-turn matches give
    byte-identical inputs, values, masks, states and moves before and after. The inputs fingerprint changed (the files
    moved), so records made before it are replayed when a training set is built.
28. **Randomness for training games, and online training (the author's request).** Every controller here is
    deterministic, so the same two controllers play the same game every time and a batch of games between them is one
    game repeated. Two seeded sources of variety, both off (0) by default and in a real match:
    * our agent: `BehaviorConfig.randomness` (`agent/behavior/style.py`). Each game draws its own switching costs,
      RETURN's buffer and sideways reach, TRACK's distance band and PATROL's exit weights, a smooth preference over the
      board for where PATROL goes out, and a rate at which EXPAND jogs one step sideways. The safety rules are only
      ever drawn more cautious (less cautious would mean avoidable deaths, not variety), and nothing the snapshot reads
      is varied, so Q's inputs mean the same in every game. At 1.0, against the hunter at the close respawn: 0 deaths
      in 3 of 3 games, territory 154-360 (unvaried: 255).
    * the opponent: `offline/noisy_enemy.py`, a copy of EnemyStateMachine (the framework file is untouched) whose
      habits are drawn per game: the size of its box (bite 5-11 instead of 7), its look-ahead, how often it prefers
      going out vertically, aims an expedition at where it last saw me, steps onto my trail next to it, or wanders on
      its land. At randomness 0 it is the original move for move (tested).
    `offline/online.py` then trains online: each epoch plays a batch of games (200 by default) with the current network
    (unvaried, with the brief's 3% exploration, recorded) against NoisyEnemy and against our own agent with randomness,
    builds labels from them, fine-tunes the network (keeping the old weights if no training epoch beats them on the
    validation games), and records the epoch (its games, weights and a summary). The games are simulated on the
    headless engine in parallel, never in the GUI.
29. **Simulation speed.** Profiling one game showed two thirds of its time in our belief filter and a quarter in the
    engine's bookkeeping, mostly in a few places that did far more work than needed. Fixed, each checked to change
    nothing (the play fingerprint of decision 27 is byte-identical, and tests compare each with what it replaced):
    counting a board's cells with `np.bitwise_count` instead of unpacking every bit (`bitboard.popcount`); finding the
    distinct territory differences by hashing each row to one number instead of sorting whole rows
    (`territory.distinct_rows`; new ids are still handed out in the same order); a headless engine that does not copy
    all 2,500 cells into new objects after every turn when nobody keeps the boards (`match._ResultOnlyEngine`, a
    subclass; the framework file is untouched); and NoisyEnemy fetching its view once per turn instead of once per
    cell it looks at. One profiled 400-turn game: 68 s -> 17 s; a full 2000-turn game alone: about 150 s -> 71 s
    against NoisyEnemy, about 300 s -> 151 s against our own agent. What remains is spread thinly over the belief.
    Parallelism: games already run in separate worker processes, which scale with the cores. Threads inside one
    process do not: the work is many small NumPy and Python steps that hold Python's global lock, so 4 threads ran 4
    games only 1.7 times as fast as one (4 processes: 3.6 times). Inside a game there is little to run in parallel
    either: the players alternate and each turn needs the move before it, and with every core already busy playing
    other games, splitting one game across cores could not raise the number of games per hour anyway.
30. **The belief's motion model K, fitted to recorded games, and the belief's accuracy (`offline/motion_fit.py`).**
    The 698 recorded headless games were replayed from their move tapes with our own rules (exactly: every replay
    ends at the engine's recorded counts; 60 s for all, no engine needed), giving 1,396,000 opponent moves with the
    opponent's true situation. K's seven weights were fitted by maximum likelihood under the belief's own probability
    model, split by game, every opponent type weighing the same, with L2 towards the hand-set weights (a prior: in all
    that data a move that would kill me was available only 48 times, always taken, too rare to estimate). Findings:
    * One-step (the true state as the hypothesis, held-out games): fitted to the purposeful opponents (enemy, noisy,
      hunter, basic), K predicts their moves better than the hand-set one (log-likelihood per move, enemy -0.842 vs
      -0.902, noisy -0.798 vs -0.873, hunter -0.347 vs -0.868); it is worse for the random mover and our own agent.
      The data says: purposeful opponents never step on their own trail (0 of 490,000 chances), mostly keep their
      direction (79-97%), and leave their land when on it; the pull home matters less than set.
    * The whole belief (32 held-out games replayed through our agent's real belief, scored every turn): the true cell
      was never ruled out (possible on 100% of turns, every K). But a better one-step K did not make a better belief
      for EnemyStateMachine and NoisyEnemy: the mean log-probability of the true cell got slightly worse (enemy -7.77 ->
      -8.29), while it improved greatly against the hunter (-11.3 -> -5.6). Out of view, against enemy and noisy, the
      belief's mean probability on the true cell is about 0.004 (ten times a uniform guess over the board), but its
      mean log-probability is about that of a uniform guess (-7.8): it is often confidently wrong. The limit is the
      belief's calibration out of view, not K's one-step accuracy. Not adopted: the hand-set K stays the default.
      Next: tune K's sharpness (BeliefConfig.beta) and floor (epsilon) on the belief-level score directly.
31. **The probability map replaces the confidence map (the author's spec, `Q_Probability_Map_Input.md`).** Q's third
    input is now 10 x 10 x 10: per cell, ownership (free, mine, enemy) and trail (none, my trail, enemy trail) as
    layers that each sum to 1, a confidence per group (1 - entropy / log 3), and the two position layers; pooled in
    5 x 5 blocks (mean, positions summed). Layers keep the exclusivity a single value loses (0.45 mine / 0.45 enemy /
    0.10 free is a coin flip between two owners, not "free"). Residual mass follows memory by halves inside the reach
    region and memory's category outside it; my trail is exact. The network's third branch takes 10 channels (79,349
    parameters). Checked against the engine's true board over live matches: cells it is >= 99% sure of out of view are
    wrong at most 1% of the time. Networks and records made before it cannot be used as they are: the networks do not
    load (they are removed), and the records are stale and are regenerated from their tapes.
32. **Most training games are played with the true world, not the belief (the author's decision).** In all but
    `QConfig.belief_share` (20%) of the training games, our agents get the engine's true board as a certain belief
    (`offline/oracle.py`, `OracleBelief`, used through `Controller(belief=...)`), so the belief filter does not run.
    A 2000-turn game: against NoisyEnemy 78.5 s -> 16.3 s, against our own agent 158 s -> 26 s; an epoch of 200 games
    about 17 instead of about 50 minutes. Q's inputs in those games carry no belief noise; the probability map is then
    one-hot, and the belief games teach the spread-out case. The cost, accepted: with the truth the states and the
    safety rules act on it too, so the agent plays far more boldly than it can with the belief (in the timing games:
    1,257 cells against 248 with the belief, same matchup). Values learned in oracle games describe that bolder agent;
    the belief games, reported on their own in each epoch, show how it really plays. Records are marked (`oracle`), and
    a stale oracle record is regenerated with the truth too. Evaluation always uses the real belief.
33. **The states rewritten, six instead of five (the author's design, after watching games).** Each state now has one
    simple job, and Q decides when: EXPAND goes out from the nearest exit and straight on; the new EXPAND2 is the turn
    (sideways from the way out, keeping one side, towards more room); RETURN takes the shortest way to my nearest land
    (the old path search for a wider loop is gone: widening the loop is EXPAND2's job now); HUNT goes for the nearest
    cell the world map gives at least `hunt_min_probability` (20%) of holding the enemy's trail, instead of only cells
    above 50%; TRACK TARGET gains information while the enemy is out of view (its position, and land whose owner the
    map is unsure of), and is invalid when the enemy is seen; PATROL repositions at the best exit. The world map moved
    into the behaviour layer (`behavior/worldmap.py`, Snapshot.world) so the states can read it; Q's probability map is
    it, pooled. Q has six outputs and 23 scalars (the one-hot state); 79,478 parameters; the stand-in plays a rectangle
    (out 5, turn, home). Networks trained on five states cannot load; records made before are stale and are
    regenerated from their tapes.
    The rewrite exposed a hole in the safety look-ahead: it treated any step onto the enemy's trail as perfectly safe
    ("it sends the enemy home") and never checked RETURN's moves. But a killed enemy reappears at once on its respawn
    and moves next; hunting beside its land, a forced step home cut its trail, it reappeared next to my trail and
    stepped on it (1 death in 200 turns against the hunter at close respawns). Now a kill is scored with the enemy at
    its respawn, every move that would kill is checked (RETURN's included), and when even RETURN's step home would
    leave a negative margin the safest move is taken. Against the hunter: 0 deaths in both seats, 128-129 cells vs 9.
34. **A longer reward horizon (the author's request: turns lasted one step).** Measured in round 17's
    games: with the real belief, 77% of EXPAND2 runs lasted one frame, two thirds ended by the safety rules (the
    belief's uncertainty) and one third by Q; with the true world, 69% ended by Q, which valued RETURN or EXPAND a median
    7 cells above the turn. Changes: the label window is 100 frames (was 50), and online training bootstraps by default
    (`--no-bootstrap` to stop): a label adds the current network's value of the state taken at the end of the window, so
    it looks about 200 frames ahead. (A rule holding EXPAND2 for a minimum number of frames was added and then removed
    at the author's request: how long to turn is Q's decision, not a hard-coded one.) Found on the way: the round-17 network, with the real belief, never left
    home (PATROL 578 of 600 frames, territory 10-36 against the enemy copy) while with the true world it played well
    (369 vs 53): it was trained 80% on true-world games. Fine-tuned once on round 17's games with the new labels it went
    out (41-266, HUNT, RETURN, EXPAND). Balancing the belief games to the same total weight was tried and made it stay
    home again (bootstrapping amplified the old network's pessimism there); it is an option, off. Still open: raising
    the belief share, so the real-belief case gets more data of its own.
35. **D1..D8 from the full-resolution world map, above thresholds (the author's request); safety kept on the worst
    case.** Q's D1..D8 are read from the world map (not the pooled one), keeping only cells at least this likely:
    enemy position >= `position_threshold` (1e-4), enemy trail >= `trail_threshold` (0.5), land >= `land_threshold`
    (0.5); the belief-spread scalar counts the likely cells. Measured before choosing: out of view, the belief's
    probability on the enemy's true cell is below 1e-6 on 12% of turns, below 1e-4 on 28%, below 1e-3 on 44% (half the
    turns against the hunter below 1e-4), so any position threshold drops the truth often. That is acceptable for an
    input saying where the enemy probably is, not for a rule that must never be wrong, so the hard safety rules (forced
    RETURN, the look-ahead) keep the worst case under their own names (`safety_d1`, `safety_margin`, from every cell
    with any probability), and the record's `d7` (the label's "was it safe") is that worst-case margin, as before.
    With the stand-in, which decides on the worst case, nothing changed in play: 0 deaths against the hunter and the
    enemy in both seats. The inputs fingerprint changed (records are replayed when a set is built).
36. **A deeper belief model K, trained with Q; RETURN by highest safe gain; TRACK towards the enemy (the author's
    design).**
    * K is now a small neural network (`agent/belief/neural.py`: 17 features -> 32 -> 32 -> 1 per hypothesis and move,
      NumPy at play time) over features computed for every hypothesis from its own position, trail and land: the old
      seven, plus its distance home, my distance home (to my land as that hypothesis has it), its distance to my trail,
      the enemy-kills-me margin, my distance to its trail, the I-kill-enemy margin, its own safety as it can see it,
      whether it can see me, and its gain if it went home now (the enclosed part exact for the 256 most likely
      hypotheses with a trail, a lower bound for the rest: an exact capture costs about 27 us a board). The hand-set K
      stays the fallback. Cost in play: a 300-turn match 11.5 s -> 14.6 s.
    * K is trained every online epoch (`offline/k_train.py`) on the epoch's games against the enemy state machines and
      our own agent only, from the opponent's real moves replayed from the tapes, starting from the previous K; the
      next epoch's belief games run with it, records name their K, and replay uses the same K. First check on 20 games:
      validation log-likelihood of the move made, trained vs hand-set, enemy_copy -0.818 vs -0.889, noisy -0.722 vs
      -0.823, our agent -1.282 vs -1.274.
    * Checked on the way (the author asked): with the enemy in view the belief puts probability 1 on its true cell on
      100% of 506 turns and is certain of every visible trail cell; the part of its trail out of view stays uncertain
      (mass on its whole true trail 0.91 when all of it is in view, 0.25 when not), and about 27% of the mass is
      residual (position known, history not) even then.
    * RETURN now looks at every cell of my land it could close on (up to 8 steps beyond the nearest, at most 64), traces
      the shortest path to each, scores gain (enemy cells double) and the worst-case margin, and takes the highest gain
      with at least `return_margin` (1) turns to spare, else the safest. TRACK heads for the cell whose view covers the
      most of the enemy's position probability, less a small walking cost. With the stand-in, 600 turns: against the
      hunter 578 and 679 vs 9 (was 133 and 432), against the enemy 614 vs 643 and 560 vs 447 (was 531 vs 29 and 450 vs
      525), 0 deaths.
37. **Faster training games (the author asked, after measuring where an epoch goes).** An epoch (200 games of 2000
    turns, 16 workers, belief share 0.4) took about 26 minutes, 85-90% of it playing; the real-belief games were a
    third of the games and three quarters of the play time, a belief game against our own agent the slowest (two
    belief filters).
    * In half (`--self-oracle-share`, 0.5; 0 since decision 38) of the belief games against our own agent, that opponent alone is given the
      true world (`GameSpec.opponent_oracle`): our agent still runs the real belief, the game runs one filter.
    * The costliest games start first (`collect.run`), so no long game runs alone at the end; results keep the plan's
      order.
    * The belief itself, measured without a profiler (cProfile inflated the many small calls, PyTorch's especially; Q
      is under 1% of a game): the flood fill runs whole rows per round and drops finished boards (3x), the distance
      from me to each trail reads only the trails' non-empty row words (7x), the trail votes merge shared words
      before expanding them (2.4x), the return gain fills each land's holes once and finds the nearest land cell once
      per position, and the hypothesis store stops copying twice. All exact (checked against the old code on inputs
      recorded from games; the trail votes up to float rounding). A 1000-turn belief game: 50.5 s -> 38.1 s.
38. **Training that keeps only what plays better (the author asked, after online7 got worse epoch by epoch).** What
    the runs showed: every network of online6 and online7 read the 23 scalars only (the staged build at epoch 0 kept
    neither the grid nor the probability map, by its 2% rule, and fine-tuning never tried them again); the state the
    network favoured swung from epoch to epoch (RETURN/PATROL, then TRACK, then HUNT), each network trained on the last
    epoch's games only; and the belief games against an all-seeing copy of our agent (decision 37) were lost 37 times
    in 39. Changes (offline/online.py, offline/train.py):
    * All three branches always (`--branches all`): from scratch with train.full; a scalars-only network gets the
      other two switched on with their head weights at zero (same outputs to begin with).
    * `--history 3`: each network trains on the last three epochs' games.
    * A gate: after training, the learner plays `--gate-games` (16) fixed games with the real belief and no exploration
      (enemy_copy and the noisy enemy, both seats, far and close, the same seeds every epoch); it becomes the player
      (best.pt, which plays most training games) only if its mean territory difference is at least the player's on
      the same games. The learner itself is never thrown away and plays `--learner-share` (half) of the training
      games, so it still learns from its own choices. The stand-in is the first player: a network has to beat it.
    * Games per epoch start small (`--games` 60) and grow by `--grow` (1.5) each time the player is replaced, up to
      `--max-games` (200): many quick updates while the network changes a lot, more games per update once the
      improvements get small and the noise matters more.
    * `--self-oracle-share` is 0 by default.
39. **HUNT and TRACK rebuilt, measured against the true board (the author asked where HUNT goes).** Real-belief games,
    Q forced to prefer the state under test, the true board read only to score the moves:
    * HUNT went to the nearest cell the world map gave 20% enemy-trail probability. The world map spreads half of the
      residual mass (positions without a history) as enemy trail over the whole reach region, so hundreds of cells
      pass 20% while a real trail is a dozen cells: its target was a true trail cell 1% of the time, its move got
      closer to the true trail 35% of the time, 2 kills in 4,325 HUNT frames. Now, per hypothesis (its own trail, cell
      and land), a move cuts that trail if I can reach one of its cells before the enemy can reach its land (taxicab);
      the chance of a move is the share of the whole belief it cuts (the residual has no trail and adds nothing), and
      HUNT takes the best move when that chance is at least `hunt_min_chance`. That threshold is 0.4 because the
      chance was measured against reality: below 0.4 the true trail could be cut on 0 of 567 frames, above it on 23 of
      26. Result: 18 HUNT frames in 6 games, 5 kills, every move closer to the true trail, the right move whenever one
      existed. The world map itself is unchanged (it is Q's input).
    * TRACK only looked 30 steps far and charged 0.01 per step: with far respawns the likely area was out of reach,
      every reachable cell was worth about nothing, and the agent paced at home all game (2 of 6 games, 9 cells each).
      Now a place's worth is multiplied by 0.97 per step, with no reach limit: closer to the true enemy 71% of moves
      (was 63%), the target would see it 45% (was 31%), no game stuck.
40. **The training split and the frames.** The split by game is now fixed by each game's id (crc32): with `--history 3`
    a re-split at random put games the starting network had trained on into its validation set, which made its
    starting loss look better than it was and fine-tuning keep the old weights (online8 epoch 2). 85% train, 15%
    validation, no test split (the gate is the test); R^2 is reported on the validation games. Only every second
    frame is trained on (`--frame-stride 2`): consecutive frames are nearly the same board, and with 200 games and 3
    epochs of history the full set (about 6 GB on the GPU) would not have fitted the 8 GB card next to everything else.
41. **Training in many situations, and a learner that keeps learning (the author asked, after online9).** online9's
    learner rose from -1274 to +667 on the gate in 4 epochs (the players it replaced were only as good as the
    stand-in: paired +79 +- 238 and +132 +- 278), then stopped: in 4 of the last 7 epochs fine-tuning kept its starting
    weights, and in its own games it went back to PATROL and TRACK. The author also saw it fight well with the enemy in
    view but expand badly. Changes:
    * Situations (offline/scenarios.py): spawns far / close / very close (5-11 apart) / random, always mirrored with
      player 0 on the left (the agent's MatchRules.mirrored and its seat inference depend on that; checked: the true
      enemy cell stayed in the belief on all 3,000 turns of 10 such games); game lengths 2000, 1000, 500, 300; and in
      30% of the games another agent (the enemy machine, the noisy enemy, basic, random, or our stand-in at full
      randomness) plays our seat for the first 10-50% of the turns, our agent watching, then our agent takes over. Q's
      frames are recorded from the take-over on; the dataset also drops earlier turns, which a replay records. The
      gate and the benchmark stay in the real situation.
    * Labels bootstrap from the player (a target network), not from the learner's own estimates.
    * Fine-tuning at learning rate 3e-4 with patience 6 (was 1e-3, 4: the first pass made validation worse and
      training stopped before it recovered).
    * The gate is paired: the learner's per-game advantage over the player must be above 0 and at least 1 standard
      error.
42. **Bigger epochs, pretraining, an inherited K (the author asked, during online10).** online10 found that the trained
    K makes play worse: on the 16 gate games, the same network scored -398 +- 200 (the player) and -447 +- 211 (a
    learner) with the trained K against the hand-set one, while it predicts the opponent's moves better (log-likelihood
    about -0.55 against -0.84); with the same K the learner and the player were within the noise of each other, so the
    gate, which measured the player with the hand-set K and the learner with the trained one, was unfair by about 400.
    Changes: 150 games per epoch at first (was 60), up to 300; `--pretrain` trains the first learner on games recorded
    elsewhere before epoch 0 (e.g. `online10:0-6`); `--start-k` starts K, and the player's K, from a trained K while Q
    starts from `--start`.
43. **A champion opponent, less noise, a wider network, more games, exploration that rises (the author asked).**
    * `champion`: our own agent with a fixed Q (`--champion`, our strongest network, or the stand-in), the hand-set K,
      no exploration and behaviour randomness 0.1; in the default opponent mix. K also learns from its moves.
    * Less noise: the noisy enemy and our `self` opponent at randomness 0.2 (were 0.5). The gate keeps 0.5.
    * A wider network (`--width 2`, for networks built from scratch): the grid and probability-map branches have
      twice the channels (32/64/64/64 and 32/64/64), a 128-wide head; 298,726 parameters (79,478 at width 1), 0.59 ms a
      frame on one CPU thread. The weights file records its width; older files load as width 1. Q already overfits
      (training loss falls while validation does not), so the wider network relies on the bigger epochs and the gate.
    * 200 games per epoch at first (was 150), up to 400.
    * Exploration rises by 1.5x (up to 0.15) after every 3 learners in a row the gate rejects, and returns to its base
      when one is accepted.
    * Why the trained K plays worse (measured on 12 real-belief games, out of view): it puts 2.4x more probability on
      the enemy's true cell on average and its top guess is right 3.3% of turns against 0.9%, but it is overconfident:
      the median probability on the true cell is lower (0.00061 against 0.00094) and the true cell falls below 1e-4,
      where the snapshot's D's stop seeing it, on 20% of turns against 16%. Trained on single moves from the true
      state, its confidence compounds over the many moves the belief makes from guessed states.
44. **One K in play, and a cap on exploration (after online12).** online12 (width 2, pretrained on online10's
    epochs 0-6, champion in the mix) improved on the gate from -318 to -7 against the stand-in by epoch 8, then got
    worse as the rising exploration reached 10-15%: about one frame in seven a random state held for 8 frames, the
    labels full of random choices, its own play 21-22% TRACK and PATROL. The gate had measured the stand-in with the
    hand-set K and the learners with the trained K. Changes: our agents play with the hand-set K in training and gate
    games (`--play-k hand`, the default; K still trains and is reported every epoch; `--play-k trained` for the old
    way), and the exploration boost stops at 0.05. The fair check of online12's epoch 8 (both with the hand-set K):
    +463 against the stand-in's +536, -73 +- 180 per game, a tie; for that network the K made no difference (hand-set
    minus trained -66 +- 194), unlike the -400 measured on online10's networks.
45. **EXPAND by the angle to the border, labels weighted by distance, fine-tuning that moves (the author asked).**
    * EXPAND (expand.py) looks at the exits at most 2 steps away inside my land, fits the border's line at each (the
      principal direction of my land's edge cells within 3), and leaves on the side with the widest angle to it,
      1 - a|d . b| in quarter steps, then the nearest exit, then the exit score; with none in reach, the nearest exit
      as before. Two earlier versions paced inside the land (a walk cost, and the exit score before nearness: the
      score's staleness term changes as the agent walks), 718-848 of 2000 frames against 377-387. Against the old
      EXPAND on 36 paired games: territory -41 +- 52 a game (even), difference -174 +- 91 (fewer kills: 73 against 131).
    * Labels (dataset.py): frames 1-20, 21-50 and 51-100 weighted 1.0, 0.6 and 0.3, the tail value the last weight,
      all scaled so a steady gain gives the unweighted label. On online12's epochs 18-20, a network from scratch
      explained 0.306 of the weighted labels' variance on held-out games against 0.225 of the unweighted.
    * online14 learned nothing: in all 7 epochs fine-tuning kept its starting weights (the first pass already raised
      the validation loss). Validation is now checked 8 times a pass (`--val-checks`), the best point kept, and the
      fine-tuning learning rate is 3e-5 (was 3e-4): on online14's games with its own labels, +3.3% against +0.5%.

## 5. Build order (each step ends with tests)

1. **Config, snapshot.** `config.py`, `states.py`, `votes.py`, `snapshot.py`, `BeliefDetail`. D7 = D1 - D4,
   D8 = D2 - D3, D5/D6 = 0 without a trail.
2. **Scalars and grid.** 22 scalars (scaling, clipping, one-hot), 21x21 grid; scalars invariant under board
   rotation; off-board cells are 0 in every channel.
3. **Confidence map.** The riskiest input. Acceptance: one hypothesis and no residual gives confidence 1
   everywhere; all residual gives enemy-trail confidence 0 over the reach region; visible cells have confidence
   1; after pooling the enemy-position and my-position channels each sum to 1. Cross-checked against the
   engine's true state in a live match.
4. **The five states**, in the order PATROL (defines the exits), EXPAND, RETURN, HUNT, TRACK TARGET. Each is a
   pure function from the frame to a move or `None`, with a validity test.
5. **Chooser.** Masks, switching cost h (and the larger RETURN->EXPAND cost), the two hard rules.
6. **Play without a network.** `QStrategy` with the stand-in Q against EnemyStateMachine, Basic, Random and the
   hunter: no exceptions, no forfeits, no avoidable deaths; the inputs build every frame. This is the baseline
   everything later is compared against.
7. **Logging and data collection** with the 3% exploration (random valid state held up to 8 frames), parallel
   across processes; inputs logged exactly as played.
8. **Labels and training** (scalars only, then grid, then confidence branch, keeping each only if validation
   loss clearly drops), in PyTorch on the GPU.
9. **The network in play** (the brief: a NumPy forward pass with parity checks; replaced by decision 26).
10. **Iterate** (collect with the trained model, retrain, 2-3 rounds) and evaluate: per-state calibration,
    choice frequencies, 200-game A/B against EnemyStateMachine.

Steps 1-6 are the scope of the first pass; 7-10 follow.

## 6. Risks

- **Cold start**: no model yet, hence the stand-in and the exploration.
- **Stale logs**: any later change to the belief or to an input makes old logs wrong. Every log carries a
  version tag for the belief and for the feature set.
- **Speed**: the belief costs 30-60 ms a frame; Q's inputs must stay small next to it. The enemy-trail vote
  touches every hypothesis and cell, so it is computed in chunks and measured.
- **Over-caution**: see decision 1.
- **Overfitting**: splits by game, never by frame.

## 7. Status

| Step | State |
| --- | --- |
| 1 Config, snapshot | done: `config`, `types`, `votes`, `snapshot`, `BeliefDetail` on the belief view |
| 2 Scalars and grid | done, rotation/mirror invariance tested |
| 3 Confidence map | done; the brief's acceptance checks pass; checked against the engine |
| 4 The five states | done, with unit tests for each and the validity masks |
| 5 Chooser | done, plus the look-ahead (decision 13) and the blocking fix (decision 22) |
| 6 Play without a network | done: `QStrategy` + `HeuristicQ`, the GUI's Candidate Controller, `runtime_check --strategy q` |
| 7 Logging and data collection | done: `record.py`, exploration, `collect.py`, tapes and exact replay, GUI recording, a headless direct run of the real window (`headless.py`) |
| 8 Labels and training | done: `dataset.py` (data handler), `train.py` (trainer); runs on the GPU |
| 9 The network in play | done: `network.py`, one PyTorch module for training and play (decision 26) |
| 10 Iterate and evaluate | tools done: `online.py` (online training by epochs with bootstrapped labels as an option, and its smoke test; decision 28), `evaluate.py` (paired A/B), calibration and choice frequency in every training report. Not yet run for real (on hold until asked) |

### How to run it

```
# record: direct run of the real window with no display (also what you get in the GUI)
python -m candidate.offline.headless --turns 2000 --seat 0 --opponent "Enemy State Machine"
# or many headless games, recorded with the brief's exploration
python -m candidate.offline.collect --out round1 --games 96 --workers 16 --model heuristic
# build a training set, look at it
python -m candidate.offline.dataset build --raw candidate/q_training/raw/round1 --name round1
python -m candidate.offline.dataset info round1
# train (GPU if it works), evaluate
python -m candidate.offline.train --dataset round1 --name q_round1
python -m candidate.offline.evaluate --variants heuristic q_round1 --games 40
# online training: per epoch 200 games (half against NoisyEnemy, half against our agent with randomness), then
# fine-tuning; resumable; files in q_training/online/<name>/; and its end-to-end check (two tiny epochs, ~40 s)
python -m candidate.offline.online run --name online1 --epochs 10 --games 200 --workers 16
python -m candidate.offline.online smoke
```

Time per 2000-turn game, alone on the machine (decision 29): 71 s against NoisyEnemy, 151 s against our own agent (two
belief filters). Games run in parallel processes; on this machine (Core Ultra 9 275HX: 8 performance and 16 efficiency
cores) throughput grows up to about 16-20 workers and no further (300-turn games: 5.5 per minute on one worker, 33 on
8, 43 on 16, 45 on 20, 39 on 24). An epoch of 200 full games, half against each, takes about 45 minutes on 16
workers, plus the dataset and the training. `--turns` and `--games` trade time for data.

To play a trained network in the GUI, set the environment variable `CANDIDATE_Q_WEIGHTS` to its file before starting
the GUI (PowerShell: `$env:CANDIDATE_Q_WEIGHTS = "candidate\q_training\online\online15\latest.pt"`), or set
`DEFAULT_WEIGHTS` in `candidate_controller.py` (online15's epoch 25, with its K in `DEFAULT_K_WEIGHTS`), or pass
`weights=` to `CandidateController`. The console says which Q model it plays, and warns if the network was trained on
inputs built by other code (its inputs version differs from the current one).

### What step 6 measured (stand-in Q, 400 turns, both seats, close and far respawns)

- 16 of 16 runs clean (no errors, no forfeits, predictions exact), **0 deaths in all 16**, `guarded` (the final move
  filter had to step in) at most once per run, no retries.
- Against the hunter the stand-in kills 4-14 times per match at the close respawn and ends far ahead (255 vs 9, 244 vs
  9); against the random and basic opponents it ends 10-50 times their territory; against EnemyStateMachine it loses
  on territory (133-184 vs 245-456 in 400 turns): the baseline builds small loops (mean trail 4-8) because the
  worst-case margin is conservative (decision 1). This is the number a trained Q has to beat.
- Cost per frame (mean): snapshot 8-18 ms (the trail vote is about 9 of it), inputs 0.4 ms, Q 0.5 ms, state logic
  2-3 ms, next to the belief's 30-80 ms.
- The confidence map against the engine's true board over live matches: out-of-view cells with confidence >= 0.99
  were wrong 0.0000 of the time for enemy land and for my land, and 0.002 for the enemy trail (hunter); the
  enemy's true cell never had zero probability (600 frames). Low-confidence cells are wrong far more often (up to
  7%), as they should be.

### What steps 7-9 measured

- **Recording from a direct run.** Two full 2000-turn games in the real window with no display, Candidate Controller
  against Enemy State Machine, one from each seat: the record matched the engine's published boards on every frame
  (counts, and the move against the agent's actual step), 0 deaths (220 s and 210 s per game).
- **Collection.** Round 0b (after the blocking fix): 96 games of 2000 turns with the stand-in and 3% exploration in 25
  minutes on 16 processes, 0 failed turns, 2 deaths in all. By opponent: EnemyStateMachine 2 wins of 48 (mean
  territory 202 vs 1337), basic 12/12 (566 vs 9), random 12/12 (606 vs 9), hunter 24/24 (327 vs 9). Excursions: median
  6 frames out, 99% within 46, 0.89% longer than 50, so the 50-turn label window covers almost all of them (the brief's
  open question 3).
- **Replay.** A recorded game replayed from its tapes regenerated every recorded array byte for byte.
- **Training, round 0** (run on the CPU before the GPU build was fixed, and before training was put on hold; kept as
  `q_training/weights/q_round0`): 96 games / 192,000 frames, split 68 / 14 / 14 games. The grid branch did not beat its
  control (6,882 vs 6,777) nor did the confidence branch (6,472 vs 6,501), so the model uses the scalars branch. Test
  loss 4,900 against 6,732 for a constant predictor (R^2 0.27); calibration on the test games is close for EXPAND,
  RETURN and TRACK, high in the middle bins for the rare HUNT and PATROL. Its evaluation against the stand-in was
  stopped before it finished; it is not the default.
- **Smoke test of the whole loop** (`online smoke`): two epochs of 4 short games each, the first training a network
  from scratch and the second playing and fine-tuning it, on the GPU: about 40 s. (It replaced `pipeline.py`, whose
  "round" of collect, dataset, train and evaluate is one epoch of `online.py`.)

### Open points

- EnemyStateMachine beats the stand-in in 46 of 48 full games: its territory reaches 1,300-2,300 cells while ours stays
  around 200, partly because the worst-case safety rules keep our loops small (decision 1). A trained Q chooses among
  states but cannot overrule those rules, so the next decision is whether to train on more data first or revisit the
  possibility-mask cut-off for this opponent.
- Whole-suite time is now around 15 minutes (engine matches in the Q end-to-end, pipeline, headless and recording tests).
