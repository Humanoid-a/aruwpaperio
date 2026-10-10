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
| **A. Inputs**: snapshot (distance maps, margins), 22 scalars, 7x21x21 grid, 8x10x10 confidence map | Pure functions of belief, memory and observation | Hand-built boards; the brief's acceptance list |
| **B. States and chooser**: five movers, validity masks, switching cost, hard rules | Deterministic logic, no learning | Unit tests, then whole matches with a stand-in for Q |
| **C. The network**: data collection, labels, PyTorch training, NumPy deployment | Offline learning plus a forward pass | NumPy-vs-PyTorch parity, calibration, win rate |

Constraints (from the brief, confirmed with the author): the framework is untouched; during a match only
NumPy runs (the author confirmed NumPy is allowed in the submission; SciPy is not used at all, a box
filter is an integral image); PyTorch (installed, CUDA) is used offline only, and the trained network is
re-implemented in NumPy for play. Every input builder is a pure function.

## 2. Mapping to what already exists

| The brief needs | Provided by |
| --- | --- |
| Possibility mask, position marginal | `BeliefView.support`, `BeliefView.position` |
| Weighted hypotheses, residual mass | the belief filter; **new** `BeliefView.detail` (`BeliefDetail`) exposes weights, trails, both lands (per distinct land) and the residual grid |
| Distances, paths, flood-fill gain | `geometry.distance` (`bfs_distance`, `shortest_path`, `taxicab_distance`), `geometry.fill.capture_region` |
| Memory map, last-seen ages, sightings | `MemoryMap`, `TurnView.opp_last_seen` |
| Playing a match headlessly, scoring | `runtime_check`, `GameEngine` |
| Everything else (snapshot, inputs, states, chooser, network, training) | new, under `candidate/agent/q/` |

## 3. Module layout (`candidate/agent/q/`)

```
config.py      every constant of the brief in one place (QConfig / Q)
types.py       State enum (EXPAND, RETURN, HUNT, TRACK, PATROL = output order of the network)
votes.py       hypothesis votes per cell: enemy land, enemy trail, my land (chunked)
snapshot.py    per-frame distance maps, margins, D1..D8 (pure)
maps.py        box filter (integral image) and neighbour shifts, plain NumPy
paths.py       path following, capture gain, the 8 tie-break orders
features.py    the 22 scalars
grid.py        7 x 21 x 21 egocentric grid (+ rotate() for augmentation)
confidence.py  8 x 50 x 50 confidence board, pooled to 8 x 10 x 10
inputs.py      the three inputs for a frame, together
frame.py       Frame (view + snapshot + remembered expedition direction) used by the states
exits.py       PATROL's exit score
movers.py      the five states' own logic, RETURN's candidate paths, and the validity masks
chooser.py     masks, switching cost, the two hard rules
safety.py      the margin a move would leave (look-ahead)
heuristic.py   a stand-in for Q (hand-set values) so the controller plays before any model exists
strategy.py    QStrategy: the Strategy plugged into agent.controller.Controller (and the GUI's Candidate Controller)
```

Later (steps 7-10): `log.py`, `network.py` (NumPy forward pass), `train/` (labels, dataset, PyTorch model,
training, export, parity). Training code never lives under `agent/`, so a match never imports PyTorch.

## 4. Decisions on points where the brief is ambiguous or inconsistent

Each of these is also a constant or a documented choice in the code.

1. **Possibility mask: resolved by measurement, default kept at the brief's literal reading.** The belief's
   support is every cell with non-zero probability, which after some unseen turns is a large part of the board,
   so the worst-case margin D1 is small and RETURN fires early. A probability floor
   (`QConfig.possible_min_probability`) trades safety for territory. Measured with the stand-in, 400 turns:
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
   already handles that). Margins use the brief's definitions with `QConfig.margin_offset = 0` as the single
   place to shift them.
6. **PATROL's exit score** is only named in the brief. Defined here as `w_room*room + w_safety*safety +
   w_info*info - w_dist*walk`, each term scaled to 0..1 (room: free-cell density around the cell just outside;
   safety: worst-case enemy distance / 50; info: age of the cells beyond / 100; walk: steps from me inside my
   territory / 50). Weights are in the config.
7. **"50 turns"** in the label means 50 of my own frames.
8. **Rotation/flip augmentation** transforms the grid and the confidence map together. 50 is divisible by 5, so
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

## 5. Build order (each step ends with tests)

1. **Config, snapshot.** `config.py`, `types.py`, `votes.py`, `snapshot.py`, `BeliefDetail`. D7 = D1 - D4,
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
9. **NumPy forward pass**, parity with PyTorch within 1e-5 on saved inputs.
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
| 3 Confidence map | done; the brief's acceptance checks pass; checked against the engine (below) |
| 4 The five states | done, with unit tests for each and the validity masks |
| 5 Chooser | done, plus the look-ahead (decision 13) |
| 6 Play without a network | done: `QStrategy` + `HeuristicQ`, wired in as the GUI's Candidate Controller and into `runtime_check --strategy q` |
| 7 Logging and data collection | not started (the strategy already keeps a `FrameRecord` of the inputs exactly as Q saw them) |
| 8 Labels and training | not started (PyTorch with CUDA is installed) |
| 9 NumPy forward pass and parity | not started |
| 10 Iterate and evaluate | not started |

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

### Open points

- The brief's open question 2 is answered (NumPy is allowed). Open question 3 (is 50 turns long enough for
  excursions) is for the labels step to measure: the stand-in's excursions are short (mean trail 4-30).
- The stand-in's constants (`h_*` in the config) are untuned; they only have to produce safe, varied play to
  collect round-0 data.
- Whole-suite time grew to several minutes because of the engine matches in `test_q_e2e.py` (about 4.5 minutes).
