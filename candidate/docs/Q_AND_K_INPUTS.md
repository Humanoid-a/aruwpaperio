# The inputs of Q and K, and how each is calculated

Q and K answer different questions, so their inputs are built from different places.

| Model | Question | Evaluated | Inputs built from |
|---|---|---|---|
| **Q** (`agent/q/`) | What is each of the six action states worth from here? | once per turn | my observation, my memory, the belief's world map |
| **K** (`agent/belief/`) | Which of its 4 moves will the enemy make? | once per belief hypothesis, per move | that hypothesis's own position, trail and land; my known position and trail |

Both are read from the same per-turn **snapshot** (`agent/behavior/snapshot.py`), which holds the bitboards, the
world map and the distance values described below.

---

## Part 1. Q: three inputs, built once per turn

Built by `agent/q/inputs.py` as `QInputs(grid, scalars, probmap)`.

### 1.1 The grid, 7 x 21 x 21 (`agent/q/grid.py`)

A crop of the board centred on my position. It is 21 cells wide, larger than the 11 x 11 view, so that my whole
trail fits. Every channel is 0/1 (stored as uint8).

| Channel | Meaning | Source |
|---|---|---|
| 0 | the cell is on the board | geometry (cells past the edge are 0 in every channel) |
| 1 | the cell is inside my current 11 x 11 view | observation |
| 2 | my territory, visible cells only | observation |
| 3 | my trail, the whole trail as far as the crop reaches | observation (always known) |
| 4 | enemy territory, visible cells only | observation |
| 5 | enemy trail, visible cells only | observation |
| 6 | the enemy's cell, if visible | observation |

Outside the view every channel is 0 except channel 3, so the grid carries only what is seen right now plus what the
agent knows for certain.

### 1.2 The scalars, 23 numbers (`agent/q/features.py`)

Each entry is scaled to about [0, 1] or [-1, 1]. The scales are in `QConfig` (`agent/q/config.py`).

| # | Name | Calculation |
|---|---|---|
| 1 | `trail_len` | cells of my trail / 100 |
| 2 | `my_count` | my territory count (engine) / 500 |
| 3 | `enemy_count` | the enemy's count (engine) / 500 |
| 4 | `score_diff` | (my count - enemy count) / 500 |
| 5 | `d1` | the enemy's taxicab distance to my nearest trail cell, from the *likely* enemy cells / 50 |
| 6 | `d2` | my shortest path (avoiding my trail) to the nearest *likely* enemy-trail cell / 50 |
| 7 | `d3` | the enemy's distance to its own land: from its seen cell, else a probability-weighted average over the likely cells / 50 |
| 8 | `d4` | my shortest path home to my likely land / 50 |
| 9 | `d5` | what I would capture by going straight home now: a flood fill of the loop, enemy land counting double / 100 |
| 10 | `d6` | the same for the enemy, from its likely trail, likely land and its seen or most likely cell / 100 |
| 11 | `d7` | the safety margin, d1 - d4, clipped to +-25 / 25 |
| 12 | `d8` | the kill race, d2 - d3, clipped to +-25 / 25 |
| 13 | `turns_left` | turns left / 2000 |
| 14 | `enemy_visible` | 1 if the enemy is in view, else 0 |
| 15 | `enemy_trail_known` | 1 if some enemy-trail cell is seen or at least 50% likely |
| 16 | `turns_since_sighting` | turns since the enemy was last seen, capped / 100 |
| 17 | `belief_spread` | log(1 + number of likely enemy cells) / log 2500 |
| 18-23 | `state_*` | the current state, one-hot over EXPAND, EXPAND2, RETURN, HUNT, TRACK, PATROL |

**"Likely" means thresholded.** D1 to D8 read the world map (1.3) above fixed thresholds (`BehaviorConfig`):

| Quantity | Threshold | Snapshot field |
|---|---|---|
| enemy position | P >= 1e-4 (if nothing passes: the most likely cell) | `likely` |
| enemy trail | P >= 0.5 | `enemy_trail_bb` |
| enemy land | P(enemy) >= 0.5 | `enemy_land_map_bb` |
| my land | P(mine) >= 0.5 | `my_land_map_bb` |

The hard safety rules do **not** use these thresholded values: they use every cell with any probability
(`possible`, `enemy_arrival`, `safety_margin`), because the thresholds drop the enemy's true cell on about 28% of
the turns it is out of view.

**Why counts and distances:** none of them change when the board is rotated or flipped, which lets training augment
games with rotations and mirrors (the grid and the probability map are rotated along).

### 1.3 The probability map, 10 x 10 x 10 (`agent/q/probmap.py`)

The belief's full 10-channel **world map** (`agent/behavior/worldmap.py`, 10 x 50 x 50) pooled into 5 x 5 blocks.

| Channels | Content | Pooling |
|---|---|---|
| 0-2 | P(free), P(mine), P(enemy land); they sum to 1 at every cell | mean |
| 3 | ownership confidence, 1 - entropy / log 3 | mean |
| 4-6 | P(no trail), P(my trail), P(enemy trail); they sum to 1 at every cell | mean |
| 7 | trail confidence | mean |
| 8 | enemy position probability (sums to 1 over the board) | sum |
| 9 | my position (one-hot) | sum |

How the world map is made, per cell:

1. **Hypotheses vote.** Each belief hypothesis adds its weight to the land it owns, its view of my land and its trail
   (`agent/behavior/votes.py`).
2. **The residual** (probability whose history was pruned) is spread over the reach region, the diamond around the
   last sighting the enemy could have walked since: ownership half by what memory says and half enemy, trail half
   none and half enemy trail.
3. **Cells in view are one-hot:** what is seen overrides the votes.
4. **My trail is exact,** and the enemy trail is 0 on it.
5. The position layer is the belief's position marginal, with the seen cell set to 1 when the enemy is visible.

Confidence is 1 for a certain cell and 0 for an even three-way split; it is pooled as a mean so a block half
certainly mine and half certainly free does not read as a block of guesses.

---

## Part 2. K: 17 features per (hypothesis, move)

K scores each of the enemy's 4 possible moves for every belief hypothesis; `move_probabilities`
(`agent/belief/motion.py`) then turns the 4 scores into probabilities with a softmax over the legal moves, mixed with
a small floor, so no legal move is ever impossible:

    P(a | s) = (1 - eps) * exp(beta * K[a]) / sum over legal a' of exp(beta * K[a'])  +  eps / |legal|

**Starting point: the hypothesis's own world.** Every feature is computed from that hypothesis's position, trail,
land and its view of my land, *after* the move being scored. The only shared inputs are my own position and trail,
which are known exactly. The outcome of each move (does it die, does it kill me, does it close a loop and how much
would that capture) is resolved first by the game rules in `agent/belief/generate.py` (`classify`).

### 2.1 The features (`agent/belief/neural.py`, `features`)

| # | Name | Calculation |
|---|---|---|
| 0 | `self_trail` | the move steps onto its own trail (it dies) |
| 1 | `kills_me` | the move steps onto my trail (I die): the enemy-kill factor |
| 2 | `capture` | log(1 + cells its loop closure would capture) / 5; 0 if the move closes no loop |
| 3 | `home_change` | +1 / 0 / -1: the move brings it nearer its own land (taxicab) or not |
| 4 | `pressure` | trail / (trail + 6): how strongly a long trail pulls it home |
| 5 | `chase_change` | +1 / 0 / -1: nearer my trail, counted only while my trail is within 5 cells of it |
| 6 | `momentum` | 1 if the move repeats its last direction (while it has a trail) |
| 7 | `explore` | 1 if the move leaves its land while it stands on it |
| 8 | `enemy_home` | its taxicab distance home after the move / 50: the enemy-return-home factor |
| 9 | `my_home` | my BFS distance (avoiding my trail) to my land as this hypothesis has it / 50: the I-return-home factor |
| 10 | `to_my_trail` | its distance to my nearest trail cell after the move / 50 |
| 11 | `enemy_kill_me` | (my_home - its distance to my trail) / 25, clipped to +-1; positive means it can reach my trail before I am home |
| 12 | `to_its_trail` | my distance to its nearest trail cell after the move, its trail plus the cell it steps onto / 50 |
| 13 | `i_kill_enemy` | (its distance home - my distance to its trail) / 25, clipped; positive means I can cut its trail before it is home |
| 14 | `enemy_safety` | the same race from its side, as far as it can know: if it can see me, (my distance to its trail - its distance home) / 25 clipped; if not, 1 |
| 15 | `sees_me` | 1 if I am inside its 11 x 11 view after the move |
| 16 | `return_gain` | log(1 + cells it would capture by going straight home now) / 5: its trail, an L-shaped path to the nearest cell of its land, and what they enclose |

All distances are capped at 50, which also means "none".

### 2.2 The two K models

| Model | Features used | How they combine | When used |
|---|---|---|---|
| Hand-set K (`motion.FeatureMotionModel`) | 0-7, as seven features (`home_change` x `pressure` is one feature, `home_pull`) | a weighted sum with 9 hand-set weights (`MotionWeights`) | the default in a real match and, now, in training games |
| Neural K (`neural.NeuralMotionModel`) | all 17 | a 17 -> 32 -> 32 -> 1 multilayer perceptron with ReLU, 1,665 parameters, run in NumPy | optional (`CANDIDATE_K_WEIGHTS`); trained each epoch by `offline/k_train.py` |

The hand-set weights: `self_trail` -6.0, `kills_my_trail` 2.0, `capture` 0.25, `home_pull` 1.0 (with
`home_pressure_half` 6), `chase` 1.0 (`view_radius` 5), `momentum` 1.2, `explore` 0.4.

### 2.3 How the features are computed efficiently

- **Lands by id.** A hypothesis stores an id into a table of distinct land versions, so the taxicab map to the
  enemy's land is computed once per distinct land (`_taxicab_maps`), not once per hypothesis.
- **My way home per view of my land** (`_my_home_distance`): one BFS per distinct view.
- **My distance to each trail** (`_distance_to_sets`): a trail is a few cells, so only its non-empty row words are
  examined, with bit arithmetic for the nearest set bit on each side of my column.
- **Return gain** (`_return_gain`): the straight part (trail + L-path) for every hypothesis; the enclosed part by an
  exact flood fill for the 256 most likely hypotheses with a trail (`GAIN_EXACT`), a lower bound for the rest.

### 2.4 A known gap

Feature 16 is computed differently in training and in play: in training (`offline/k_train.py`) each row is the
enemy's one true state, so the gain is always exact; in play only 256 of about 3,300 hypotheses get the exact value.
That mismatch is the best current explanation for why the neural K predicts the enemy's moves better (log-likelihood
about -0.55 against -0.84 for the hand-set K) but its belief plays worse (about -400 cells a game on the gate's 16
games). The fix is to compute the feature identically in both places and retrain K.

---

## Where each piece lives

| Piece | File | Function |
|---|---|---|
| Q's three inputs | `agent/q/inputs.py` | `build_inputs` |
| the grid | `agent/q/grid.py` | `build_grid`, `rotate` |
| the scalars | `agent/q/features.py` | `build_scalars`, `NAMES` |
| the probability map | `agent/q/probmap.py` | `build_probmap`, `pool` |
| the world map | `agent/behavior/worldmap.py` | `world_map`, `probability_board`, `reach_region`, `confidence` |
| the votes | `agent/behavior/votes.py` | `enemy_land_votes`, `enemy_trail_votes`, `my_land_votes` |
| D1 to D8 and the safety values | `agent/behavior/snapshot.py` | `build_snapshot`, `_enemy_gain` |
| the thresholds and scales | `agent/behavior/config.py`, `agent/q/config.py` | `BehaviorConfig`, `QConfig` |
| K's move outcomes | `agent/belief/generate.py` | `classify` |
| K's features and the neural K | `agent/belief/neural.py` | `features`, `NeuralMotionModel` |
| the hand-set K and the softmax | `agent/belief/motion.py` | `FeatureMotionModel`, `MotionWeights`, `move_probabilities` |
