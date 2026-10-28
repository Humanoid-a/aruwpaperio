# Q model: probability-map input and network structure

This replaces the earlier 8-channel value/confidence encoding of the confidence map.

## Idea

The belief over the enemy is a weighted set of hypotheses (each a complete guess of the hidden world) plus residual mass (weight whose position is known but whose trail and territory history was pruned). Summarize the whole belief as a per-cell probability map and feed it to Q through its own CNN branch, independent of Q's other inputs.

For every cell, describe two mutually exclusive groups as probability layers that sum to 1:

- Ownership: free, mine, enemy.
- Trail: none, my trail, enemy trail. Two trails can never share a cell, because stepping on a trail kills its owner.

Add two position layers, each summing to 1 over the whole board:

- Enemy position: the belief's position marginal.
- My position: one-hot.

Why layers instead of a single value with a confidence: layers keep the fact that the categories exclude each other. A cell at 0.45 mine / 0.45 enemy / 0.10 free shows as a near coin flip between mine and enemy, not as free.

## Computing the layers

- Hypotheses: a layer's probability at a cell is the total weight of hypotheses that put that cell in that category. Each hypothesis picks exactly one category per group, so the layers sum to 1.
- Residual mass uses the reach region: a diamond whose radius is the number of turns since the enemy was last seen, centered on that sighting, minus the current view.
  - Ownership inside the reach region, by what the memory map says: free gives ½ free and ½ enemy; mine gives ½ mine and ½ enemy; enemy stays all enemy. Outside the reach region, all residual goes to the memory map's category.
  - Trail inside the reach region: ½ none and ½ enemy trail, except on my trail cells, where it is none. Outside the reach region: none.
- My trail is always exact (0 or 1). Enemy-trail probability is 0 on my trail cells.
- Visible cells override everything with a one-hot of what is observed.

## Confidence channels

For each cell and each group: confidence = 1 − entropy / log(3). It is 1 when the cell is certain and 0 for an even three-way split.

This is needed because pooling averages. A block that is half certainly mine and half certainly free averages to the same layers as a block of 50/50 guesses. Only the confidence channel tells them apart.

## Input tensor: 10 channels, pooled from 50 × 50 to 10 × 10 with 5 × 5 blocks

| Channel | Contents | Pooling |
|---|---|---|
| 0–2 | Ownership: free, mine, enemy | Mean |
| 3 | Ownership confidence | Mean |
| 4–6 | Trail: none, my trail, enemy trail | Mean |
| 7 | Trail confidence | Mean |
| 8 | Enemy position probability | Sum |
| 9 | My position | Sum |

Do the pooling while building features, not inside the network.

## Q network

Q runs every frame and outputs one value per action-state, in this order: EXPAND, RETURN, HUNT, TRACK TARGET, PATROL. Each value is the expected change in score difference over the next 50 turns if the controller enters that state now.

Three independent branches feed one head:

| Branch | Input | Layers | Output |
|---|---|---|---|
| Grid | 7 × 21 × 21 egocentric grid (existing input) | conv 3×3, 16 → conv 3×3, 32, stride 2 → conv 3×3, 32, stride 2 → conv 3×3, 32, stride 2 → flatten → dense 64 | 64 |
| Scalars | 22 scalar features (existing input) | dense 64 | 64 |
| Probability map | 10 × 10 × 10 pooled map | conv 3×3, 16 → conv 3×3, 32, stride 2 → conv 3×3, 32, stride 2 → flatten → dense 32 | 32 |
| Head | 160, concatenated | dense 64 → dense 5 | 5 |

All convolutions use padding 1. Every layer except the last uses ReLU. Spatial sizes run 21 → 21 → 11 → 6 → 3 in the grid branch and 10 → 10 → 5 → 3 in the probability-map branch.