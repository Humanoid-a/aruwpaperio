"""The opponent motion model K as a small neural network, over features computed for every hypothesis on its own.

The hand-set K (FeatureMotionModel) is a weighted sum of seven features; it cannot express interactions ("go home if
the trail is long AND the enemy is near"). NeuralMotionModel scores each (hypothesis, move) with a multilayer
perceptron (FEATURES -> 32 -> 32 -> 1, ReLU) and the belief turns the scores into probabilities exactly as before
(a softmax over the legal moves, mixed with a floor, so no legal move is ever impossible). It runs in NumPy: the belief
never needs PyTorch. It is trained offline (candidate/offline/k_train.py) on what real opponents did.

Every feature is computed per hypothesis, from that hypothesis' own position, trail and land (and its view of my land),
after the move being scored. Nothing is shared between hypotheses except my own trail and position, which are known.

     0  self_trail        the move steps onto its own trail (it dies)
     1  kills_me          the move steps onto my trail: the enemy-kill factor (I die now)
     2  capture           log(1 + cells the move's loop closure would capture) / 5
     3  home_change       +1 / 0 / -1: the move brings it nearer its own land (taxicab), or not
     4  pressure          trail / (trail + 6): how much a long trail should pull it home
     5  chase_change      +1 / 0 / -1: nearer my trail, while my trail is within its view distance (5)
     6  momentum          the move repeats its last direction (while it has a trail)
     7  explore           the move leaves its land while it stands on it
     8  enemy_home        its distance home after the move (taxicab to its land) / 50: the enemy-return-home factor
     9  my_home           my distance home (avoiding my trail) to my land as this hypothesis has it / 50: the
                          I-return-home factor
    10  to_my_trail       its distance to my nearest trail cell after the move / 50
    11  enemy_kill_me     (my_home - its distance to my trail) / 25, clipped to +-1: > 0 means it could reach my trail
                          before I am home -- the enemy-kills-me factor
    12  to_its_trail      my distance to its nearest trail cell after the move (its trail plus the cell it steps onto) / 50
    13  i_kill_enemy      (its distance home - my distance to its trail) / 25, clipped: > 0 means I could cut its trail
                          before it is home -- the I-kill-enemy factor
    14  enemy_safety      the same race from its side, as it can know it: if it can see me,
                          (my distance to its trail - its distance home) / 25 clipped; if not, 1 (as far as it can tell
                          it is safe)
    15  sees_me           my agent is inside its 11 x 11 view after the move
    16  return_gain       log(1 + cells it would capture by going straight home now) / 5: the trail, the way home
                          (an L-shaped path to the nearest cell of its land: along the column, then along the row), and
                          what they enclose. The enclosed part is exact for the GAIN_EXACT most likely hypotheses that
                          have a trail out; the others get the trail and the way home only (a lower bound), because an
                          exact capture costs about 27 microseconds a board

Distances are clipped to 50 (also "none").
"""

import json
from pathlib import Path

import numpy as np

from candidate.agent.belief.motion import MotionModel, SuccessorBatch
from candidate.agent.config import BOARD_HEIGHT, BOARD_WIDTH, CELL_COUNT
from candidate.agent.geometry import bitboard
from candidate.agent.geometry.distance import UNREACHABLE, bfs_distance, taxicab_distance
from candidate.agent.geometry.fill import BOARD, holes

FEATURES = ("self_trail", "kills_me", "capture", "home_change", "pressure", "chase_change", "momentum", "explore",
            "enemy_home", "my_home", "to_my_trail", "enemy_kill_me", "to_its_trail", "i_kill_enemy", "enemy_safety",
            "sees_me", "return_gain")
N_FEATURES = len(FEATURES)
HIDDEN = (32, 32)
CAP = 50
VIEW = 5                        # the opponent sees 11 x 11: 5 cells each way
GAIN_EXACT = 256                # hypotheses with an exact return gain per call (the most likely ones with a trail)
_DY = np.array([-1, 0, 1, 0])
_DX = np.array([0, 1, 0, -1])


def _taxicab_maps(lands: np.ndarray) -> np.ndarray:
    """(k, 2500) int64: taxicab distance to each land, clipped to CAP (CAP where a land is empty)."""
    if lands.shape[0] == 0:
        return np.zeros((0, CELL_COUNT), dtype=np.int64)
    maps = taxicab_distance(lands).reshape(lands.shape[0], CELL_COUNT).astype(np.int64)
    return np.minimum(maps, CAP)


def _my_home_distance(lands: np.ndarray, my_pos: int, my_trail: np.ndarray) -> np.ndarray:
    """(k,) my distance home, avoiding my trail, to each of k views of my land."""
    passable = BOARD & ~my_trail
    out = np.full(lands.shape[0], CAP, dtype=np.int64)
    y, x = divmod(my_pos, BOARD_WIDTH)
    for i, land in enumerate(lands):
        if bitboard.has_cell(land, my_pos):
            out[i] = 0
            continue
        targets = land & passable
        if not targets.any():
            continue
        dist = bfs_distance(targets, passable).reshape(CELL_COUNT)
        best = UNREACHABLE
        for dy, dx in zip(_DY, _DX):
            ny, nx = y + dy, x + dx
            if 0 <= ny < BOARD_HEIGHT and 0 <= nx < BOARD_WIDTH:
                best = min(best, int(dist[ny * BOARD_WIDTH + nx]))
        out[i] = min(best + 1, CAP)
    return out


def _top_bit(words: np.ndarray) -> np.ndarray:
    """The index of the highest set bit of each positive word (exact: the words are below 2^50 < 2^53)."""
    return np.frexp(words.astype(np.float64))[1] - 1


def _distance_to_sets(origin: int, sets: np.ndarray) -> np.ndarray:
    """(n,) taxicab distance from `origin` to the nearest cell of each bitboard in `sets`, clipped to CAP (CAP if
    empty). The sets are trails, a few cells each: only their non-empty row words are looked at, and in each the
    nearest set bit on either side of the origin's column is found with bit arithmetic."""
    n = sets.shape[0]
    out = np.full(n, CAP, dtype=np.int64)
    hyp, row = np.nonzero(sets)
    if hyp.size == 0:
        return out
    words = sets[hyp, row]
    oy, ox = divmod(int(origin), BOARD_WIDTH)
    far = 2 * BOARD_WIDTH
    right = words >> ox                                       # columns >= ox, the origin's column at bit 0
    lowest = right & -right
    to_right = np.where(right != 0, _top_bit(np.where(right != 0, lowest, 1)), far)
    left = words & ((np.int64(2) << ox) - 1)                  # columns <= ox
    to_left = np.where(left != 0, ox - _top_bit(np.where(left != 0, left, 1)), far)
    d = np.minimum(to_right, to_left) + np.abs(row - oy)
    starts = np.flatnonzero(np.r_[True, hyp[1:] != hyp[:-1]])  # np.nonzero lists each hypothesis' words together
    out[hyp[starts]] = np.minimum(np.minimum.reduceat(d, starts), CAP)
    return out


def _l_paths(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """(m, 50) bitboards: from each start, along its column to the goal's row, then along that row to the goal; the
    start and the goal themselves left out."""
    sy, sx = np.divmod(start, BOARD_WIDTH)
    gy, gx = np.divmod(goal, BOARD_WIDTH)
    rows = np.arange(BOARD_HEIGHT)[None, :]
    lo, hi = np.minimum(sy, gy)[:, None], np.maximum(sy, gy)[:, None]
    column = np.where((rows >= lo) & (rows <= hi), np.left_shift(np.int64(1), sx.astype(np.int64))[:, None], 0)
    a, b = np.minimum(sx, gx), np.maximum(sx, gx)
    span = np.left_shift(np.int64(1), (b + 1).astype(np.int64)) - np.left_shift(np.int64(1), a.astype(np.int64))
    row = np.where(rows == gy[:, None], span[:, None], 0)
    path = column | row
    path[np.arange(start.size), sy] &= ~np.left_shift(np.int64(1), sx.astype(np.int64))
    path[np.arange(start.size), gy] &= ~np.left_shift(np.int64(1), gx.astype(np.int64))
    return path


def _return_gain(batch: SuccessorBatch, land_index: np.ndarray, lands: np.ndarray) -> np.ndarray:
    """(n,) cells each hypothesis would capture by going straight home now (see the module notes)."""
    n = batch.n
    gain = np.zeros(n, dtype=np.int64)
    out = np.flatnonzero(batch.trail_len > 0)
    if out.size == 0:
        return gain
    goal = np.full(out.size, -1, dtype=np.int64)
    pos = batch.pos[out].astype(np.int64)
    py, px = np.divmod(pos, BOARD_WIDTH)
    for k in np.unique(land_index[out]):                               # the nearest cell of each hypothesis' land
        members = np.flatnonzero(land_index[out] == k)
        cells = np.asarray(bitboard.to_cells(lands[k]), dtype=np.int64)
        if cells.size == 0:
            continue
        cy, cx = np.divmod(cells, BOARD_WIDTH)
        at, which = np.unique(pos[members], return_inverse=True)       # many hypotheses share a cell: once per cell
        ay, ax = np.divmod(at, BOARD_WIDTH)
        d = np.abs(ay[:, None] - cy[None, :]) + np.abs(ax[:, None] - cx[None, :])
        goal[members] = cells[d.argmin(axis=1)][which.reshape(-1)]
    has = goal >= 0
    own = lands[land_index[out]]
    path = np.zeros((out.size, BOARD_HEIGHT), dtype=np.int64)
    if has.any():
        path[has] = _l_paths(pos[has], goal[has])
    closed = (batch.trail_bb[out] | path) & ~own
    gain[out] = bitboard.popcount(closed)
    order = np.argsort(-batch.weight[out], kind="stable")
    top = order[has[order]][:GAIN_EXACT]
    if top.size:                     # capture_region, with the holes of each distinct land filled once
        top_lands, which = np.unique(land_index[out[top]], return_inverse=True)
        before = holes(lands[top_lands])[which.reshape(-1)]
        gain[out[top]] += bitboard.popcount(holes(own[top] | closed[top]) & ~before)
    return gain


def features(batch: SuccessorBatch) -> np.ndarray:
    """(n, 4, N_FEATURES) float32, in FEATURES order (see the module notes)."""
    n = batch.n
    pos = batch.pos.astype(np.int64)
    dest = batch.dest.astype(np.int64)
    trail = batch.trail_len.astype(np.float64)
    sit = batch.situation
    out = np.zeros((n, 4, N_FEATURES), dtype=np.float32)
    if n == 0:
        return out

    # its land, per hypothesis (one taxicab map per distinct land)
    if sit is not None:
        land_ids, land_index = np.unique(batch.opp_land_id, return_inverse=True)
        lands = sit.opp_diffs.territory(land_ids, sit.opp_base)
    else:
        land_index = np.zeros(n, dtype=np.int64)
        lands = batch.opp_land_bb[None, :]
    land_index = land_index.reshape(-1)
    land_maps = _taxicab_maps(lands)
    home_now = land_maps[land_index, pos]                                 # (n,)
    home_after = land_maps[land_index[:, None], dest]                      # (n, 4)

    # my side: my trail (known), my way home to my land as each hypothesis has it
    my_trail_map = np.minimum(taxicab_distance(batch.my_trail_bb).reshape(CELL_COUNT).astype(np.int64), CAP)
    if sit is not None:
        my_ids, my_index = np.unique(batch.my_land_id, return_inverse=True)
        my_home = _my_home_distance(sit.my_diffs.territory(my_ids, sit.my_base), batch.my_pos, batch.my_trail_bb)
        my_home = my_home[my_index.reshape(-1)]
    else:
        my_home = np.full(n, CAP, dtype=np.int64)
    to_my_trail = my_trail_map[dest]                                       # (n, 4)

    # my distance to its trail after the move: the trail it has, plus the cell it steps onto if that stays out
    to_trail_now = _distance_to_sets(batch.my_pos, batch.trail_bb)        # (n,)
    my_y, my_x = divmod(batch.my_pos, BOARD_WIDTH)
    dy, dx = np.divmod(dest, BOARD_WIDTH)
    to_dest = np.abs(dy - my_y) + np.abs(dx - my_x)
    stays_out = ~batch.enters_own & ~batch.self_trail
    to_its_trail = np.where(stays_out, np.minimum(to_trail_now[:, None], to_dest), CAP)
    sees_me = np.maximum(np.abs(dy - my_y), np.abs(dx - my_x)) <= VIEW
    gain = _return_gain(batch, land_index, lands)

    clip = lambda v: np.clip(v / 25.0, -1.0, 1.0)  # noqa: E731
    out[..., 0] = batch.self_trail
    out[..., 1] = batch.kills_me
    out[..., 2] = np.log1p(batch.capture_gain.astype(np.float64)) / 5.0
    out[..., 3] = np.clip(home_now[:, None] - home_after, -1, 1)
    out[..., 4] = (trail / (trail + 6.0))[:, None]
    in_view = (my_trail_map[pos] <= VIEW)[:, None]
    out[..., 5] = np.clip(my_trail_map[pos][:, None] - to_my_trail, -1, 1) * in_view
    out[..., 6] = (batch.last_dir[:, None] == np.arange(4)[None, :])
    out[..., 7] = (batch.trail_len == 0)[:, None] & ~batch.enters_own
    out[..., 8] = home_after / CAP
    out[..., 9] = (my_home / CAP)[:, None]
    out[..., 10] = to_my_trail / CAP
    out[..., 11] = clip(my_home[:, None] - to_my_trail)
    out[..., 12] = to_its_trail / CAP
    out[..., 13] = clip(home_after - to_its_trail)
    out[..., 14] = np.where(sees_me, clip(to_its_trail - home_after), 1.0)
    out[..., 15] = sees_me
    out[..., 16] = (np.log1p(gain) / 5.0)[:, None]
    return out


class NeuralMotionModel(MotionModel):
    """K as a multilayer perceptron over `features`. `layers` is [(W, b), ...]; ReLU between them, none after the last."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]], meta: dict | None = None) -> None:
        self.layers = [(np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32)) for w, b in layers]
        if self.layers[0][0].shape[0] != N_FEATURES or self.layers[-1][0].shape[1] != 1:
            raise ValueError(f"a K network takes {N_FEATURES} features and gives 1 score")
        self.meta = dict(meta or {})
        self.name = self.meta.get("name", "neural K")

    def scores_of(self, f: np.ndarray) -> np.ndarray:
        """Scores for features of any leading shape (..., N_FEATURES)."""
        h = f.reshape(-1, N_FEATURES).astype(np.float32)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h.reshape(f.shape[:-1]).astype(np.float64)

    def score(self, batch: SuccessorBatch) -> np.ndarray:
        return self.scores_of(features(batch))

    @classmethod
    def initial(cls, seed: int = 0) -> "NeuralMotionModel":
        """Small random weights (a nearly uniform K), the start of training."""
        rng = np.random.default_rng(seed)
        sizes = (N_FEATURES,) + HIDDEN + (1,)
        layers = [(rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32) * (0.1 if b == 1 else 1.0),
                   np.zeros(b, dtype=np.float32)) for a, b in zip(sizes, sizes[1:])]
        return cls(layers, dict(name="untrained"))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"w{i}": w for i, (w, _) in enumerate(self.layers)} | {f"b{i}": b for i, (_, b) in enumerate(self.layers)}
        np.savez(path, meta=np.array(json.dumps(dict(self.meta, features=list(FEATURES)))), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NeuralMotionModel":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if tuple(meta.get("features", ())) != FEATURES:
                raise ValueError(f"{path} was trained on other features ({meta.get('features')})")
            count = sum(1 for k in z.files if k.startswith("w"))
            layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(count)]
        return cls(layers, meta)


def load_motion(path: str | Path | None, echo=None) -> MotionModel:
    """The K to use: the neural one in `path`, or the hand-set FeatureMotionModel if there is none or it will not load."""
    from candidate.agent.belief.motion import FeatureMotionModel

    if path is None:
        return FeatureMotionModel()
    try:
        return NeuralMotionModel.load(path)
    except Exception as error:  # noqa: BLE001 - a bad K file must not stop the controller
        if echo is not None:
            echo(f"[candidate] could not load the motion model {path}: {error!r}; using the hand-set K")
        return FeatureMotionModel()
