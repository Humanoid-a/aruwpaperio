"""Where the training pipeline keeps its files: everything under candidate/q_training/.

    q_training/raw/<run>/        recorded games: <game id>.npz (the controller's game record) and, for games played
                                 headless, <game id>.json (the sidecar: move tapes, config, engine result).
                                 raw/gui/ is where the GUI's Candidate Controller records its games.
    q_training/datasets/<name>.npz   training sets built from raw games (inputs, actions, labels, game ids, weights)
    q_training/weights/<name>.pt     trained networks (agent.q.network's format, loaded by the controller) and
                                     <name>.json, the training report
    q_training/logs/             logs of collection, training and evaluation runs
    q_training/online/<run>/     online training runs (online.py): per epoch its games, its weights and a summary
    q_training/belief/           the belief's motion model K fitted to recorded games (motion_fit.py): the data
                                 (k_data.npz, git-ignored), the fitted weights and reports (k_fit.json, k_check.json)
    q_training/smoke/            scratch space of the online loop's smoke test (online.py smoke)

raw/, datasets/, logs/, online/ and smoke/ are git-ignored (large, reproducible), except the networks that ship, which
q_training/.gitignore lists (the submitted controller loads online15's epoch 25); weights/ is tracked.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "q_training"
RAW = ROOT / "raw"
DATASETS = ROOT / "datasets"
WEIGHTS = ROOT / "weights"
LOGS = ROOT / "logs"
SMOKE = ROOT / "smoke"
ONLINE = ROOT / "online"
BELIEF = ROOT / "belief"


def raw_run(name: str) -> Path:
    return RAW / name


def dataset_path(name_or_path: str | Path) -> Path:
    """A dataset given by name (looked up in datasets/) or by path."""
    p = Path(name_or_path)
    if p.suffix == ".npz" or p.parent != Path("."):
        return p
    return DATASETS / f"{p.name}.npz"


def weights_path(name_or_path: str | Path) -> Path:
    """Weights given by name (looked up in weights/) or by path."""
    p = Path(name_or_path)
    if p.suffix == ".pt" or p.parent != Path("."):
        return p
    return WEIGHTS / f"{p.name}.pt"
