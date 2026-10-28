# Prompt: trim this project to the run pipeline for submission

Copy everything below the line into a fresh Claude Code session opened in the copied project. Fill in the two
choices first.

---

## Context

This repository is a competitive territory-capture game. The framework under `territorygame/` is the organiser's
and must not be changed. Our agent lives under `candidate/`; the framework loads it through
`candidate/candidate_controller.py` (`CandidateController`).

The agent has three layers, all under `candidate/agent/`:

- `belief/` — a particle-style belief over where the unseen enemy is, with a motion model K (`motion.py` hand-set,
  `neural.py` a small NumPy network).
- `behavior/` — six hand-written action states (EXPAND, EXPAND2, RETURN, HUNT, TRACK, PATROL), a chooser and safety
  rules.
- `q/` — a value model Q that picks which state runs each turn: a PyTorch network (`network.py`) or a hand-set
  stand-in (`heuristic.py`) when no network is configured.

Everything else under `candidate/` is training, simulation, testing and analysis tooling that a real match never
uses. The goal is to remove all of it so that only the run pipeline remains, without changing how a match plays.

**Verified fact to rely on:** instantiating `CandidateController` and playing a match imports exactly these packages:
`candidate/candidate_controller.py` and every module under `candidate/agent/` (its `belief/`, `behavior/`, `q/`,
`data/`, `geometry/`, `rules/` subpackages plus `adapter.py`, `runtime.py`, `controller.py`, `config.py`,
`diagnostics.py`, `prediction.py`, `probe.py`). It imports nothing from `candidate/offline/`, `candidate/tests/`,
`candidate/examples/` or `candidate/runtime_check.py`. Re-verify this yourself before deleting anything (step 1).

## Two choices to fill in before starting

- **CHOICE A — the agent to submit:** either `stand-in` (no network, no PyTorch needed at run time) or
  `network: <path to weights.pt> and <path to k.npz>` (the gated best network; currently
  `candidate/q_training/online/online15/epoch_025/`, the controller's default). Choice: ____________
- **CHOICE B — keep `candidate/docs/`?** yes / no. Choice: ____________

## Steps

1. **Verify the import boundary.** From the project root, in the project's Python environment, run a script that
   records `sys.modules` before and after `import candidate.candidate_controller` and
   `CandidateController(record=False)`, and lists every imported module starting with `candidate`. Confirm none
   starts with `candidate.offline`, `candidate.tests`, `candidate.examples` or `candidate.runtime_check`. If any
   does, stop and report it; do not delete anything.

2. **Work on a git branch** (`git checkout -b submission`), so nothing is lost. Do not delete on `main`.

3. **Apply CHOICE A in `candidate/candidate_controller.py`.** The two module-level settings `DEFAULT_WEIGHTS` and
   `DEFAULT_K_WEIGHTS` currently point into `candidate/q_training/...`, which will be deleted.
   - For `stand-in`: set both to `None`. The controller then plays the hand-set stand-in and never imports PyTorch.
   - For `network`: create `candidate/weights/`, copy the chosen `weights.pt` and `k.npz` there, and point both
     settings at those files (`Path(__file__).parent / "weights/weights.pt"` and `.../weights/k.npz`). Note in the
     final report that PyTorch is then required on the grader's machine.
   Also remove the commented-out alternative `DEFAULT_*` lines just above them.

4. **Turn off recording and per-match logging in the same file.** `RECORD_DIR` points into `candidate/q_training/`
   and `_LOG_DIR` into `candidate/runtime_logs/`; by default the GUI's own controller records every game to the
   first and logs to the second, recreating those folders on the grader's machine. Make the default `record` False
   (the line `record = diagnostics is None and strategy is None` becomes `record = False`), pass no `log_dir` to
   `Diagnostics` (so `log_dir=None`), and remove `RECORD_DIR`, `_LOG_DIR` and the now-unused `GameRecorder` import
   if nothing else in the file uses them. Update the module docstring where it describes recording and logging.

5. **Delete these, and nothing else:**
   - `candidate/offline/` (all training, simulation, scenarios, reports)
   - `candidate/tests/` (all tests and test rigs)
   - `candidate/examples/` (only used by the runtime check and tests)
   - `candidate/runtime_check.py`
   - `candidate/q_training/` (about 12 GB of runs, records and plots locally; a fresh clone has only the shipped networks)
   - `candidate/runtime_logs/`
   - every `__pycache__/` directory under `candidate/`
   - `candidate/docs/` only if CHOICE B is `no`
   Keep every file under `candidate/agent/`, including `probe.py`, `q/record.py`, `behavior/style.py` and
   `rules/transition.py` and `rules/reference.py`: they are imported by match-time code even where they are inert
   in a real match. Do not touch `territorygame/`, the root `tests/`, `requirements.txt`, `README.md` or any PDF.

6. **Check for dangling references** in the kept code: grep `candidate/agent/` and `candidate_controller.py` for
   `offline`, `runtime_check`, `q_training`, `runtime_logs`, `examples` and `tests`. Docstring mentions are fine to
   leave or trim; any live import or path use must be fixed. `candidate/agent/q/version.py` hashes a list of source
   files under `candidate/agent/` to fingerprint Q's inputs: confirm every file it lists still exists.

7. **Verify the trimmed tree runs.** In order:
   - `python -c "import candidate.candidate_controller as c; c.CandidateController(record=False)"` from the root.
   - If CHOICE A is `network`: confirm the console line reports the network loaded and does not print the warning
     about inputs of another code version.
   - Launch the game with `python -m territorygame.main` and play, or let it autoplay, at least one full match with
     our controller selected; it must finish with no exceptions and no new folders created under `candidate/`.
   - Confirm `candidate/` contains no `__pycache__`, `q_training` or `runtime_logs` afterwards.

8. **Report** what was deleted (with sizes), the two choices applied, the diff of `candidate_controller.py`, the
   match result, and anything you were unsure about. Do not commit; leave the branch for review.

## Boundaries

- Change behaviour nowhere except the three controller settings in steps 3 and 4. If a deletion would require
  editing a file under `candidate/agent/`, stop and report instead.
- Do not reformat, rename or "clean up" kept files.
- Do not run any training command (`candidate.offline.online`, `collect`, `train`) at any point; they are being
  removed, and training is run by the owner only.
