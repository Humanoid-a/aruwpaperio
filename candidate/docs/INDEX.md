# Project File Guide

This repository is an ARUW Software Vision interview project. All project notes live in `candidate/docs/`; this file is the entry point. The root `FileGuides.md` only redirects here.

## Working rules

- **Stay inside `candidate/`.** Everything we add or change lives under `candidate/`: code in `candidate/agent/` and `candidate/offline/`, tests in `candidate/tests/`, notes in `candidate/docs/`. Nothing outside it is modified: not the framework (`territorygame/`), not `tests/`, not `requirements.txt`, `README.md`, `CANDIDATE_GUIDE.md`, or the PDFs.
- **The user directs the work.** [`method1.0.pdf`](../../method1.0.pdf) is a design reference only. Its build order, milestones, and other development-process guidance are not followed; what to build, and in what order, comes from the user's instructions.

## Authority order

1. [`ARUW_Vision_Interview__27_Question.pdf`](../../ARUW_Vision_Interview__27_Question.pdf) — source of truth for the task, deliverables, game rules, and evaluation criteria.
2. [`method1.0.pdf`](../../method1.0.pdf) — design reference: belief filter, explicit state machine, plan scoring, and model fitting. It is the only design reference; earlier exploratory notes were removed because they contradicted it.
3. [`CANDIDATE_GUIDE.md`](../../CANDIDATE_GUIDE.md) — Python controller API and the candidate editing boundary.
4. Source code and tests — exact behavior of this Python implementation.

Split of responsibility: **method1.0 informs the design; the guide, source, and tests decide framework facts.** method1.0 states that its framework assumptions came from the booklet alone, so where it disagrees with the guide or the code on a framework fact, the guide or code wins. [`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md) lists the known disagreements.

## Working references

- [`ARCHITECTURE_GUIDE.md`](ARCHITECTURE_GUIDE.md) — **start here**: a guided tour of the whole design, what each part does and why, how it is verified, and likely interview questions.
- [`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md) — source-verified game facts, move lifecycle, constraints, method1.0 reconciliation, and current status.
- [`Q_BUILD_PLAN.md`](Q_BUILD_PLAN.md) — the build plan, decisions and status of the Q model and the five action-states, from the brief [`Q_Model.pdf`](Q_Model.pdf).
- [`Q_AND_K_INPUTS.md`](Q_AND_K_INPUTS.md) — every input of Q (the grid, the 23 scalars, the probability map) and of K (the 17 per-hypothesis features), how each is calculated, and where it lives in the code.
- [`CODEBASE_INDEX.md`](CODEBASE_INDEX.md) — task-oriented map of the implementation and tests, plus where our code goes.
- [`README.md`](../../README.md) — setup, execution, and high-level repository structure (upstream starter doc).

Keep experimental conclusions separate from requirements, and label unmeasured strategy ideas as hypotheses.
