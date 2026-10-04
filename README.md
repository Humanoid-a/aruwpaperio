# Territory Capture (Python)

A local two-player territory-capture game, built as a Python framework
for an interview assessment: the framework owns the board, rules, turn
execution, and GUI; a candidate implements one file — an
`AgentController` — to play.

Two agents move around a grid, laying trails outside their own territory
and closing them to capture the enclosed area (and any opponent territory
inside it). Crossing your own trail kills you; crossing an opponent's
trail kills them. Whoever holds more territory when both players run out
of turns wins.

This is a Python port of the original Java version — same game, same
rules, hand-translated. The two repos aren't kept in sync going forward,
so they may drift over time; the GUI in particular was rebuilt for
pygame rather than translated line-for-line, so don't expect it to look
identical.

## Requirements

- Python 3.11+

## Getting started

### 1. Check your Python version

```
python3 --version
```

If it's older than 3.11, install a newer Python from
[python.org](https://www.python.org/downloads/) or your platform's
package manager.

### 2. Clone the repository

```
git clone https://gitlab.com/aruw/vision/applications/python-state-machine-starter.git
cd python-state-machine-starter
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

This installs `pygame` (for the GUI) and `pytest` (for the test suite).
No other setup is needed.

### 4. Run it

```
python -m territorygame.main
```

In the window, click a player's controller label to cycle through the
available options (Basic State Machine, Enemy State Machine, Random
State Machine, or the candidate's own controller), then use
Start / Pause / Step / Reset to run a match. Faster / Slower adjust the
pause between turns during continuous play.

## Candidate assessment

See `CANDIDATE_GUIDE.md` for the assessment task: the file to edit, the
rules from a player's perspective, the API reference, and some tips.

## Project structure

```
territorygame/resources/
  game-config.properties       board size, visibility, turn count, respawn
                                positions, starting-territory size (same
                                file, format, and values as the Java repo)

territorygame/
  api/          Candidate-facing types: GameApi, AgentController,
                GridPosition, VisibleCell, OccupantView, TerritoryView,
                Direction, MoveResult. Nothing outside this package is
                ever handed to candidate code.

  domain/       Authoritative game state: PlayerId, GameConfig, Agent,
                Player, Board, BoardCell, GameState. Not exposed to
                candidates or the GUI.

  rules/        The actual rules, isolated from turn management:
                MoveResolver (one move's resolution order),
                TerritoryResolver (flood-fill capture), RespawnService
                (death/reset).

  visibility/   VisibilityService — builds a candidate's visible-cell
                window and translates internal player identities to
                SELF_*/OPPONENT_* types.

  engine/       Match orchestration: GameEngine (lifecycle, Start/Pause/
                Step/Reset), TurnManager (one turn, controller retry-on-
                invalid), GameApiImpl (per-player facade over GameApi),
                GameSnapshot/GameObserver (the read-only view the GUI
                renders from).

  helpers/      Provided utilities candidates may use: ObservedBoard
                (remembers the latest observed value per cell),
                MovementUtils (position/bounds arithmetic, mechanical
                move validation, Manhattan distance).

  controller/   Framework-internal AgentController implementations not
                meant as examples to copy: EnemyStateMachine (the
                standard assessment opponent) and AvailableControllers
                (the registry the GUI's controller picker reads from —
                Basic State Machine, Enemy State Machine, Random State
                Machine, Candidate Controller).

  gui/          pygame viewer: game_window.py (controls, status, wiring)
                and board_renderer.py (board painting). No game-rule
                logic lives here.

  main.py       Entry point — launches the pygame window.

candidate/
  candidate_controller.py   the file to edit for the assessment.

  examples/                 Read-only reference controllers:
                             basic_state_machine.py (a deliberately weak
                             state-machine example) and
                             random_state_machine.py (the simplest
                             possible baseline) are not templates for a
                             good strategy.

tests/                       mirrors the territorygame/candidate package layout.
```
