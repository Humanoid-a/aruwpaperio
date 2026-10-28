"""A direct run of the game, without a display: the real GameWindow and its own event loop, driven by real clicks.

    python -m candidate.offline.headless --turns 2000 --seat 0 --opponent "Enemy State Machine"

This is `python -m territorygame.main` with the screen replaced by SDL's dummy video driver. The clicks a person would
make are posted as real pygame mouse events, in this order: the player picker of the candidate's seat is clicked until
it reads "Candidate Controller" and the other one until it reads the opponent, then Reset; once the engine has
published the fresh board, Start (Reset and Start cannot go in the same frame: the reset runs on the engine's thread and
would stop the run it just started). When the game is over the window is closed with a QUIT event.

The Candidate Controller is built by the GUI's own registry, exactly as when you play, so it records its game the way it
does in the GUI. An observer on the engine keeps, from the fresh board on, every published snapshot's counts and
positions: the truth the recording is checked against (verify_record).
"""

import argparse
import dataclasses
import os
import sys
import threading
import time
from pathlib import Path

from candidate.agent.q.record import load_record
from candidate.offline import locations

CANDIDATE = "Candidate Controller"


class _Truth:
    """GameObserver: after the reset, (counts, positions, remaining turns, game over) of every published board."""

    def __init__(self) -> None:
        self.boards: list[dict] = []
        self.armed = False
        self.fresh = threading.Event()
        self.over = threading.Event()

    def on_game_state_changed(self, snapshot) -> None:
        if not self.armed:
            return
        players = snapshot.players
        board = dict(counts=[p.territory_count for p in players], positions=[(p.position.y, p.position.x) for p in players],
                     remaining=[p.remaining_turns for p in players], deaths=[p.death_count for p in players],
                     over=snapshot.game_over, error=snapshot.error_message)
        if not self.boards and board["remaining"][0] != board["remaining"][1]:
            return                                  # a turn of the previous match still being published
        self.boards.append(board)
        self.fresh.set()
        if snapshot.game_over:
            self.over.set()


def _click(pygame, rect) -> None:
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=rect.center, button=1))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=rect.center, button=1))


def run_window(turns: int = 2000, seat: int = 0, opponent: str = "Enemy State Machine", record_dir: str | Path | None = None,
               timeout: float = 3600.0) -> dict:
    """Play one game in the real window with no display. Returns the truth boards, the candidate controller, the window's
    final state and the path of the recorded game."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    import candidate.candidate_controller as candidate_controller
    from territorygame.controller.available_controllers import ALL
    from territorygame.domain.game_config import GameConfig
    from territorygame.gui.game_window import GameWindow

    labels = [option.label for option in ALL]
    if opponent not in labels:
        raise ValueError(f"unknown opponent {opponent!r}; the GUI offers {labels}")
    if record_dir is not None:
        candidate_controller.RECORD_DIR = Path(record_dir)   # where the registry's CandidateController records

    pygame.init()
    config = dataclasses.replace(GameConfig.load_default(), turns_per_player=turns, autoplay_turn_delay_millis=0)
    window = GameWindow(config)
    truth = _Truth()
    window._engine.add_observer(truth)
    wanted = {seat: CANDIDATE, 1 - seat: opponent}
    failure: list[str] = []

    def drive() -> None:
        try:
            for slot, picker in enumerate(window._player_pickers):
                clicks = (labels.index(wanted[slot]) - picker.index) % len(labels)
                for _ in range(clicks):
                    _click(pygame, picker.rect)
            time.sleep(0.5)                                    # let the window's loop handle the picker clicks
            labels_now = [p.label for p in window._player_pickers]
            if labels_now != [f"Player: {wanted[0]}", f"Player: {wanted[1]}"]:
                failure.append(f"the pickers read {labels_now}")
                return
            truth.armed = True
            _click(pygame, next(b for b in window._buttons if b.label == "Reset").rect)
            if not truth.fresh.wait(60):
                failure.append("the engine did not publish the fresh board after Reset")
                return
            _click(pygame, next(b for b in window._buttons if b.label == "Start").rect)
            if not truth.over.wait(timeout):
                failure.append(f"the game did not end within {timeout} s")
        finally:
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    driver = threading.Thread(target=drive, name="headless-driver", daemon=True)
    started = time.perf_counter()
    driver.start()
    window.run()                                           # the GUI's own loop, until the QUIT above
    driver.join(5)
    if failure:
        raise RuntimeError("; ".join(failure))
    controller = window._engine._controllers_in_player_order[seat]
    strategy = controller._agent.strategy
    strategy.close()                                       # save the game if the last turn did not already
    recorder = strategy.recorder
    return dict(truth=truth.boards, seat=seat, opponent=opponent, turns=turns, controller=controller,
                controllers=list(window._engine._controllers_in_player_order),
                record=recorder.last_path if recorder is not None else None,
                record_error=recorder.last_error if recorder is not None else None,
                seconds=time.perf_counter() - started)


def verify_record(record_path: str | Path, truth: list[dict], seat: int) -> list[str]:
    """Problems with a recorded game, checked against the engine's published boards; empty if it is right.

    Frame t (my t-th turn) was recorded when 2t + seat turns had been played, so it must show board 2t + seat's
    counts; the move recorded for it must be the step my agent made on the next board (unless it died or respawned)."""
    meta, a = load_record(record_path)
    problems = []
    n = len(a["turn"])
    total = meta["total_turns"]
    if not meta["complete"]:
        problems.append("the game is not marked complete")
    if n != total or list(a["turn"]) != list(range(total)):
        problems.append(f"{n} frames recorded for {total} turns")
    steps = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    for i, t in enumerate(a["turn"]):
        k = 2 * int(t) + seat
        if k + 1 >= len(truth):
            problems.append(f"no board for turn {t}")
            break
        board, after = truth[k], truth[k + 1]
        if (int(a["my_count"][i]), int(a["opp_count"][i])) != (board["counts"][seat], board["counts"][1 - seat]):
            problems.append(f"turn {t}: recorded counts {int(a['my_count'][i])}/{int(a['opp_count'][i])}, the board had "
                            f"{board['counts'][seat]}/{board['counts'][1 - seat]}")
        (y0, x0), (y1, x1) = board["positions"][seat], after["positions"][seat]
        step = steps.get((y1 - y0, x1 - x0))
        if after["deaths"][seat] == board["deaths"][seat] and step is not None and step != int(a["direction"][i]):
            problems.append(f"turn {t}: recorded move {int(a['direction'][i])}, the agent moved {step}")
        if len(problems) > 10:
            break
    for key in ("grid_bits", "scalars", "probmap", "values", "valid"):
        if a[key].shape[0] != n:
            problems.append(f"{key} has {a[key].shape[0]} rows for {n} frames")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--turns", type=int, default=2000)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--opponent", default="Enemy State Machine")
    parser.add_argument("--record-dir", default=str(locations.RAW / "gui"))
    args = parser.parse_args(argv)
    result = run_window(args.turns, args.seat, args.opponent, args.record_dir)
    final = result["truth"][-1]
    seat = args.seat
    print(f"played {args.turns} turns as seat {seat} against {args.opponent} in {result['seconds']:.0f}s: "
          f"final {final['counts'][seat]} vs {final['counts'][1 - seat]}, my deaths {final['deaths'][seat]}")
    if result["record"] is None:
        print(f"NO RECORD written ({result['record_error']})")
        return 1
    problems = verify_record(result["record"], result["truth"], seat)
    print(f"record: {result['record']}")
    print("RECORD OK: every frame matches the engine" if not problems else "RECORD PROBLEMS:\n  " + "\n  ".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
