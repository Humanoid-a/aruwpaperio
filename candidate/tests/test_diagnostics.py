import json

import pytest

from candidate.agent.diagnostics import Diagnostics


def test_counts_accumulate_and_timings_give_mean_p95_and_max():
    diag = Diagnostics()
    diag.count("turns", 3)
    diag.count("turns")
    assert diag.counts["turns"] == 4
    for ms in range(1, 101):
        diag.time("total", float(ms))
    mean, p95, worst = diag.timing_stats("total")
    assert mean == pytest.approx(50.5) and p95 == pytest.approx(95.05) and worst == 100.0
    assert diag.timing_stats("never") is None


def test_errors_are_separated_from_info_and_counted():
    diag = Diagnostics()
    diag.anomaly("RULE_MISMATCH", 3, "predicted X, engine said Y")
    diag.anomaly("COUNT_MISMATCH", 4, "off by one", severity="info")
    assert [a.kind for a in diag.errors()] == ["RULE_MISMATCH"]
    assert diag.counts["anomaly_error"] == 1 and diag.counts["anomaly_info"] == 1
    assert "turn 3" in str(diag.anomalies[0])


def test_only_the_first_errors_are_echoed_and_info_is_never_echoed():
    echoed: list[str] = []
    diag = Diagnostics(echo=echoed.append, max_echoed_anomalies=3)
    for turn in range(10):
        diag.anomaly("BAD", turn, "x")
    diag.anomaly("FYI", 0, "y", severity="info")
    assert len(echoed) == 4 and "not echoed" in echoed[-1]
    assert len(diag.errors()) == 10


def test_the_report_names_the_counts_and_the_anomalies():
    diag = Diagnostics()
    diag.start_match()
    diag.note_turn(0, 100)
    for name, n in [("turns", 100), ("move_NORMAL", 90), ("move_CAPTURED", 8), ("move_DIED", 2), ("pred_result_ok", 100), ("state_ok", 99), ("gain_ok", 8)]:
        diag.count(name, n)
    diag.count("event_CAPTURED", 8)
    diag.time("total", 1.5)
    diag.anomaly("STATE_MISMATCH", 7, "expected a, found b")
    text = diag.report()
    assert "match 1" in text and "of 100" in text
    assert "CAPTURED 8" in text and "move result 100/100" in text and "capture size 8/8" in text
    assert "1 errors" in text and "STATE_MISMATCH at turn 7" in text


def test_starting_a_match_reports_and_closes_the_previous_one_exactly_once():
    echoed: list[str] = []
    diag = Diagnostics(echo=echoed.append)
    diag.start_match()
    diag.count("turns", 5)
    diag.start_match()           # a second match begins: the first is reported
    assert len([t for t in echoed if "runtime report" in t]) == 1 and diag.match_number == 2
    assert diag.counts["turns"] == 0
    diag.count("turns", 2)
    diag.finish_match()
    diag.finish_match()          # finishing twice must not print twice
    diag.start_match()           # and the next match must not re-report a finished one
    assert len([t for t in echoed if "runtime report" in t]) == 2


def test_an_empty_first_match_is_not_reported():
    echoed: list[str] = []
    diag = Diagnostics(echo=echoed.append)
    diag.start_match()
    diag.start_match()
    assert echoed == []


def test_the_log_directory_is_created_with_an_ignore_file_and_records_are_json_lines(tmp_path):
    diag = Diagnostics(log_dir=tmp_path / "a" / "logs")
    diag.start_match()
    diag.record({"turn": 0, "x": 1})
    diag.record({"turn": 1, "cells": (1, 2)})
    diag.close()
    assert (tmp_path / "a" / "logs" / ".gitignore").exists()
    rows = [json.loads(line) for line in diag.log_path.read_text().splitlines()]
    assert rows == [{"turn": 0, "x": 1}, {"turn": 1, "cells": [1, 2]}]


def test_without_a_log_directory_recording_is_a_no_op():
    diag = Diagnostics()
    diag.start_match()
    diag.record({"turn": 0})
    assert diag.log_path is None


def test_drift_is_summarised_by_turns_and_the_largest_gap():
    diag = Diagnostics()
    for drift in (0, 2, -5, 0, 3):
        diag.note_drift(drift)
    assert diag.counts["drift_turns"] == 3 and diag.max_drift == 5
