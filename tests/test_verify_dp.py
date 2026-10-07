"""Tests for verify_dp.py: exact dynamic-programming cross-check of the MILP minima."""
import csv
import random

from fsm import labels_from_index, path_from_labels
from model import solve_min_active
from run_all import main as run_all_main
from verify_dp import dp_min_active, main

SB, SR, MC, ARX = 0, 1, 2, 3


def test_dp_hand_values_identity():
    assert dp_min_active((SB,)) == 1
    assert dp_min_active((SR, ARX, MC)) == 0
    assert dp_min_active((SB, MC, SB)) == 5
    assert dp_min_active((SB, SR, MC, SB)) == 5
    assert dp_min_active((SB, MC, SB, ARX, SB)) == 6
    assert dp_min_active((SB, SR, MC) * 3 + (SB,)) == 25
    assert dp_min_active(path_from_labels((1,) * 10)) == 25
    assert dp_min_active(path_from_labels((2, 0) * 5)) == 11
    assert dp_min_active(path_from_labels((0,) * 10)) == 26


def test_dp_free_arx_mode():
    assert dp_min_active((SB, ARX, SB), arx_mode="free") == 1
    assert dp_min_active((SB, MC, SB, ARX, SB), arx_mode="free") == 5
    assert dp_min_active(path_from_labels((0,) * 10), arx_mode="free") == 1


def test_dp_agrees_with_milp_on_random_paths():
    rng = random.Random(2026)
    for _ in range(12):
        states = path_from_labels(labels_from_index(rng.randrange(3**10)), rng.randrange(4))
        for mode in ("identity", "free"):
            assert dp_min_active(states, mode) == solve_min_active(states, mode)[0], (states, mode)


def test_cli_checks_csv_and_reports_zero_mismatches(tmp_path):
    csv_path = tmp_path / "paths.csv"
    run_all_main(["--limit", "9", "--workers", "1", "--out", str(csv_path)])
    report = tmp_path / "dp_check.json"
    assert main(["--csv", str(csv_path), "--sample", "0", "--report", str(report)]) == 0
    assert report.exists()


def test_cli_detects_a_corrupted_row(tmp_path):
    csv_path = tmp_path / "paths.csv"
    run_all_main(["--limit", "3", "--workers", "1", "--out", str(csv_path)])
    rows = list(csv.DictReader(csv_path.open()))
    rows[1]["min_active"] = str(int(rows[1]["min_active"]) + 1)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert main(["--csv", str(csv_path), "--sample", "0", "--report", str(tmp_path / "r.json")]) == 1
