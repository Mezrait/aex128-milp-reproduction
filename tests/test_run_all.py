"""Tests for run_all.py (exhaustive driver)."""
import csv
import json
import subprocess
import sys
from pathlib import Path

from fsm import NUM_PATHS, index_from_labels
from run_all import CSV_FIELDS, main, solve_path_record

ROOT = Path(__file__).resolve().parents[1]


def test_record_for_all_ones_path():
    rec = solve_path_record(index_from_labels((1,) * 10), initial_state=0, arx_mode="identity")
    assert list(rec) == CSV_FIELDS
    assert rec["labels"] == "1111111111"
    assert rec["states"] == "02" * 10
    assert (rec["n_sb"], rec["n_sr"], rec["n_mc"], rec["n_arx"]) == (10, 0, 10, 0)
    assert rec["min_active"] == 25


def test_record_respects_initial_state_and_arx_mode():
    rec = solve_path_record(index_from_labels((1,) * 10), initial_state=1, arx_mode="identity")
    assert rec["states"] == "13" * 10 and rec["min_active"] == 0
    rec = solve_path_record(0, initial_state=0, arx_mode="free")
    assert rec["states"] == "0123" * 5 and rec["min_active"] == 1  # cancel right after the first ARX


def test_cli_limit_writes_csv_and_metadata(tmp_path):
    out = tmp_path / "paths.csv"
    n = main(["--limit", "7", "--workers", "1", "--out", str(out)])
    assert n == 7
    rows = list(csv.DictReader(out.open()))
    assert [int(r["path_id"]) for r in rows] == list(range(7))
    assert rows[0]["labels"] == "0000000000" and int(rows[0]["min_active"]) == 26
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["initial_state"] == 0 and meta["arx_mode"] == "identity" and meta["rows"] == 7
    assert meta["total_paths"] == NUM_PATHS


def test_cli_multiprocessing_path(tmp_path):
    out = tmp_path / "mp.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "run_all.py"), "--limit", "5", "--workers", "2", "--out", str(out)],
        check=True, cwd=ROOT, capture_output=True, text=True,
    )
    rows = list(csv.DictReader(out.open()))
    assert [int(r["path_id"]) for r in rows] == list(range(5))
    assert int(rows[0]["min_active"]) == 26
