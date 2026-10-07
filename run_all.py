"""Exhaustive driver: minimum number of active S-boxes for every AEX-128 FSM path.

Writes one CSV row per label sequence (3^10 = 59,049 rows) plus a JSON sidecar
with run metadata.  Under the ARX-identity model the optimum of a path depends
only on its state sequence with the ARX steps removed, so paths sharing that
reduced sequence share one MILP; by default each distinct model is solved once
(12,836 models instead of 59,049).  Use --no-dedup to solve every path
independently; both modes must produce identical CSVs.

Usage:
    python run_all.py                       # S0 start, ARX identity, HiGHS, all cores
    python run_all.py --initial-state 2     # sensitivity: other initial state
    python run_all.py --arx-mode free       # sensitivity: unconstrained ARX
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pulp

from fsm import NUM_PATHS, format_labels, format_states, labels_from_index, op_counts, path_from_labels
from model import ARX, ARX_MODES, make_solver, solve_min_active

CSV_FIELDS = ["path_id", "labels", "states", "n_sb", "n_sr", "n_mc", "n_arx", "min_active"]
SOLVERS = ("highs", "cbc", "gurobi")
DEFAULT_OUT = Path("results") / "all_paths.csv"


def model_key(states, arx_mode):
    """Paths with equal keys have MILPs that differ only by identity steps."""
    if arx_mode == "identity":
        return tuple(s for s in states if s != ARX)
    return tuple(states)


def _record(path_id, labels, states, min_active):
    counts = op_counts(states)
    return {
        "path_id": path_id,
        "labels": format_labels(labels),
        "states": format_states(states),
        "n_sb": counts[0],
        "n_sr": counts[1],
        "n_mc": counts[2],
        "n_arx": counts[3],
        "min_active": min_active,
    }


def solve_path_record(path_id, initial_state=0, arx_mode="identity", solver_name="highs"):
    """Solve one path (full 20-step model) and return its CSV record."""
    labels = labels_from_index(path_id)
    states = path_from_labels(labels, initial_state)
    min_active, _ = solve_min_active(states, arx_mode, make_solver(solver_name))
    return _record(path_id, labels, states, min_active)


def _solve_states(job):
    states, arx_mode, solver_name = job
    min_active, _ = solve_min_active(states, arx_mode, make_solver(solver_name))
    return states, min_active


def _solve_many(jobs, workers):
    if workers <= 1:
        for job in jobs:
            yield _solve_states(job)
        return
    # maxtasksperchild bounds the memory of long-lived workers (tens of thousands of solves each)
    with mp.Pool(workers, maxtasksperchild=500) as pool:
        yield from pool.imap_unordered(_solve_states, jobs, chunksize=16)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--initial-state", type=int, default=0, choices=range(4),
                        help="FSM state applied at step 0 (paper: S0)")
    parser.add_argument("--arx-mode", default="identity", choices=ARX_MODES,
                        help="truncated model of the ARX layer (paper reading: identity)")
    parser.add_argument("--solver", default="highs", choices=SOLVERS,
                        help="PuLP backend; cbc stalls for minutes on AES-like paths")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=NUM_PATHS, help="only the first N path ids (testing)")
    parser.add_argument("--no-dedup", action="store_true", help="solve every path separately")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    n = min(args.limit, NUM_PATHS)

    paths = []
    representatives = {}  # key -> full state sequence of the first path with that key
    for path_id in range(n):
        labels = labels_from_index(path_id)
        states = path_from_labels(labels, args.initial_state)
        key = (path_id,) if args.no_dedup else model_key(states, args.arx_mode)
        paths.append((path_id, labels, states, key))
        representatives.setdefault(key, states)
    key_of_representative = {states: key for key, states in representatives.items()}
    jobs = [(states, args.arx_mode, args.solver) for states in representatives.values()]
    print(f"{n} paths, {len(jobs)} distinct MILPs, initial state S{args.initial_state}, "
          f"ARX {args.arx_mode}, solver {args.solver}, {args.workers} worker(s)", flush=True)

    results = {}
    report_every = max(1, len(jobs) // 20)
    for done, (states, min_active) in enumerate(_solve_many(jobs, args.workers), start=1):
        results[key_of_representative[states]] = min_active
        if done % report_every == 0 or done == len(jobs):
            print(f"  solved {done}/{len(jobs)} models ({time.perf_counter() - t0:.0f}s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, CSV_FIELDS)
        writer.writeheader()
        for path_id, labels, states, key in paths:
            writer.writerow(_record(path_id, labels, states, results[key]))

    elapsed = time.perf_counter() - t0
    meta = {
        "initial_state": args.initial_state,
        "arx_mode": args.arx_mode,
        "solver": args.solver,
        "workers": args.workers,
        "dedup": not args.no_dedup,
        "rows": n,
        "total_paths": NUM_PATHS,
        "unique_models": len(jobs),
        "elapsed_seconds": round(elapsed, 1),
        "started": started.isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "pulp": pulp.__version__,
        "platform": platform.platform(),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    values = [results[key] for _, _, _, key in paths]
    print(f"done: {n} rows -> {args.out}  (min {min(values)}, max {max(values)}, {elapsed:.0f}s)", flush=True)
    return n


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
