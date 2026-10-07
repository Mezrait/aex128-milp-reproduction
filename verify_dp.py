"""Exact cross-check of the MILP minima by dynamic programming over truncated patterns.

The truncated model has only 2^16 byte-activity patterns per time point, so the
minimum number of active S-boxes of a path can be computed exactly without any
MILP solver: keep, for every pattern, the cheapest cost of reaching it and push
that table through the steps (SubBytes adds the pattern weight, ShiftRows
permutes, MixColumns applies the branch-number-5 relation column by column, ARX
is the identity or - in "free" mode - a reset).  The code shares only the byte
layout constants with model.py, so agreement with the MILP results is strong
evidence that both implementations encode the same model correctly.

Usage:
    python verify_dp.py                      # re-check 3000 random rows of results/all_paths.csv
    python verify_dp.py --sample 0           # re-check every row
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from model import ARX, ARX_MODES, MC, MDS_BRANCH_NUMBER, SB, SHIFT_ROWS_PERM, SR

NUM_PATTERNS = 1 << 16
INF = 1 << 20
_PATTERNS = np.arange(NUM_PATTERNS, dtype=np.int64)
POPCOUNT = np.array([bin(p).count("1") for p in range(NUM_PATTERNS)], dtype=np.int32)

# ShiftRows on whole patterns: bit i (byte i) moves to bit SHIFT_ROWS_PERM[i].
SR_MAP = np.zeros(NUM_PATTERNS, dtype=np.int64)
for _i in range(16):
    SR_MAP |= ((_PATTERNS >> _i) & 1) << SHIFT_ROWS_PERM[_i]

# Column c occupies bits 4c..4c+3 (column-major layout), i.e. nibble c of the pattern.
NIBBLE_WEIGHT = [bin(a).count("1") for a in range(16)]
# Output nibble of weight k >= 1 admits input nibbles of weight >= 5 - k; output 0 admits only input 0.
ALLOWED_INPUTS = {k: [a for a in range(16) if NIBBLE_WEIGHT[a] >= MDS_BRANCH_NUMBER - k] for k in range(1, 5)}


def _mixcolumns(cost: np.ndarray) -> np.ndarray:
    """Apply the MDS branch-number relation to all four columns of the cost table."""
    table = cost.reshape(16, 16, 16, 16)  # axes = (column 3, column 2, column 1, column 0)
    for axis in range(4):
        best_by_weight = {
            k: np.min(np.take(table, ALLOWED_INPUTS[k], axis=axis), axis=axis) for k in range(1, 5)
        }
        new = np.empty_like(table)
        for b in range(16):
            index = [slice(None)] * 4
            index[axis] = b
            if b == 0:
                new[tuple(index)] = np.take(table, 0, axis=axis)
            else:
                new[tuple(index)] = best_by_weight[NIBBLE_WEIGHT[b]]
        table = new
    return table.reshape(-1)


def dp_min_active(states, arx_mode: str = "identity") -> int:
    """Exact minimum number of active S-boxes for a state sequence (same model as model.py)."""
    if arx_mode not in ARX_MODES:
        raise ValueError(f"arx_mode must be one of {ARX_MODES}, got {arx_mode!r}")
    cost = np.zeros(NUM_PATTERNS, dtype=np.int32)
    cost[0] = INF  # nontrivial input difference
    for s in states:
        if s == SB:
            cost = cost + POPCOUNT
        elif s == SR:
            new = np.empty_like(cost)
            new[SR_MAP] = cost
            cost = new
        elif s == MC:
            cost = _mixcolumns(cost)
        elif s == ARX:
            if arx_mode == "free":
                cost = np.full(NUM_PATTERNS, cost.min(), dtype=np.int32)
        else:
            raise ValueError(f"unknown state {s}")
    best = int(cost.min())
    if best >= INF:
        raise RuntimeError("no feasible pattern sequence")
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("results") / "all_paths.csv")
    parser.add_argument("--sample", type=int, default=3000, help="rows to re-check (0 = all)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--report", type=Path, default=Path("results") / "verification" / "dp_check.json")
    args = parser.parse_args(argv)

    with args.csv.open() as f:
        rows = list(csv.DictReader(f))
    meta_path = args.csv.with_suffix(".meta.json")
    arx_mode = json.loads(meta_path.read_text()).get("arx_mode", "identity") if meta_path.exists() else "identity"
    picked = rows if args.sample == 0 or args.sample >= len(rows) else random.Random(args.seed).sample(rows, args.sample)

    t0 = time.perf_counter()
    mismatches = []
    values = []
    for row in picked:
        states = tuple(int(ch) for ch in row["states"])
        dp_value = dp_min_active(states, arx_mode)
        values.append(dp_value)
        if dp_value != int(row["min_active"]):
            mismatches.append({"path_id": int(row["path_id"]), "labels": row["labels"], "states": row["states"],
                               "milp": int(row["min_active"]), "dp": dp_value})
    report = {
        "csv": str(args.csv),
        "arx_mode": arx_mode,
        "rows_in_csv": len(rows),
        "checked": len(picked),
        "seed": args.seed,
        "dp_min": min(values),
        "dp_max": max(values),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:50],
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"DP cross-check: {len(picked)}/{len(rows)} rows, {len(mismatches)} mismatches, "
          f"{report['elapsed_seconds']}s -> {args.report}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
