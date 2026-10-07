"""Truncated (byte-level) differential MILP for one AEX-128 FSM state sequence.

Model (Mouha et al. style, active S-box counting):
  * Byte layout: column-major, byte index i = 4*col + row (paper Sec. III-D.2:
    "4 x 4 byte matrix in column-major order"), so bytes 0..3 form column 0.
  * Variables x[t][i] in {0,1}, t = 0..T (T = number of steps): x[t][i] = 1 iff
    byte i carries a nonzero difference at the input of step t.
  * S0 SubBytes:   bijective per byte -> pattern unchanged, x[t+1] = x[t];
                   objective += sum_i x[t][i]  (input side).
  * S1 ShiftRows:  byte permutation, x[t+1][perm(i)] = x[t][i].
  * S2 MixColumns: AES MDS matrix, branch number 5, per column c with indicator
                   d[t][c]:  sum_in + sum_out >= 5*d,  every byte var <= d.
  * S3 ARX:        "transparent" (paper): identity on the truncated pattern
                   (primary reading) or fully unconstrained (arx_mode="free",
                   sensitivity check only).
  * AddRoundKey:   XOR with a constant does not change a difference -> omitted.
  * Nontrivial input: sum_i x[0][i] >= 1.
  * Objective: minimise the total number of active S-boxes.
"""
from __future__ import annotations

from typing import Sequence

import pulp

SB, SR, MC, ARX = 0, 1, 2, 3
ARX_MODES = ("identity", "free")
MDS_BRANCH_NUMBER = 5
NUM_BYTES = 16


def row_of(i: int) -> int:
    return i % 4


def col_of(i: int) -> int:
    return i // 4


COLUMNS = tuple(tuple(4 * c + r for r in range(4)) for c in range(4))

# AES ShiftRows: row r is rotated left by r positions: new[r][c] = old[r][(c + r) mod 4].
# SHIFT_ROWS_PERM[src] = dest, i.e. byte (r, c') moves to (r, (c' - r) mod 4).
SHIFT_ROWS_PERM = tuple(4 * ((col_of(i) - row_of(i)) % 4) + row_of(i) for i in range(NUM_BYTES))


def build_model(states: Sequence[int], arx_mode: str = "identity", name: str = "aex128_active_sboxes"):
    """Build the MILP for a state sequence. Returns (problem, x) with x[t][i] binaries."""
    states = tuple(states)
    if any(s not in (SB, SR, MC, ARX) for s in states):
        raise ValueError(f"states must be in 0..3, got {states}")
    if arx_mode not in ARX_MODES:
        raise ValueError(f"arx_mode must be one of {ARX_MODES}, got {arx_mode!r}")

    prob = pulp.LpProblem(name, pulp.LpMinimize)
    x = [
        [pulp.LpVariable(f"x_{t}_{i}", cat=pulp.LpBinary) for i in range(NUM_BYTES)]
        for t in range(len(states) + 1)
    ]
    objective = []
    for t, s in enumerate(states):
        cur, nxt = x[t], x[t + 1]
        if s == SB:
            objective.extend(cur)
            for i in range(NUM_BYTES):
                prob += nxt[i] == cur[i], f"sb_{t}_{i}"
        elif s == SR:
            for i in range(NUM_BYTES):
                prob += nxt[SHIFT_ROWS_PERM[i]] == cur[i], f"sr_{t}_{i}"
        elif s == MC:
            for c, col in enumerate(COLUMNS):
                d = pulp.LpVariable(f"d_{t}_{c}", cat=pulp.LpBinary)
                prob += (
                    pulp.lpSum(cur[i] for i in col) + pulp.lpSum(nxt[i] for i in col)
                    >= MDS_BRANCH_NUMBER * d
                ), f"mc_branch_{t}_{c}"
                for i in col:
                    prob += cur[i] <= d, f"mc_in_{t}_{i}"
                    prob += nxt[i] <= d, f"mc_out_{t}_{i}"
        elif s == ARX and arx_mode == "identity":
            for i in range(NUM_BYTES):
                prob += nxt[i] == cur[i], f"arx_{t}_{i}"
        # arx_mode == "free": no constraint links x[t] and x[t+1]

    prob += pulp.lpSum(x[0]) >= 1, "nontrivial_input"
    prob.setObjective(pulp.lpSum(objective))
    return prob, x


def make_solver(name: str = "highs", **kwargs):
    """Solver factory (all through PuLP).

    "highs"  - HiGHS via highspy, open source, default: solves every path in < 1 s.
    "cbc"    - CBC bundled with PuLP; measured to stall for minutes on AES-like
               paths such as (SB,SR,MC,ARX)x5, so it is not the default.
    "gurobi" - needs gurobipy and a licence (the brief's preferred solver).
    """
    name = name.lower()
    if name == "highs":
        return pulp.HiGHS(msg=False, threads=1, **kwargs)
    if name == "cbc":
        return pulp.PULP_CBC_CMD(msg=False, threads=1, **kwargs)
    if name == "gurobi":
        return pulp.GUROBI(msg=False, **kwargs)
    raise ValueError(f"unknown solver {name!r}; use highs, cbc or gurobi")


def solve_min_active(states: Sequence[int], arx_mode: str = "identity", solver=None):
    """Minimise active S-boxes for one state sequence.

    Returns (min_active, patterns) where patterns[t] is the 16-tuple of byte
    activity bits at the input of step t (t = 0..T) in the optimal solution.
    """
    prob, x = build_model(states, arx_mode)
    status = prob.solve(solver if solver is not None else make_solver())
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"solver returned status {pulp.LpStatus[status]} for states {states}")
    patterns = tuple(tuple(int(round(v.value() or 0.0)) for v in row) for row in x)
    min_active = sum(sum(patterns[t]) for t, s in enumerate(states) if s == SB)
    obj = pulp.value(prob.objective)
    if obj is not None and int(round(obj)) != min_active:
        raise RuntimeError(f"objective {obj} disagrees with recounted value {min_active}")
    return min_active, patterns
