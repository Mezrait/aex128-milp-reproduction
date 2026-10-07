Paper: M. Alawida, M. Menber, W. H. Alshoura, R. Almajed, S. Almuhammadi, "AEX-128: A Key-Dependent Finite-State SPN-ARX Hybrid Block Cipher", IEEE Access, 2026, https://doi.org/10.1109/ACCESS.2026.3730739

# aex128-milp-reproduction

Independent, from-scratch reproduction of the MILP active-S-box analysis of
AEX-128 (Section IV, Table 10 of the paper) by the paper's second author. The
paper's Java implementation and the co-author's MILP code were not used. The
target numbers were known in advance and are quoted below for comparison only;
nothing in this repository takes them as input.

## What the paper claims

Table 10 of the paper, "MILP-based active S-box analysis across all
3^10 = 59,049 FSM transition-label sequences":

| Metric | Active S-boxes |
|---|---|
| Minimum | 36 |
| Average | 87.17 |
| Median | 85 |
| Maximum | 133 |

The paper describes the model as follows (Section IV, "Resistance to
Differential and Linear Cryptanalysis"): "the MILP analysis exhaustively
considers all 3^10 = 59,049 possible transition-label sequences. Each
transition-label sequence determines a 20-step FSM traversal from the fixed
initial state. For each resulting execution path, the MILP model minimizes the
number of active S-boxes under the specified propagation constraints. To obtain
a conservative bound, the MILP model captures the diffusion properties of the
SPN components while treating the ARX layer as transparent. Specifically,
differences and linear masks are allowed to propagate through the ARX layer
without an associated probability or correlation penalty." From the minimum it
derives the single-characteristic bound (2^-6)^36 = 2^-216.

## What this repository contains

| File | Purpose |
|---|---|
| `fsm.py` | Table 3 transition table, label sequence to 20-step state sequence, path indexing |
| `model.py` | builds and solves the truncated-differential MILP for one state sequence (PuLP) |
| `run_all.py` | solves all 59,049 paths, writes `results/all_paths.csv` and a `.meta.json` sidecar |
| `analyze.py` | min / average / median / max, histogram, side-by-side comparison, `results/summary.md` |
| `verify_dp.py` | solver-free exact recomputation of the minima by dynamic programming (cross-check) |
| `tests/` | unit tests for all of the above, including hand-verifiable miniature paths |

## The cipher facts the model relies on (from the paper)

* 128-bit block viewed as a 4 x 4 byte matrix in **column-major order**
  (Section III-D.2). This repository indexes bytes as `i = 4*col + row`, so
  bytes 0..3 form column 0. ShiftRows rotates row r left by r positions
  (byte (r, c) moves to (r, (c - r) mod 4)); MixColumns is the AES matrix,
  applied to each of the four columns.
* Four FSM states, one elementary operation each: S0 SubBytes, S1 ShiftRows,
  S2 MixColumns, S3 ARX (Section III-C.1). Transition table (Table 3):

  | current | lambda = 0 | lambda = 1 | lambda = 2 |
  |---|---|---|---|
  | S0 | S1 | S2 | S3 |
  | S1 | S2 | S3 | S0 |
  | S2 | S3 | S0 | S1 |
  | S3 | S0 | S1 | S2 |

  i.e. `next = (current + lambda + 1) mod 4`, never a self-loop.
* 10 main rounds of 2 FSM steps each, 20 steps; one label per round derived
  from the round subkey, used for **both** steps of the round (Section III-C.4).
* **Initial state and step order (the brief's open detail).** Section III-C.5
  of the paper is explicit: "Set the current state s to an initial state
  s = 0", then for each step "execute the operation associated with the
  current state s" and only afterwards "move to the next state s <- T[s, lambda]".
  So the traversal starts in **S0 = SubBytes, apply-then-transition**, and the
  first operation of every path is SubBytes. This is what `fsm.path_from_labels`
  implements; the other three initial states were run as a sensitivity check
  (results below).

## The MILP model, exactly as implemented (`model.py`)

Variables: for every time point t = 0..20 and byte i = 0..15 a binary
`x[t][i]`, equal to 1 iff byte i carries a nonzero difference at the input of
step t. For every MixColumns step and column c an indicator binary `d[t][c]`.

Per step t, depending on the FSM state of that step:

| state | constraints | objective |
|---|---|---|
| S0 SubBytes | `x[t+1][i] = x[t][i]` for all i (bijective per byte) | `+ sum_i x[t][i]` (input side) |
| S1 ShiftRows | `x[t+1][perm(i)] = x[t][i]` with the AES permutation on the column-major layout | none |
| S2 MixColumns | per column: `sum_in + sum_out >= 5 * d[t][c]` and every one of the 8 byte variables `<= d[t][c]` (MDS, branch number 5) | none |
| S3 ARX | primary model: `x[t+1][i] = x[t][i]` (identity); sensitivity model `--arx-mode free`: no constraint | none |

Global: `sum_i x[0][i] >= 1` (nontrivial input difference). Objective: minimise
the total number of active S-boxes. Model size for one path: 336 pattern
binaries plus 4 indicators per MixColumns step.

Simplifications, each with its justification:

1. **Byte-level (truncated) differences.** Whether an S-box is active depends
   only on whether its input byte difference is nonzero, and the AES
   MixColumns branch number gives a valid byte-level constraint; the result is
   a lower bound on the active S-boxes of any characteristic (some truncated
   trails have no bit-level instantiation).
2. **ARX as the identity on the truncated pattern.** This is the reading of
   the paper's "propagate through the ARX layer without an associated
   probability penalty"; it charges no cost for the ARX step and keeps the
   byte-activity pattern. Because "transparent" could also mean that any
   output pattern is admissible, that fully unconstrained reading is run as a
   second model (`--arx-mode free`).
3. **AddRoundKey and whitening keys omitted.** XOR with a key-dependent
   constant does not change a difference.
4. **Key-dependent rotation amounts and the exact label derivation ignored.**
   Under the identity reading the ARX internals do not appear in the model,
   and the analysis is over all 3^10 label sequences rather than over keys.
5. **Deduplication of identical models.** SubBytes, ShiftRows and identity-ARX
   steps only copy or permute the pattern, so two paths whose state sequences
   coincide after deleting the ARX steps have the same optimum; `run_all.py`
   solves each of the 12,836 distinct models once and still writes one row
   per path. `--no-dedup` solves every path separately .

Solver: all models are built with PuLP. The default backend is HiGHS
(`highspy`, open source), because the CBC binary bundled with PuLP did not
finish the 5-round AES-like path (SB, SR, MC, ARX) x 5 within 150 s, while
HiGHS solved it in 0.4 s; both are exact, so the choice affects time only.
Gurobi can be selected with `--solver gurobi` when `gurobipy` and a licence
are available (it was not available in this reproduction).

## Results

The commands under "How to run" produce the result files.

Verification built into the code:

* **Solver-free recomputation.** `verify_dp.py` recomputes the minimum of
  every path by dynamic programming over the 2^16 byte-activity patterns,
  with no MILP solver involved, and reports any row that disagrees.
* **Hand-verifiable anchors** are unit tests: a single S-box gives 1;
  SB-MC-SB gives 5; four AES rounds (SB, SR, MC) x 3 + SB give 25 (Daemen and
  Rijmen's bound); (SB, SR, MC, ARX) x 5 gives 26, the classic 5-round AES
  bound.
* Every returned solution is re-counted from the pattern sequence and checked
  against the solver's objective; the tests additionally verify that the
  returned patterns satisfy every per-step constraint.

## Differences and open questions

**1. Assumptions in the brief that did not hold.** "Any solver finishes in
milliseconds" is true for HiGHS on the deduplicated sweep (735 s for 12,836
models on 8 workers) but false for the CBC binary bundled with PuLP, which did
not finish the path (SB, SR, MC, ARX) x 5 within 150 s. The rule "same label
for both steps of a round" restricts round-start states to S0 and S2, so the
plain AES order SubBytes-ShiftRows-MixColumns-SubBytes never occurs without an
ARX step in between; the sequence (SB, SR, MC, ARX) x 5 does occur (labels all
0) and gives the 5-round AES bound of 26.

**2. Out of scope, as in the brief:** no linear-mask model, no ARX probability
model, no bound tightening.

## How to run

```
python -m venv .venv            # or: uv venv --python 3.12 .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q             # unit tests (about 1 minute)
python run_all.py               # all 59,049 paths, S0 start, ARX identity -> results/all_paths.csv
python analyze.py --variants results/variants/*.csv   # results/summary.md + results/histogram.png
python verify_dp.py --sample 0  # solver-free recomputation of every row
python run_all.py --initial-state 2 --out results/variants/init_S2.csv   # sensitivity runs
python run_all.py --arx-mode free --out results/variants/arx_free.csv
python run_all.py --no-dedup --out results/verification/all_paths_nodedup.csv
```

CSV columns: `path_id` (base-3 index of the label sequence, lambda_1 most
significant), `labels` (10 digits), `states` (20 digits, the FSM state whose
operation is applied at each step), `n_sb`, `n_sr`, `n_mc`, `n_arx` (operation
counts), `min_active`.
