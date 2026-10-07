"""AEX-128 finite state machine (paper Sec. III-C, Table 3).

States:   S0 = SubBytes, S1 = ShiftRows, S2 = MixColumns, S3 = ARX.
Labels:   lambda in {0, 1, 2}; TRANSITION[current][lambda] is the next state and
          is never equal to `current` (no self-loops).
Rounds:   10 main rounds, 2 FSM steps each -> 20 steps.  Step c belongs to round
          r = floor(c / 2) + 1 (paper eq. 3).  One label per round, used for BOTH
          steps of that round (paper Sec. III-C.4).
Order:    at step c the operation of the *current* state is executed and THEN
          the FSM moves (paper Sec. III-C.5, "apply-then-transition").  The
          traversal starts from state S0 (paper: "initial state s = 0").

A path is therefore fully determined by the label sequence
(lambda_1, ..., lambda_10) in {0,1,2}^10 -> 3^10 = 59,049 paths.
"""
from __future__ import annotations

import itertools
from typing import Iterable, Iterator, Sequence

NUM_STATES = 4
NUM_LABELS = 3
NUM_ROUNDS = 10
STEPS_PER_ROUND = 2
NUM_STEPS = NUM_ROUNDS * STEPS_PER_ROUND  # 20
NUM_PATHS = NUM_LABELS**NUM_ROUNDS  # 59049
INITIAL_STATE = 0  # paper Sec. III-C.5: "Set the current state s to an initial state s = 0"

STATE_NAMES = ("S0", "S1", "S2", "S3")
OP_NAMES = ("SubBytes", "ShiftRows", "MixColumns", "ARX")
OP_SHORT = ("SB", "SR", "MC", "ARX")

# Paper Table 3: rows = current state S0..S3, columns = label 0, 1, 2.
TRANSITION = (
    (1, 2, 3),  # from S0
    (2, 3, 0),  # from S1
    (3, 0, 1),  # from S2
    (0, 1, 2),  # from S3
)


def next_state(state: int, label: int) -> int:
    """Table 3 lookup (equivalent to (state + label + 1) mod 4)."""
    return TRANSITION[state][label]


def round_of_step(c: int) -> int:
    """1-based main-round index of FSM step c (paper eq. 3)."""
    return c // STEPS_PER_ROUND + 1


def _validate_labels(labels: Iterable[int]) -> tuple[int, ...]:
    labels = tuple(labels)
    if len(labels) != NUM_ROUNDS:
        raise ValueError(f"expected {NUM_ROUNDS} labels, got {len(labels)}")
    if any(lam not in range(NUM_LABELS) for lam in labels):
        raise ValueError(f"labels must be in {{0,1,2}}, got {labels}")
    return labels


def path_from_labels(labels: Iterable[int], initial_state: int = INITIAL_STATE) -> tuple[int, ...]:
    """Return the 20-step state (= operation) sequence visited for the given labels.

    states[c] is the state whose operation is applied at step c.
    """
    labels = _validate_labels(labels)
    if initial_state not in range(NUM_STATES):
        raise ValueError(f"initial_state must be in 0..3, got {initial_state}")
    states = []
    s = initial_state
    for c in range(NUM_STEPS):
        states.append(s)  # (c) execute Op(s) ...
        s = next_state(s, labels[round_of_step(c) - 1])  # (d)-(e) ... then move
    return tuple(states)


def labels_from_index(k: int) -> tuple[int, ...]:
    """Base-3 decoding; lambda_1 is the most significant digit."""
    if not 0 <= k < NUM_PATHS:
        raise ValueError(f"path index out of range: {k}")
    digits = []
    for _ in range(NUM_ROUNDS):
        digits.append(k % NUM_LABELS)
        k //= NUM_LABELS
    return tuple(reversed(digits))


def index_from_labels(labels: Iterable[int]) -> int:
    k = 0
    for lam in _validate_labels(labels):
        k = NUM_LABELS * k + lam
    return k


def all_label_sequences() -> Iterator[tuple[int, ...]]:
    """All 3^10 label sequences in index order (lexicographic, lambda_1 first)."""
    return itertools.product(range(NUM_LABELS), repeat=NUM_ROUNDS)


def op_counts(states: Sequence[int]) -> dict[int, int]:
    return {s: states.count(s) for s in range(NUM_STATES)}


def format_states(states: Sequence[int]) -> str:
    return "".join(str(s) for s in states)


def format_labels(labels: Sequence[int]) -> str:
    return "".join(str(lam) for lam in labels)


def describe(states: Sequence[int]) -> str:
    """Human-readable operation sequence, e.g. 'SB-SR-MC-ARX-...'."""
    return "-".join(OP_SHORT[s] for s in states)
