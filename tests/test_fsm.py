"""Unit tests for fsm.py (AEX-128 finite state machine, paper Table 3 / Sec. III-C)."""
import itertools

import pytest

from fsm import (
    NUM_PATHS,
    NUM_ROUNDS,
    NUM_STEPS,
    TRANSITION,
    all_label_sequences,
    index_from_labels,
    labels_from_index,
    next_state,
    op_counts,
    path_from_labels,
)


def test_transition_table_matches_paper_table_3():
    # rows = current state S0..S3, columns = label 0,1,2
    assert TRANSITION == ((1, 2, 3), (2, 3, 0), (3, 0, 1), (0, 1, 2))


def test_transition_table_is_shift_by_label_plus_one():
    # Structural property visible in Table 3: next = (current + lambda + 1) mod 4
    for s in range(4):
        for lam in range(3):
            assert next_state(s, lam) == (s + lam + 1) % 4


def test_no_self_loops_in_table():
    for s in range(4):
        for lam in range(3):
            assert next_state(s, lam) != s


def test_each_row_reaches_all_other_states():
    for s in range(4):
        assert sorted(TRANSITION[s]) == sorted(set(range(4)) - {s})


def test_constants():
    assert NUM_ROUNDS == 10
    assert NUM_STEPS == 20
    assert NUM_PATHS == 3**10 == 59049


def test_hand_computed_paths():
    # lambda = 0 everywhere: S0 -> S1 -> S2 -> S3 -> S0 ...
    assert path_from_labels((0,) * 10) == tuple(c % 4 for c in range(20))
    # lambda = 1 everywhere: S0 -> S2 -> S0 -> S2 ...
    assert path_from_labels((1,) * 10) == tuple((2 * c) % 4 for c in range(20))
    # lambda = 2 everywhere: S0 -> S3 -> S2 -> S1 -> S0 ...
    assert path_from_labels((2,) * 10) == tuple((3 * c) % 4 for c in range(20))
    # mixed labels, both steps of a round use the same label:
    # r1 lam0: (S0,S1)->S2 ; r2 lam1: (S2,S0)->S2 ; r3 lam2: (S2,S1)->S0 ; repeat
    labels = (0, 1, 2, 0, 1, 2, 0, 1, 2, 0)
    expected = (0, 1, 2, 0, 2, 1) * 3 + (0, 1)
    assert path_from_labels(labels) == expected


def test_first_operation_is_the_initial_state():
    # Paper Sec. III-C.5: the operation of the *current* state is executed,
    # then the FSM moves (apply-then-transition).
    for init in range(4):
        assert path_from_labels((1, 0, 2, 1, 1, 0, 2, 2, 0, 1), initial_state=init)[0] == init


def test_initial_state_parameter():
    assert path_from_labels((1,) * 10, initial_state=1) == (1, 3) * 10
    assert path_from_labels((0,) * 10, initial_state=3) == tuple((3 + c) % 4 for c in range(20))


def test_both_steps_of_a_round_use_same_label():
    for labels in itertools.islice(all_label_sequences(), 0, 59049, 97):
        states = path_from_labels(labels)
        for c in range(NUM_STEPS - 1):
            r = c // 2  # zero-based round index of step c
            assert states[c + 1] == next_state(states[c], labels[r])


def test_all_paths_no_self_loops_count_and_distinct():
    seen = set()
    n = 0
    for labels in all_label_sequences():
        states = path_from_labels(labels)
        assert len(states) == NUM_STEPS
        for a, b in zip(states, states[1:]):
            assert a != b
        seen.add(states)
        n += 1
    assert n == NUM_PATHS
    assert len(seen) == NUM_PATHS  # label sequence -> state sequence is injective


def test_round_start_states_are_only_S0_or_S2_from_S0():
    # Consequence of applying the same label twice per round: 2*(lam+1) mod 4 in {0,2}
    for labels in all_label_sequences():
        states = path_from_labels(labels)
        assert all(states[c] in (0, 2) for c in range(0, NUM_STEPS, 2))


def test_index_roundtrip():
    assert labels_from_index(0) == (0,) * 10
    assert labels_from_index(NUM_PATHS - 1) == (2,) * 10
    assert labels_from_index(1) == (0,) * 9 + (1,)  # least significant digit = lambda_10
    for k in range(0, NUM_PATHS, 1234):
        assert index_from_labels(labels_from_index(k)) == k
    # enumeration order equals index order
    for k, labels in enumerate(itertools.islice(all_label_sequences(), 500)):
        assert labels_from_index(k) == labels


def test_op_counts():
    counts = op_counts(path_from_labels((1,) * 10))
    assert counts == {0: 10, 1: 0, 2: 10, 3: 0}
    counts = op_counts(path_from_labels((0,) * 10))
    assert counts == {0: 5, 1: 5, 2: 5, 3: 5}


def test_invalid_labels_rejected():
    with pytest.raises(ValueError):
        path_from_labels((0,) * 9)
    with pytest.raises(ValueError):
        path_from_labels((0,) * 9 + (3,))
    with pytest.raises(ValueError):
        path_from_labels((0,) * 10, initial_state=4)
