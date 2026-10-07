"""Unit tests for model.py (truncated-differential MILP for one FSM state sequence)."""
import pytest

from fsm import path_from_labels
from model import (
    ARX_MODES,
    COLUMNS,
    SHIFT_ROWS_PERM,
    build_model,
    col_of,
    make_solver,
    row_of,
    solve_min_active,
)

SB, SR, MC, ARX = 0, 1, 2, 3


# ---------------------------------------------------------------- layout maps
def test_layout_is_column_major():
    for i in range(16):
        assert row_of(i) == i % 4
        assert col_of(i) == i // 4
    assert COLUMNS == ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))


def test_shift_rows_is_a_row_preserving_permutation():
    assert sorted(SHIFT_ROWS_PERM) == list(range(16))
    for i in range(16):
        assert row_of(SHIFT_ROWS_PERM[i]) == row_of(i)


def test_shift_rows_rotates_row_r_left_by_r():
    # AES ShiftRows: new[r][c] = old[r][(c + r) mod 4]  <=>  old (r, c') -> new (r, c' - r)
    for r in range(4):
        for c in range(4):
            assert SHIFT_ROWS_PERM[4 * c + r] == 4 * ((c - r) % 4) + r
    assert SHIFT_ROWS_PERM[0] == 0 and SHIFT_ROWS_PERM[5] == 1 and SHIFT_ROWS_PERM[1] == 13


def test_shift_rows_spreads_one_column_over_all_columns():
    assert {col_of(SHIFT_ROWS_PERM[i]) for i in COLUMNS[0]} == {0, 1, 2, 3}


# ------------------------------------------------------------ pattern checker
def _check_patterns(states, patterns, arx_mode="identity"):
    """Independently verify that a pattern sequence satisfies every step constraint."""
    assert len(patterns) == len(states) + 1
    assert all(v in (0, 1) for p in patterns for v in p)
    assert sum(patterns[0]) >= 1
    for t, s in enumerate(states):
        a, b = patterns[t], patterns[t + 1]
        if s == SB or (s == ARX and arx_mode == "identity"):
            assert a == b
        elif s == SR:
            assert all(b[SHIFT_ROWS_PERM[i]] == a[i] for i in range(16))
        elif s == MC:
            for col in COLUMNS:
                w = sum(a[i] for i in col) + sum(b[i] for i in col)
                assert w == 0 or w >= 5
    return sum(sum(patterns[t]) for t, s in enumerate(states) if s == SB)


# ------------------------------------------------------------- miniature paths
def test_single_subbytes_step_min_is_one():
    n, patterns = solve_min_active((SB,))
    assert n == 1
    assert _check_patterns((SB,), patterns) == 1


@pytest.mark.parametrize("states", [(SR,), (MC,), (ARX,), (SR, ARX, SR, MC)])
def test_paths_without_subbytes_give_zero_but_keep_nontrivial_input(states):
    n, patterns = solve_min_active(states)
    assert n == 0
    assert sum(patterns[0]) >= 1
    _check_patterns(states, patterns)


def test_subbytes_mixcolumns_subbytes_is_five():
    # branch number 5: 1 active byte in, 4 out (or 4 in, 1 out)
    n, patterns = solve_min_active((SB, MC, SB))
    assert n == 5
    assert _check_patterns((SB, MC, SB), patterns) == 5


def test_subbytes_shiftrows_mixcolumns_subbytes_is_five():
    n, patterns = solve_min_active((SB, SR, MC, SB))
    assert n == 5
    assert _check_patterns((SB, SR, MC, SB), patterns) == 5


def test_four_aes_rounds_give_classic_bound_25():
    # (SB SR MC) x3 + SB is the AES 4-round structure: 1 + 4 + 16 + 4 = 25
    states = (SB, SR, MC) * 3 + (SB,)
    n, patterns = solve_min_active(states)
    assert n == 25
    assert _check_patterns(states, patterns) == 25


def test_alternating_sb_mc_path_all_labels_one_is_25():
    states = path_from_labels((1,) * 10)  # (SB, MC) x 10, no ShiftRows: 1,4,1,4,...
    n, patterns = solve_min_active(states)
    assert n == 25
    assert _check_patterns(states, patterns) == 25


def test_sb_arx_mc_arx_path_is_11():
    states = path_from_labels((2, 0) * 5)  # (SB, ARX, MC, ARX) x 5 -> 1,4,1,4,1 under ARX identity
    assert states == (SB, ARX, MC, ARX) * 5
    n, patterns = solve_min_active(states)
    assert n == 11
    assert _check_patterns(states, patterns) == 11


def test_five_aes_rounds_with_arx_identity_is_26():
    states = path_from_labels((0,) * 10)  # (SB SR MC ARX) x 5 -> 1,4,16,4,1
    n, patterns = solve_min_active(states)
    assert n == 26
    assert _check_patterns(states, patterns) == 26


# ---------------------------------------------------------------- ARX modes
def test_arx_modes_constant():
    assert ARX_MODES == ("identity", "free")


def test_arx_identity_keeps_pattern_and_free_can_cancel():
    n_id, p_id = solve_min_active((SB, ARX, SB), arx_mode="identity")
    n_free, p_free = solve_min_active((SB, ARX, SB), arx_mode="free")
    assert n_id == 2
    assert n_free == 1
    _check_patterns((SB, ARX, SB), p_id, "identity")
    _check_patterns((SB, ARX, SB), p_free, "free")


def test_arx_free_only_cancels_after_the_arx_step():
    states = (SB, MC, SB, ARX, SB)
    # identity: layers 2 and 3 share one pattern of weight w, layer 1 has weight u, u + w >= 5 -> u + 2w >= 6
    assert solve_min_active(states, arx_mode="identity")[0] == 6
    assert solve_min_active(states, arx_mode="free")[0] == 5


# -------------------------------------------------------------- model shape
def test_objective_counts_input_side_variables_of_subbytes_steps():
    prob, x = build_model((SB, SR, SB))
    names = {v.name for v in prob.objective.keys()}
    assert names == {f"x_0_{i}" for i in range(16)} | {f"x_2_{i}" for i in range(16)}


def test_model_size_for_a_full_path():
    states = path_from_labels((0,) * 10)  # 5 MixColumns steps
    prob, x = build_model(states)
    assert len(x) == 21 and all(len(row) == 16 for row in x)
    assert prob.numVariables() == 16 * 21 + 4 * 5


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        build_model((4,))
    with pytest.raises(ValueError):
        build_model((SB,), arx_mode="bogus")


def test_cbc_solver_is_available():
    assert make_solver("cbc").available()


def test_default_solver_is_highs_and_cbc_is_selectable():
    # CBC stalls (>150 s) on AES-like 5-round paths; HiGHS solves them in well under a second.
    assert make_solver().name == "HiGHS"
    assert make_solver("highs").available()
    assert make_solver("cbc").name == "PULP_CBC_CMD"
