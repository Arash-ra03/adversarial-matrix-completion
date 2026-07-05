import numpy as np
import pytest
from matrix_arena.instance import generate_instance, SMALL, MEDIUM, LARGE


def test_determinism_same_seed():
    r1 = generate_instance(42, SMALL)
    r2 = generate_instance(42, SMALL)
    np.testing.assert_array_equal(r1[2], r2[2])


def test_determinism_diff_seed():
    r1 = generate_instance(42, SMALL)
    r2 = generate_instance(43, SMALL)
    assert not np.allclose(r1[2], r2[2])


def test_shapes_small():
    X, Z, Y_gt = generate_instance(0, SMALL)
    n, m, d = SMALL["n"], SMALL["m"], SMALL["d"]
    assert X.shape == (n, d)
    assert Z.shape == (m, d)
    assert Y_gt.shape == (n, m)


def test_shapes_medium():
    X, Z, Y_gt = generate_instance(0, MEDIUM)
    n, m, d = MEDIUM["n"], MEDIUM["m"], MEDIUM["d"]
    assert X.shape == (n, d)
    assert Z.shape == (m, d)
    assert Y_gt.shape == (n, m)


def test_y_gt_normalized():
    _, _, Y_gt = generate_instance(7, SMALL)
    assert abs(Y_gt.mean()) < 0.3
    assert abs(Y_gt.std() - 1.0) < 0.3


def test_budget_config_keys():
    for cfg in [SMALL, MEDIUM, LARGE]:
        for key in ["n", "m", "d", "rank", "nonlinear_rank", "k",
                    "solve_timeout_s", "attack_timeout_s", "name"]:
            assert key in cfg


def test_multiple_seeds_differ():
    results = [generate_instance(s, SMALL)[2] for s in range(5)]
    for i in range(5):
        for j in range(i + 1, 5):
            assert not np.allclose(results[i], results[j]), f"seeds {i} and {j} produced same Y_gt"
