import numpy as np
import pytest
from matrix_arena.scoring import (
    nrmse_hidden, validate_prediction, solve_score_instance,
    safe_nrmse, INVALID_LOSS,
)


def _make_data(n=20, m=20, obs_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    Y_gt = rng.standard_normal((n, m))
    mask = rng.random((n, m)) < obs_frac
    if not mask.any():
        mask[0, 0] = True
    return Y_gt, mask


def test_nrmse_perfect_prediction():
    Y_gt, mask = _make_data()
    loss = nrmse_hidden(Y_gt, Y_gt, mask)
    assert loss < 1e-8


def test_nrmse_bad_prediction():
    Y_gt, mask = _make_data()
    Y_bad = np.zeros_like(Y_gt)
    loss = nrmse_hidden(Y_bad, Y_gt, mask)
    assert loss > 0.5


def test_nrmse_hidden_only():
    """Score must only use hidden (mask==False) entries."""
    n, m = 10, 10
    Y_gt = np.ones((n, m))
    Y_hat = np.ones((n, m))
    mask = np.zeros((n, m), dtype=bool)
    mask[0, 0] = True
    Y_hat[0, 0] = 999.0  # corrupt observed entry only
    loss = nrmse_hidden(Y_hat, Y_gt, mask)
    assert loss < 1e-8, "NRMSE should be 0 — only hidden entries count"


def test_validate_prediction_ok():
    ok, msg = validate_prediction(np.ones((5, 5)), expected_shape=(5, 5))
    assert ok, msg


def test_validate_prediction_nan():
    ok, msg = validate_prediction(np.full((5, 5), np.nan), expected_shape=(5, 5))
    assert not ok
    assert "nan" in msg.lower()


def test_validate_prediction_inf():
    ok, msg = validate_prediction(np.full((5, 5), np.inf), expected_shape=(5, 5))
    assert not ok


def test_validate_prediction_wrong_shape():
    ok, msg = validate_prediction(np.ones((3, 4)), expected_shape=(5, 5))
    assert not ok
    assert "shape" in msg.lower()


def test_validate_prediction_absurdly_large():
    ok, msg = validate_prediction(np.full((5, 5), 2e5), expected_shape=(5, 5))
    assert not ok


def test_solve_score_positive_when_better_than_mean():
    rng = np.random.default_rng(1)
    Y_gt = rng.standard_normal((20, 20))
    mask = rng.random((20, 20)) < 0.3
    if not mask.any():
        mask[0, 0] = True
    mean_val = float(Y_gt[mask].mean())
    Y_mean = np.full_like(Y_gt, mean_val)
    score = solve_score_instance(Y_gt, Y_gt, Y_mean, mask)
    assert score > 0


def test_solve_score_zero_for_mean():
    rng = np.random.default_rng(1)
    Y_gt = rng.standard_normal((20, 20))
    mask = rng.random((20, 20)) < 0.3
    if not mask.any():
        mask[0, 0] = True
    mean_val = float(Y_gt[mask].mean())
    Y_mean = np.full_like(Y_gt, mean_val)
    score = solve_score_instance(Y_gt, Y_mean, Y_mean, mask)
    assert abs(score) < 1e-8


def test_safe_nrmse_invalid_prediction():
    Y_gt = np.ones((5, 5))
    mask = np.zeros((5, 5), dtype=bool)
    Y_nan = np.full((5, 5), np.nan)
    assert safe_nrmse(Y_nan, Y_gt, mask) == INVALID_LOSS


def test_invalid_loss_value():
    assert INVALID_LOSS == 1e6
