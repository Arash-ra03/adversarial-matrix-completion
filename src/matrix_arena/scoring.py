"""Scoring utilities for Matrix Arena."""
from __future__ import annotations
import numpy as np

INVALID_LOSS: float = 1e6
_MAX_ABS_VALUE: float = 1e5


def nrmse_hidden(
    Y_hat: np.ndarray,
    Y_gt: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Normalized RMSE on hidden (mask==False) entries."""
    hidden = ~mask
    if not hidden.any():
        return 0.0
    err = Y_hat[hidden] - Y_gt[hidden]
    mse = float(np.mean(err ** 2))
    denom = float(np.std(Y_gt[hidden])) + 1e-8
    return float(np.sqrt(mse) / denom)


def validate_prediction(
    Y_hat: np.ndarray,
    expected_shape: tuple[int, int],
) -> tuple[bool, str]:
    """Check shape, dtype, and finite-ness of a prediction."""
    if not isinstance(Y_hat, np.ndarray):
        return False, f"prediction must be np.ndarray, got {type(Y_hat)}"
    if Y_hat.shape != expected_shape:
        return False, f"shape mismatch: expected {expected_shape}, got {Y_hat.shape}"
    if np.any(np.isnan(Y_hat)):
        return False, "prediction contains NaN"
    if np.any(np.isinf(Y_hat)):
        return False, "prediction contains Inf"
    if np.any(np.abs(Y_hat) > _MAX_ABS_VALUE):
        return False, f"prediction contains values with |v| > {_MAX_ABS_VALUE}"
    return True, ""


def safe_nrmse(
    Y_hat: np.ndarray,
    Y_gt: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Return INVALID_LOSS if prediction is invalid, else nrmse_hidden."""
    ok, _ = validate_prediction(Y_hat, expected_shape=Y_gt.shape)
    if not ok:
        return INVALID_LOSS
    return nrmse_hidden(Y_hat, Y_gt, mask)


def solve_score_instance(
    Y_gt: np.ndarray,
    Y_hat: np.ndarray,
    Y_mean: np.ndarray,
    mask: np.ndarray,
) -> float:
    """log(loss_mean / max(loss_agent, 1e-12)).

    Positive when agent beats mean baseline, zero if equal, negative if worse.
    """
    loss_agent = safe_nrmse(Y_hat, Y_gt, mask)
    loss_mean = safe_nrmse(Y_mean, Y_gt, mask)
    return float(np.log(loss_mean / max(loss_agent, 1e-12)))
