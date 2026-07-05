"""Arena ranking: arena points and Bradley-Terry Elo."""
from __future__ import annotations
import math
import numpy as np


def compute_arena_points(
    wins: np.ndarray,
    draws: np.ndarray,
    losses: np.ndarray,
) -> np.ndarray:
    """Arena points per agent = (wins + 0.5*draws) / total_duels.

    All arrays are 1-D (per-agent totals, not pairwise matrices).
    """
    total = wins + draws + losses
    total = np.where(total == 0, 1, total)   # avoid division by zero
    return (wins + 0.5 * draws) / total


def fit_bradley_terry(
    wins: np.ndarray,
    draws: np.ndarray,
    n_iter: int = 1000,
    tol: float = 1e-8,
    l2_reg: float = 0.1,
    r_clip: float = 10.0,
) -> np.ndarray:
    """Minorization-Maximization (MM) algorithm for Bradley-Terry model.

    Parameters
    ----------
    wins   : (n, n) array  — wins[i, j] = times i beat j
    draws  : (n, n) array  — draws[i, j] = draws between i and j
    l2_reg : L2 regularization on skill parameters (prevents divergence when one
             agent wins all games; standard Bayesian prior).
    r_clip : clip skills to [-r_clip, r_clip] after each step for robustness.

    Returns
    -------
    r : (n,) array of skill parameters, zero-mean
        P(i beats j) = exp(r_i) / (exp(r_i) + exp(r_j))
    """
    n = wins.shape[0]
    # Effective wins: treat draw as 0.5-win for each side
    W = wins.astype(float) + 0.5 * draws.astype(float)
    # Total games between i and j (symmetric)
    N = W + W.T

    r = np.zeros(n)

    for _ in range(n_iter):
        exp_r = np.exp(np.clip(r, -r_clip, r_clip))
        new_r = np.zeros(n)
        for i in range(n):
            numerator = W[i].sum()
            denom_terms = N[i] / (exp_r[i] + exp_r + 1e-300)
            denominator = denom_terms.sum()
            if denominator < 1e-300:
                new_r[i] = r[i]
            else:
                # MM update with L2 regularization via gradient correction
                raw = math.log(max(numerator / denominator, 1e-300))
                new_r[i] = raw - l2_reg * r[i]

        # Clip and normalize to zero mean
        new_r = np.clip(new_r, -r_clip, r_clip)
        new_r -= new_r.mean()

        if np.max(np.abs(new_r - r)) < tol:
            r = new_r
            break
        r = new_r

    return r


def bt_to_elo(r: np.ndarray) -> np.ndarray:
    """Convert Bradley-Terry skill vector to Elo-like scale.

    elo_i = 1500 + 400 / log(10) * r_i
    """
    return 1500.0 + (400.0 / math.log(10)) * r


def _minmax_normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. If range is 0, return all 0.5."""
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-12:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / (hi - lo)


def compute_final_scores(
    solve_scores: np.ndarray,
    elo_scores: np.ndarray,
    robustness_scores: np.ndarray,
    w_solve: float = 0.60,
    w_arena: float = 0.35,
    w_robust: float = 0.05,
) -> np.ndarray:
    """Weighted combination of normalized solve score, arena Elo, and robustness.

    Final = 0.60 * norm(solve) + 0.35 * norm(elo) + 0.05 * robustness
    """
    norm_solve = _minmax_normalize(solve_scores)
    norm_arena = _minmax_normalize(elo_scores)
    return w_solve * norm_solve + w_arena * norm_arena + w_robust * robustness_scores
