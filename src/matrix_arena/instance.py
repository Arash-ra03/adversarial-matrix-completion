"""Deterministic instance generator for Matrix Arena."""
from __future__ import annotations
import numpy as np
from typing import TypedDict


class BudgetConfig(TypedDict):
    name: str
    n: int
    m: int
    d: int
    rank: int
    nonlinear_rank: int
    k: int
    solve_timeout_s: float
    attack_timeout_s: float


SMALL: BudgetConfig = {
    "name": "small",
    "n": 48, "m": 48, "d": 12,
    "rank": 4, "nonlinear_rank": 2,
    "k": 10,
    "solve_timeout_s": 1.0,
    "attack_timeout_s": 0.25,
}

MEDIUM: BudgetConfig = {
    "name": "medium",
    "n": 96, "m": 96, "d": 24,
    "rank": 8, "nonlinear_rank": 4,
    "k": 12,
    "solve_timeout_s": 5.0,
    "attack_timeout_s": 1.0,
}

LARGE: BudgetConfig = {
    "name": "large",
    "n": 160, "m": 160, "d": 32,
    "rank": 12, "nonlinear_rank": 6,
    "k": 16,
    "solve_timeout_s": 15.0,
    "attack_timeout_s": 2.0,
}

# Regime: (bilinear_weight, latent_weight, nonlinear_weight, noise_std)
_REGIMES = [
    (0.85, 0.10, 0.05, 0.05),   # 0: mostly bilinear
    (0.10, 0.85, 0.05, 0.05),   # 1: mostly low-rank latent
    (0.45, 0.45, 0.10, 0.05),   # 2: mixed
    (0.25, 0.25, 0.50, 0.10),   # 3: nonlinear dominant
    (0.60, 0.30, 0.10, 0.20),   # 4: noisy
    (0.40, 0.40, 0.20, 0.05),   # 5: spiky (high leverage rows/cols)
]

# Human-readable regime labels (index-aligned with _REGIMES).
REGIME_NAMES = [
    "mostly-bilinear",
    "mostly-low-rank",
    "mixed",
    "nonlinear-dominant",
    "noisy",
    "spiky",
]


def get_regime(seed: int) -> dict:
    """Return the regime metadata used for a given seed.

    Useful for visualization and analysis. Mirrors the deterministic
    ``regime_idx = seed % len(_REGIMES)`` selection inside ``generate_instance``.
    """
    idx = seed % len(_REGIMES)
    w_bil, w_lat, w_nl, noise_std = _REGIMES[idx]
    return {
        "index": idx,
        "name": REGIME_NAMES[idx],
        "w_bilinear": w_bil,
        "w_latent": w_lat,
        "w_nonlinear": w_nl,
        "noise_std": noise_std,
    }


def generate_instance(
    seed: int,
    budget: BudgetConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Z, Y_gt) deterministically from seed and budget config.

    Y_gt[i,j] = w_bil * (x_i @ W @ z_j)
               + w_lat * (u_i @ v_j)
               + w_nl  * tanh(sum_l a_l * (x_i @ p_l) * (z_j @ q_l))
               + eps_ij
    Normalized to mean~0, std~1.
    """
    rng = np.random.default_rng(seed)
    n, m, d = budget["n"], budget["m"], budget["d"]
    r = budget["rank"]
    r_nl = budget["nonlinear_rank"]

    regime_idx = seed % len(_REGIMES)
    w_bil, w_lat, w_nl, noise_std = _REGIMES[regime_idx]

    # --- Features ---
    X = rng.standard_normal((n, d))
    Z = rng.standard_normal((m, d))

    if regime_idx == 5:  # spiky regime
        n_spike_r = max(1, n // 8)
        n_spike_c = max(1, m // 8)
        spike_rows = rng.choice(n, size=n_spike_r, replace=False)
        spike_cols = rng.choice(m, size=n_spike_c, replace=False)
        X[spike_rows] *= 3.0
        Z[spike_cols] *= 3.0

    # L2-normalise features so leverage scores are meaningful
    X /= np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-8)

    # --- Bilinear term: X @ W @ Z.T ---
    W = rng.standard_normal((d, d)) / np.sqrt(d)
    Y_bil = X @ W @ Z.T   # (n, m)

    # --- Latent low-rank term: U @ V.T ---
    U = rng.standard_normal((n, r)) / np.sqrt(r)
    V = rng.standard_normal((m, r)) / np.sqrt(r)
    Y_lat = U @ V.T       # (n, m)

    # --- Nonlinear term: sum_l a_l * tanh((x_i @ p_l) * (z_j @ q_l)) ---
    P = rng.standard_normal((d, r_nl)) / np.sqrt(d)
    Q = rng.standard_normal((d, r_nl)) / np.sqrt(d)
    a = rng.standard_normal(r_nl)
    XP = X @ P   # (n, r_nl)  projections of row features
    ZQ = Z @ Q   # (m, r_nl)  projections of col features
    # Y_nl[i,j] = sum_l a_l * tanh(XP[i,l] * ZQ[j,l])
    # Efficient: Y_nl = tanh(XP * a^{1/2}) @ tanh(ZQ * a^{1/2}).T — approximate
    # Exact: iterate over l
    Y_nl = np.zeros((n, m))
    for l in range(r_nl):
        Y_nl += a[l] * np.tanh(np.outer(XP[:, l], ZQ[:, l]))

    # --- Noise ---
    eps = rng.standard_normal((n, m)) * noise_std

    # --- Combine ---
    Y_gt = w_bil * Y_bil + w_lat * Y_lat + w_nl * Y_nl + eps

    # --- Normalize to ~N(0,1) ---
    mu = Y_gt.mean()
    sigma = Y_gt.std()
    if sigma < 1e-8:
        sigma = 1.0
    Y_gt = (Y_gt - mu) / sigma

    return X, Z, Y_gt
