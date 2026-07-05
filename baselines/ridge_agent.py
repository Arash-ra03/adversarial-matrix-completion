"""Baseline: ridge regression on [X_i, Z_j, X_i * Z_j] feature interactions."""
import numpy as np
from matrix_arena.masks import generate_random_regular_mask


class Agent:
    _ALPHA = 1e-2

    def solve(self, X, Z, Y_obs, mask, budget, seed):
        n, d = X.shape
        m = Z.shape[0]
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return np.zeros((n, m), dtype=float)

        Xi = X[rows]       # (N_obs, d)
        Zj = Z[cols]       # (N_obs, d)
        Phi = np.concatenate([Xi, Zj, Xi * Zj], axis=1)  # (N_obs, 3d)
        y = Y_obs[rows, cols]

        A = Phi.T @ Phi + self._ALPHA * np.eye(Phi.shape[1])
        b = Phi.T @ y
        w = np.linalg.solve(A, b)

        Xi_all = np.repeat(X, m, axis=0)       # (n*m, d)
        Zj_all = np.tile(Z, (n, 1))            # (n*m, d)
        Phi_all = np.concatenate([Xi_all, Zj_all, Xi_all * Zj_all], axis=1)
        return (Phi_all @ w).reshape(n, m)

    def attack(self, X, Z, k, budget, seed):
        return generate_random_regular_mask(X.shape[0], k, seed)
