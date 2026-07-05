"""Baseline: ALS matrix factorization Y ≈ P @ Q.T on observed entries."""
import numpy as np
from matrix_arena.masks import generate_random_regular_mask


class Agent:
    _RANK = 8
    _LAMBDA = 0.5   # tuned: lam=0.01 overfits; 0.5 generalises well
    _N_ITER = 80

    def solve(self, X, Z, Y_obs, mask, budget, seed):
        rng = np.random.default_rng(seed)
        n, m = Y_obs.shape
        r = min(self._RANK, budget.get("rank", 8))
        P = rng.standard_normal((n, r)) * 0.01
        Q = rng.standard_normal((m, r)) * 0.01

        obs_rows, obs_cols = np.where(mask)
        y_flat = Y_obs[obs_rows, obs_cols]
        lam = self._LAMBDA

        for _ in range(self._N_ITER):
            for i in range(n):
                idx = obs_rows == i
                if not idx.any():
                    continue
                Q_i = Q[obs_cols[idx]]
                y_i = y_flat[idx]
                A = Q_i.T @ Q_i + lam * np.eye(r)
                P[i] = np.linalg.solve(A, Q_i.T @ y_i)

            for j in range(m):
                idx = obs_cols == j
                if not idx.any():
                    continue
                P_j = P[obs_rows[idx]]
                y_j = y_flat[idx]
                A = P_j.T @ P_j + lam * np.eye(r)
                Q[j] = np.linalg.solve(A, P_j.T @ y_j)

        return P @ Q.T

    def attack(self, X, Z, k, budget, seed):
        return generate_random_regular_mask(X.shape[0], k, seed)
