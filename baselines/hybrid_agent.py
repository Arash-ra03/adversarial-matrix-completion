"""Baseline: bilinear (X W Z.T) + low-rank ALS, blended via held-out cross-validation."""
import numpy as np
from matrix_arena.masks import generate_random_regular_mask


class Agent:
    _RANK = 8
    _ALPHA_BIL = 0.1    # ridge penalty for bilinear kron regression
    _LAMBDA_LR = 0.5    # ALS regularisation
    _N_ITER = 80
    _TRAIN_FRAC = 0.8   # fraction of observed entries used for training

    def solve(self, X, Z, Y_obs, mask, budget, seed):
        rng = np.random.default_rng(seed)
        n, d = X.shape
        m = Z.shape[0]
        r = min(self._RANK, budget.get("rank", 8))

        obs_rows, obs_cols = np.where(mask)
        N = len(obs_rows)
        if N == 0:
            return np.zeros((n, m), dtype=float)

        y_flat = Y_obs[obs_rows, obs_cols]

        # --- Split into train / validation for blending ---
        perm = rng.permutation(N)
        n_train = max(1, int(N * self._TRAIN_FRAC))
        tr = perm[:n_train]
        val = perm[n_train:]
        tr_rows, tr_cols, y_tr = obs_rows[tr], obs_cols[tr], y_flat[tr]

        # --- Model 1: bilinear kron regression on train ---
        Xi = X[tr_rows]; Zj = Z[tr_cols]
        Phi = (Xi[:, :, None] * Zj[:, None, :]).reshape(n_train, d * d)
        A = Phi.T @ Phi + self._ALPHA_BIL * np.eye(d * d)
        w = np.linalg.solve(A, Phi.T @ y_tr)
        Y_bil = X @ w.reshape(d, d) @ Z.T

        # --- Model 2: low-rank ALS on train ---
        P = rng.standard_normal((n, r)) * 0.01
        Q = rng.standard_normal((m, r)) * 0.01
        lam = self._LAMBDA_LR
        for _ in range(self._N_ITER):
            for i in range(n):
                idx = tr_rows == i
                if not idx.any():
                    continue
                Q_i = Q[tr_cols[idx]]
                A_i = Q_i.T @ Q_i + lam * np.eye(r)
                P[i] = np.linalg.solve(A_i, Q_i.T @ y_tr[idx])
            for j in range(m):
                idx = tr_cols == j
                if not idx.any():
                    continue
                P_j = P[tr_rows[idx]]
                A_j = P_j.T @ P_j + lam * np.eye(r)
                Q[j] = np.linalg.solve(A_j, P_j.T @ y_tr[idx])
        Y_lr = P @ Q.T

        # --- Blend via validation loss ---
        if len(val) < 2:
            return 0.5 * Y_bil + 0.5 * Y_lr

        val_rows, val_cols = obs_rows[val], obs_cols[val]
        y_val = y_flat[val]
        pred_bil = Y_bil[val_rows, val_cols]
        pred_lr = Y_lr[val_rows, val_cols]

        # Optimal scalar blend: min_a ||a*bil + (1-a)*lr - y||^2
        diff = pred_bil - pred_lr
        denom = float(np.dot(diff, diff))
        if denom < 1e-12:
            alpha = 0.5
        else:
            alpha = float(np.clip(np.dot(y_val - pred_lr, diff) / denom, 0.0, 1.0))

        return alpha * Y_bil + (1.0 - alpha) * Y_lr

    def attack(self, X, Z, k, budget, seed):
        return generate_random_regular_mask(X.shape[0], k, seed)
