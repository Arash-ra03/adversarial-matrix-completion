import numpy as np


class Agent:
    """Ensembled Matrix Arena agent.

    The solver combines the two real signals in the generator: a feature
    bilinear model X W Z.T and a latent low-rank residual.  The attacker returns
    a connected k-regular banded mask after sorting rows/columns by feature
    geometry, which gives the opponent locally redundant observations while
    hiding many cross-feature interactions.
    """

    def solve(self, X, Z, Y_obs, mask, budget, seed):
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        n, m = Y_obs.shape
        d = X.shape[1]
        rows, cols = np.where(mask)
        n_obs = int(rows.size)
        if n_obs == 0:
            return np.zeros((n, m), dtype=float)

        y = Y_obs[rows, cols].astype(float, copy=False)

        perm = rng.permutation(n_obs)
        n_train = max(1, int(0.80 * n_obs))
        tr = perm[:n_train]
        val = perm[n_train:]
        tr_rows, tr_cols, y_tr = rows[tr], cols[tr], y[tr]

        alpha = 0.1
        rank = min(8, int(budget.get("rank", 8)), min(n, m) - 1)
        lam = 0.5
        iters = 80

        Xi = X[tr_rows]
        Zj = Z[tr_cols]
        phi = (Xi[:, :, None] * Zj[:, None, :]).reshape(n_train, d * d)
        a = phi.T @ phi + alpha * np.eye(d * d)
        b = phi.T @ y_tr
        try:
            w = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(a, b, rcond=1e-6)[0]
        y_bil = X @ w.reshape(d, d) @ Z.T

        y_lr = self._als(tr_rows, tr_cols, y_tr, n, m, rank, lam, iters, seed + 31)

        if val.size < 2:
            out = 0.5 * y_bil + 0.5 * y_lr
        else:
            val_rows, val_cols = rows[val], cols[val]
            y_val = y[val]
            pred_bil = y_bil[val_rows, val_cols]
            pred_lr = y_lr[val_rows, val_cols]
            diff = pred_bil - pred_lr
            denom = float(np.dot(diff, diff))
            if denom < 1e-12:
                blend = 0.5
            else:
                blend = float(np.clip(np.dot(y_val - pred_lr, diff) / denom, 0.0, 1.0))
            out = blend * y_bil + (1.0 - blend) * y_lr

        out = np.where(np.isfinite(out), out, float(np.mean(y)))
        return np.clip(out, -8.0, 8.0)

    def attack(self, X, Z, k, budget, seed):
        n = int(X.shape[0])
        if k >= n:
            return np.ones((n, n), dtype=bool)

        row_order = self._feature_order(X, seed + 101)
        col_order = self._feature_order(Z, seed + 211)

        # Alternate between aligned and anti-aligned local bands.  Both are
        # valid; using the seed prevents a single exploitable attack pattern.
        if (int(seed) & 1) == 1:
            col_order = col_order[::-1]

        offsets = np.arange(k, dtype=int) - (k // 2)
        if k % 2 == 0:
            offsets = offsets + 1

        mask = np.zeros((n, n), dtype=bool)
        for pos, i in enumerate(row_order):
            js = col_order[(pos + offsets) % n]
            mask[i, js] = True

        # k-regular circulant bands with k >= 2 are connected; the competition
        # budgets use k >= 10.  This repair is here for defensive completeness.
        if not self._connected(mask):
            mask = self._permuted_circulant(n, k, seed)
        return mask

    def _als(self, rows, cols, y, n, m, rank, lam, n_iter, seed):
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        rank = int(max(1, min(rank, min(n, m) - 1)))

        base = np.zeros((n, m), dtype=float)
        base[rows, cols] = y
        try:
            u, s, vt = np.linalg.svd(base, full_matrices=False)
            scale = np.sqrt(np.maximum(s[:rank], 1e-12))
            p = u[:, :rank] * scale
            q = vt[:rank, :].T * scale
        except np.linalg.LinAlgError:
            p = rng.standard_normal((n, rank)) * 0.01
            q = rng.standard_normal((m, rank)) * 0.01

        row_idx = [np.where(rows == i)[0] for i in range(n)]
        col_idx = [np.where(cols == j)[0] for j in range(m)]
        eye = np.eye(rank)
        lam_eye = float(lam) * eye

        for _ in range(int(n_iter)):
            for i in range(n):
                idx = row_idx[i]
                if idx.size:
                    qi = q[cols[idx]]
                    a = qi.T @ qi + lam_eye
                    b = qi.T @ y[idx]
                    try:
                        p[i] = np.linalg.solve(a, b)
                    except np.linalg.LinAlgError:
                        p[i] = np.linalg.lstsq(a, b, rcond=1e-6)[0]
            for j in range(m):
                idx = col_idx[j]
                if idx.size:
                    pj = p[rows[idx]]
                    a = pj.T @ pj + lam_eye
                    b = pj.T @ y[idx]
                    try:
                        q[j] = np.linalg.solve(a, b)
                    except np.linalg.LinAlgError:
                        q[j] = np.linalg.lstsq(a, b, rcond=1e-6)[0]
        return p @ q.T

    def _feature_order(self, A, seed):
        A = np.asarray(A, dtype=float)
        n = A.shape[0]
        Ac = A - A.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(Ac, full_matrices=False)
            p1 = Ac @ vt[0]
            p2 = Ac @ vt[min(1, vt.shape[0] - 1)]
            score = p1 + 0.37 * p2
        except np.linalg.LinAlgError:
            score = Ac[:, 0].copy()
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        score = score + 1e-9 * rng.standard_normal(n)
        return np.argsort(score, kind="mergesort")

    def _permuted_circulant(self, n, k, seed):
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        row_perm = rng.permutation(n)
        col_perm = rng.permutation(n)
        mask = np.zeros((n, n), dtype=bool)
        for off in range(k):
            mask[row_perm, col_perm[(np.arange(n) + off) % n]] = True
        return mask

    def _connected(self, mask):
        n, m = mask.shape
        seen_r = np.zeros(n, dtype=bool)
        seen_c = np.zeros(m, dtype=bool)
        rows = [0]
        seen_r[0] = True
        while rows:
            new_cols = np.any(mask[rows], axis=0) & ~seen_c
            if np.any(new_cols):
                seen_c |= new_cols
                new_rows = np.any(mask[:, new_cols], axis=1) & ~seen_r
                rows = np.where(new_rows)[0].tolist()
                seen_r |= new_rows
            else:
                rows = []
        return bool(seen_r.all() and seen_c.all())
