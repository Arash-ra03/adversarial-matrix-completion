import numpy as np


class Agent:
    def solve(self, X, Z, Y_obs, mask, budget, seed):
        X = np.asarray(X, dtype=float)
        Z = np.asarray(Z, dtype=float)
        Y_obs = np.asarray(Y_obs, dtype=float)
        mask = np.asarray(mask, dtype=bool)

        n, m = Y_obs.shape
        if n == 0 or m == 0 or not mask.any():
            return np.zeros((n, m), dtype=float)

        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        rows, cols = np.where(mask)
        y = Y_obs[rows, cols].astype(float)
        mean = float(y.mean())

        rank_hint = int(budget.get("rank", 8)) if isinstance(budget, dict) else 8
        k_obs = max(1, int(round(mask.sum() / max(n, 1))))
        rank = max(2, min(rank_hint, k_obs - 2, 12, n, m))

        n_obs = len(y)
        perm = rng.permutation(n_obs)
        n_train = max(1, int(0.82 * n_obs))
        tr = perm[:n_train]
        val = perm[n_train:]

        y_bil_tr = self._fit_bilinear(X, Z, rows[tr], cols[tr], y[tr], alpha=0.12)
        y_lr_tr = self._fit_als(
            n, m, rows[tr], cols[tr], y[tr] - mean,
            rank=rank, reg=0.55, iters=self._als_iters(n), rng=rng
        ) + mean

        if len(val) >= 8:
            vr = rows[val]
            vc = cols[val]
            vy = y[val]
            pred_b = y_bil_tr[vr, vc]
            pred_l = y_lr_tr[vr, vc]
            diff = pred_b - pred_l
            denom = float(np.dot(diff, diff))
            if denom < 1e-12:
                alpha_blend = 0.5
            else:
                alpha_blend = float(np.clip(np.dot(vy - pred_l, diff) / denom, 0.0, 1.0))

            mean_loss = float(np.mean((mean - vy) ** 2))
            blend_loss = float(np.mean((alpha_blend * pred_b + (1.0 - alpha_blend) * pred_l - vy) ** 2))
            use_mean = mean_loss <= 0.98 * blend_loss
        else:
            alpha_blend = 0.5
            use_mean = False

        if use_mean:
            return np.full((n, m), mean, dtype=float)

        y_bil = self._fit_bilinear(X, Z, rows, cols, y, alpha=0.12)
        y_lr = self._fit_als(
            n, m, rows, cols, y - mean,
            rank=rank, reg=0.55, iters=self._als_iters(n), rng=rng
        ) + mean

        pred = alpha_blend * y_bil + (1.0 - alpha_blend) * y_lr
        pred = self._calibrate(pred, y, rows, cols, mean)
        return np.clip(np.nan_to_num(pred, nan=mean, posinf=5.0, neginf=-5.0), -8.0, 8.0)

    def attack(self, X, Z, k, budget, seed):
        X = np.asarray(X, dtype=float)
        Z = np.asarray(Z, dtype=float)
        n = X.shape[0]
        m = Z.shape[0]
        k = int(k)
        if n == 0 or m == 0 or k <= 0:
            return np.zeros((n, m), dtype=bool)

        k = min(k, m)
        if n != m:
            return self._circulant_mask(n, m, k)

        Xs = self._standardize(X)
        Zs = self._standardize(Z)
        row_lev = np.sum(Xs * Xs, axis=1)
        col_lev = np.sum(Zs * Zs, axis=1)
        row_order = np.argsort(row_lev, kind="mergesort")
        col_order = np.argsort(-col_lev, kind="mergesort")

        cost = self._attack_cost(Xs, Zs, row_lev, col_lev)
        shift_cost = np.zeros(m, dtype=float)
        pos = np.arange(m)
        for s in range(m):
            shift_cost[s] = float(np.sum(cost[row_order[pos], col_order[(pos + s) % m]]))

        pair_cost = shift_cost + np.roll(shift_cost, -1)
        base = int(np.argmin(pair_cost))
        shifts = [base, (base + 1) % m]
        for s in np.argsort(shift_cost, kind="mergesort"):
            s = int(s)
            if s not in shifts:
                shifts.append(s)
            if len(shifts) >= k:
                break

        out = np.zeros((n, m), dtype=bool)
        for s in shifts[:k]:
            for p in range(m):
                out[row_order[p], col_order[(p + s) % m]] = True

        if self._valid_mask(out, k):
            return out
        return self._circulant_mask(n, m, k)

    def _fit_bilinear(self, X, Z, rows, cols, y, alpha):
        n, d = X.shape
        Xi = X[rows]
        Zj = Z[cols]
        phi = (Xi[:, :, None] * Zj[:, None, :]).reshape(len(rows), d * d)
        eye = np.eye(d * d)
        try:
            w = np.linalg.solve(phi.T @ phi + float(alpha) * eye, phi.T @ y)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(phi.T @ phi + float(alpha) * eye, phi.T @ y, rcond=None)[0]
        return X @ w.reshape(d, d) @ Z.T

    def _fit_als(self, n, m, rows, cols, y, rank, reg, iters, rng):
        rank = int(max(1, min(rank, n, m)))
        p = 0.01 * rng.standard_normal((n, rank))
        q = 0.01 * rng.standard_normal((m, rank))
        eye = np.eye(rank)

        row_pos = [np.flatnonzero(rows == i) for i in range(n)]
        col_pos = [np.flatnonzero(cols == j) for j in range(m)]

        for _ in range(int(iters)):
            for i, idx in enumerate(row_pos):
                if idx.size == 0:
                    continue
                a = q[cols[idx]]
                p[i] = np.linalg.solve(a.T @ a + reg * eye, a.T @ y[idx])
            for j, idx in enumerate(col_pos):
                if idx.size == 0:
                    continue
                a = p[rows[idx]]
                q[j] = np.linalg.solve(a.T @ a + reg * eye, a.T @ y[idx])
        return p @ q.T

    def _als_iters(self, n):
        if n <= 64:
            return 80
        if n <= 112:
            return 65
        return 45

    def _calibrate(self, pred, y, rows, cols, fallback_mean):
        p = pred[rows, cols]
        var = float(np.var(p))
        if var < 1e-12:
            return pred - float(np.mean(pred)) + fallback_mean
        scale = float(np.cov(p, y, bias=True)[0, 1] / (var + 1e-12))
        bias = float(np.mean(y) - scale * np.mean(p))
        scale = float(np.clip(scale, 0.35, 1.65))
        bias = float(np.clip(bias, -2.5, 2.5))
        return scale * pred + bias

    def _attack_cost(self, X, Z, row_lev, col_lev):
        rl = row_lev / (np.mean(row_lev) + 1e-12)
        cl = col_lev / (np.mean(col_lev) + 1e-12)
        xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
        cos = np.clip(xn @ zn.T, -1.0, 1.0)
        leverage = rl[:, None] * cl[None, :]
        antipodal = 0.5 * (cos + 1.0)
        diagonal = np.abs(cos)
        return 0.65 * leverage + 0.25 * antipodal + 0.10 * diagonal

    def _standardize(self, a):
        a = np.asarray(a, dtype=float)
        return (a - a.mean(axis=0, keepdims=True)) / (a.std(axis=0, keepdims=True) + 1e-8)

    def _connected_shifts(self, k, m):
        shifts = []
        candidates = [0, 1]
        for t in range(1, m):
            candidates.extend([t, -t])
        for c in candidates:
            s = c % m
            if s not in shifts:
                shifts.append(s)
            if len(shifts) >= k:
                break
        return shifts

    def _circulant_mask(self, n, m, k):
        out = np.zeros((n, m), dtype=bool)
        for i in range(n):
            for s in self._connected_shifts(k, m):
                out[i, (i + s) % m] = True
        return out

    def _valid_mask(self, mask, k):
        if mask.ndim != 2:
            return False
        if not np.all(mask.sum(axis=1) == k):
            return False
        if not np.all(mask.sum(axis=0) == k):
            return False

        n, m = mask.shape
        seen_r = np.zeros(n, dtype=bool)
        seen_c = np.zeros(m, dtype=bool)
        queue = [0]
        seen_r[0] = True
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            if node >= 0:
                for j in np.flatnonzero(mask[node]):
                    if not seen_c[j]:
                        seen_c[j] = True
                        queue.append(-(j + 1))
            else:
                j = -node - 1
                for i in np.flatnonzero(mask[:, j]):
                    if not seen_r[i]:
                        seen_r[i] = True
                        queue.append(i)
        return bool(seen_r.all() and seen_c.all())
