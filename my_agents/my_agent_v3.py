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
        obs_mean = float(y.mean())

        rank_hint = int(budget.get("rank", 8)) if isinstance(budget, dict) else 8
        k_obs = max(1, int(round(mask.sum() / max(n, 1))))
        rank = max(2, min(rank_hint, k_obs - 2, 12, n, m))

        tr, val = self._split_observed(len(y), rng)
        train_mask = np.zeros((n, m), dtype=bool)
        train_mask[rows[tr], cols[tr]] = True

        train_preds = self._candidate_predictions(
            X, Z, rows[tr], cols[tr], y[tr], train_mask, rank, rng
        )

        if len(val) >= 8:
            vr, vc, vy = rows[val], cols[val], y[val]
            losses = {}
            for name, pred in train_preds.items():
                diff = pred[vr, vc] - vy
                losses[name] = float(np.mean(diff * diff))
            names, weights = self._validation_weights(losses)
        else:
            names, weights = ["hybrid"], np.array([1.0])

        final_preds = self._candidate_predictions(X, Z, rows, cols, y, mask, rank, rng)
        pred = np.zeros((n, m), dtype=float)
        for name, weight in zip(names, weights):
            pred += float(weight) * final_preds.get(name, final_preds["hybrid"])

        pred = self._calibrate(pred, y, rows, cols, obs_mean)
        pred = np.nan_to_num(pred, nan=obs_mean, posinf=5.0, neginf=-5.0)
        return np.clip(pred, -8.0, 8.0)

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

    def _candidate_predictions(self, X, Z, rows, cols, y, mask, rank, rng):
        n, m = mask.shape
        mean = float(y.mean()) if len(y) else 0.0
        preds = {}

        preds["mean"] = np.full((n, m), mean, dtype=float)
        preds["rowcol"] = self._rowcol_prediction(rows, cols, y, n, m)

        preds["ridge"] = self._fit_ridge_features(X, Z, rows, cols, y, alpha=0.08)
        preds["bilinear"] = self._fit_bilinear(X, Z, rows, cols, y, alpha=0.12)

        centered = y - mean
        low = self._fit_als(
            n, m, rows, cols, centered,
            rank=rank, reg=0.55, iters=self._als_iters(n), rng=rng
        ) + mean
        preds["lowrank"] = low

        resid = y - preds["bilinear"][rows, cols]
        residual_rank = max(2, min(rank, 10))
        res_low = self._fit_als(
            n, m, rows, cols, resid,
            rank=residual_rank, reg=0.45, iters=max(12, self._als_iters(n) // 2), rng=rng
        )
        obs_res_low = res_low[rows, cols]
        denom = float(np.dot(obs_res_low, obs_res_low)) + 1e-12
        shrink = float(np.clip(np.dot(resid, obs_res_low) / denom, 0.0, 1.25))
        preds["hybrid"] = preds["bilinear"] + shrink * res_low

        for name in list(preds):
            preds[name] = self._calibrate(preds[name], y, rows, cols, mean)
        return preds

    def _fit_bilinear(self, X, Z, rows, cols, y, alpha):
        n, d = X.shape
        Xi = X[rows]
        Zj = Z[cols]
        phi = (Xi[:, :, None] * Zj[:, None, :]).reshape(len(rows), d * d)
        reg = float(alpha) * np.eye(d * d)
        return self._solve_linear_and_predict(phi, y, reg, X, Z, mode="bilinear")

    def _fit_ridge_features(self, X, Z, rows, cols, y, alpha):
        n, d = X.shape
        m = Z.shape[0]
        Xi = X[rows]
        Zj = Z[cols]
        phi = np.concatenate((Xi, Zj, Xi * Zj), axis=1)
        mu = phi.mean(axis=0)
        sd = phi.std(axis=0) + 1e-8
        phi_s = (phi - mu) / sd
        yc = y - float(y.mean())
        reg = float(alpha) * np.eye(phi_s.shape[1])
        try:
            w = np.linalg.solve(phi_s.T @ phi_s + reg, phi_s.T @ yc)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(phi_s.T @ phi_s + reg, phi_s.T @ yc, rcond=None)[0]

        Xi_all = np.repeat(X, m, axis=0)
        Zj_all = np.tile(Z, (n, 1))
        all_phi = np.concatenate((Xi_all, Zj_all, Xi_all * Zj_all), axis=1)
        return (float(y.mean()) + ((all_phi - mu) / sd) @ w).reshape(n, m)

    def _solve_linear_and_predict(self, phi, y, reg, X, Z, mode):
        d = X.shape[1]
        try:
            w = np.linalg.solve(phi.T @ phi + reg, phi.T @ y)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(phi.T @ phi + reg, phi.T @ y, rcond=None)[0]
        if mode == "bilinear":
            return X @ w.reshape(d, d) @ Z.T
        raise ValueError(mode)

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

    def _rowcol_prediction(self, rows, cols, y, n, m):
        mean = float(y.mean())
        row_sum = np.zeros(n, dtype=float)
        row_cnt = np.zeros(n, dtype=float)
        col_sum = np.zeros(m, dtype=float)
        col_cnt = np.zeros(m, dtype=float)
        np.add.at(row_sum, rows, y - mean)
        np.add.at(row_cnt, rows, 1.0)
        np.add.at(col_sum, cols, y - mean)
        np.add.at(col_cnt, cols, 1.0)
        rb = row_sum / np.maximum(row_cnt, 1.0)
        cb = col_sum / np.maximum(col_cnt, 1.0)
        return mean + 0.5 * (rb[:, None] + cb[None, :])

    def _validation_weights(self, losses):
        ordered = sorted(losses, key=losses.get)
        best = losses[ordered[0]]
        mean_loss = losses.get("mean")
        if mean_loss is not None and ordered[0] != "mean" and best > 0.97 * mean_loss:
            return ["mean"], np.array([1.0])
        chosen = [name for name in ordered if losses[name] <= 1.08 * best][:3]
        if not chosen:
            chosen = [ordered[0]]
        inv = np.array([1.0 / (losses[name] + 1e-12) ** 2 for name in chosen], dtype=float)
        return chosen, inv / inv.sum()

    def _split_observed(self, n_obs, rng):
        perm = rng.permutation(n_obs)
        n_train = max(1, int(0.82 * n_obs))
        return perm[:n_train], perm[n_train:]

    def _als_iters(self, n):
        if n <= 64:
            return 48
        if n <= 112:
            return 38
        return 26

    def _calibrate(self, pred, y, rows, cols, fallback_mean):
        p = pred[rows, cols]
        var = float(np.var(p))
        if var < 1e-12:
            return pred - float(np.mean(pred)) + fallback_mean
        scale = float(np.cov(p, y, bias=True)[0, 1] / (var + 1e-12))
        bias = float(np.mean(y) - scale * np.mean(p))
        scale = float(np.clip(scale, 0.25, 1.75))
        bias = float(np.clip(bias, -3.0, 3.0))
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
