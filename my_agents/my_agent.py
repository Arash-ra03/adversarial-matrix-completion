import numpy as np


class Agent:
    def solve(self, X, Z, Y_obs, mask, budget, seed):
        X = np.asarray(X, dtype=float)
        Z = np.asarray(Z, dtype=float)
        Y_obs = np.asarray(Y_obs, dtype=float)
        mask = np.asarray(mask, dtype=bool)

        n, m = Y_obs.shape
        if n == 0 or m == 0 or mask.sum() == 0:
            return np.zeros((n, m), dtype=float)

        rows, cols = np.where(mask)
        y = Y_obs[rows, cols].astype(float)
        obs_mean = float(np.mean(y))

        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        Xs = self._standardize(X)
        Zs = self._standardize(Z)

        f_nonlin = 8 if max(n, m) <= 64 else (12 if max(n, m) <= 112 else 16)
        A = rng.normal(size=(Xs.shape[1], f_nonlin))
        B = rng.normal(size=(Zs.shape[1], f_nonlin))
        A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-12
        B /= np.linalg.norm(B, axis=0, keepdims=True) + 1e-12

        train_idx, val_idx = self._validation_split(rows, cols, n, int(seed))
        choice = "hybrid"
        if val_idx.size >= max(12, n // 4) and train_idx.size >= max(30, val_idx.size):
            train_mask = np.zeros((n, m), dtype=bool)
            train_mask[rows[train_idx], cols[train_idx]] = True
            train_candidates = self._make_candidates(
                Xs, Zs, rows[train_idx], cols[train_idx], y[train_idx],
                train_mask, A, B, rng
            )
            losses = {}
            vr = rows[val_idx]
            vc = cols[val_idx]
            vy = y[val_idx]
            for name, cand in train_candidates.items():
                diff = cand[vr, vc] - vy
                losses[name] = float(np.mean(diff * diff))
            choice = min(losses, key=losses.get)

        candidates = self._make_candidates(Xs, Zs, rows, cols, y, mask, A, B, rng)
        pred = candidates.get(choice, candidates["hybrid"])

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
        Xs = self._standardize(X)
        Zs = self._standardize(Z)

        row_score = np.sum(Xs * Xs, axis=1)
        col_score = np.sum(Zs * Zs, axis=1)

        row_order = np.argsort(row_score, kind="mergesort")
        col_order = np.argsort(-col_score, kind="mergesort")

        mask = np.zeros((n, m), dtype=bool)
        shifts = self._connected_shifts(k, m)
        usable = min(n, m)
        for shift in shifts:
            for pos in range(usable):
                i = row_order[pos]
                j = col_order[(pos + shift) % m]
                mask[i, j] = True

        if n == m and self._valid_mask(mask, k):
            return mask

        fallback = self._circulant_mask(n, m, k)
        return fallback

    def _standardize(self, A):
        A = np.asarray(A, dtype=float)
        mu = np.mean(A, axis=0, keepdims=True)
        sd = np.std(A, axis=0, keepdims=True) + 1e-8
        return (A - mu) / sd

    def _features(self, X, Z, rows, cols, A, B):
        Xi = X[rows]
        Zj = Z[cols]
        kron = (Xi[:, :, None] * Zj[:, None, :]).reshape(len(rows), -1)
        nonlin = np.tanh((Xi @ A) * (Zj @ B))
        return np.hstack((Xi, Zj, kron, nonlin))

    def _fit_feature_ensemble(self, X, Z, rows, cols, y, A, B, alphas):
        Phi = self._features(X, Z, rows, cols, A, B)
        mu = Phi.mean(axis=0)
        sd = Phi.std(axis=0) + 1e-8
        Phi = (Phi - mu) / sd
        y_mean = float(y.mean())
        yc = y - y_mean

        preds = []
        eye = np.eye(Phi.shape[1])
        XtX = Phi.T @ Phi
        Xty = Phi.T @ yc
        for alpha in alphas:
            try:
                coef = np.linalg.solve(XtX + float(alpha) * eye, Xty)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(XtX + float(alpha) * eye, Xty, rcond=None)[0]
            preds.append(self._predict_feature_model(X, Z, A, B, mu, sd, coef, y_mean))

        return np.mean(preds, axis=0)

    def _make_candidates(self, X, Z, rows, cols, y, mask, A, B, rng):
        n, m = mask.shape
        obs_mean = float(np.mean(y))

        mean_pred = np.full((n, m), obs_mean, dtype=float)

        row_bias, col_bias = self._residual_biases(y - obs_mean, rows, cols, n, m)
        rowcol = obs_mean + 0.5 * (row_bias[:, None] + col_bias[None, :])
        rowcol = self._calibrate_to_observed(rowcol, y, rows, cols, obs_mean)

        alphas = (0.5, 2.0, 8.0, 32.0)
        base = self._fit_feature_ensemble(X, Z, rows, cols, y, A, B, alphas)
        base = self._calibrate_to_observed(base, y, rows, cols, obs_mean)

        resid = np.zeros((n, m), dtype=float)
        resid[rows, cols] = y - base[rows, cols]

        k_obs = int(round(mask.sum() / max(n, 1)))
        rank = max(2, min(k_obs - 2, X.shape[1] // 2 + 2, 10))
        iters = 5 if max(n, m) <= 64 else 6
        low = self._masked_als(resid, mask, rank=rank, reg=0.35, iters=iters, rng=rng)

        low_obs = low[rows, cols]
        denom = float(np.dot(low_obs, low_obs)) + 1e-12
        shrink = float(np.dot(y - base[rows, cols], low_obs) / denom)
        shrink = float(np.clip(shrink, 0.0, 1.25))

        res_row, res_col = self._residual_biases(y - base[rows, cols], rows, cols, n, m)
        bias = 0.15 * (res_row[:, None] + res_col[None, :])

        hybrid = base + shrink * low + bias
        hybrid = self._calibrate_to_observed(hybrid, y, rows, cols, obs_mean)

        return {
            "mean": mean_pred,
            "rowcol": rowcol,
            "base": base,
            "hybrid": hybrid,
        }

    def _predict_feature_model(self, X, Z, A, B, mu, sd, coef, y_mean):
        n = X.shape[0]
        m = Z.shape[0]
        out = np.empty((n, m), dtype=float)
        all_cols = np.arange(m)
        for i in range(n):
            rows = np.full(m, i, dtype=int)
            Phi = self._features(X, Z, rows, all_cols, A, B)
            out[i] = y_mean + ((Phi - mu) / sd) @ coef
        return out

    def _masked_als(self, R, mask, rank, reg, iters, rng):
        n, m = R.shape
        rank = int(max(1, min(rank, n, m)))
        P = 0.01 * rng.normal(size=(n, rank))
        Q = 0.01 * rng.normal(size=(m, rank))
        eye = np.eye(rank)

        row_cols = [np.flatnonzero(mask[i]) for i in range(n)]
        col_rows = [np.flatnonzero(mask[:, j]) for j in range(m)]

        for _ in range(int(iters)):
            for i, js in enumerate(row_cols):
                if js.size == 0:
                    continue
                A = Q[js]
                b = R[i, js]
                P[i] = np.linalg.solve(A.T @ A + reg * eye, A.T @ b)

            for j, is_ in enumerate(col_rows):
                if is_.size == 0:
                    continue
                A = P[is_]
                b = R[is_, j]
                Q[j] = np.linalg.solve(A.T @ A + reg * eye, A.T @ b)

        return P @ Q.T

    def _validation_split(self, rows, cols, n, seed):
        val = []
        for i in range(n):
            idx = np.flatnonzero(rows == i)
            if idx.size >= 6:
                order = np.argsort((cols[idx] * 1103515245 + i * 12345 + seed) & 0x7FFFFFFF)
                take = 2 if idx.size >= 10 else 1
                val.extend(idx[order[:take]].tolist())
        if not val:
            return np.arange(rows.size), np.array([], dtype=int)

        val_idx = np.array(sorted(val), dtype=int)
        keep = np.ones(rows.size, dtype=bool)
        keep[val_idx] = False
        train_idx = np.flatnonzero(keep)
        return train_idx, val_idx

    def _residual_biases(self, residual, rows, cols, n, m):
        row_sum = np.zeros(n, dtype=float)
        row_cnt = np.zeros(n, dtype=float)
        col_sum = np.zeros(m, dtype=float)
        col_cnt = np.zeros(m, dtype=float)

        np.add.at(row_sum, rows, residual)
        np.add.at(row_cnt, rows, 1.0)
        np.add.at(col_sum, cols, residual)
        np.add.at(col_cnt, cols, 1.0)

        row_bias = row_sum / np.maximum(row_cnt, 1.0)
        col_bias = col_sum / np.maximum(col_cnt, 1.0)
        return row_bias, col_bias

    def _calibrate_to_observed(self, pred, y, rows, cols, fallback_mean):
        p = pred[rows, cols]
        pv = float(np.var(p))
        if pv < 1e-12:
            return pred - float(np.mean(pred)) + fallback_mean

        a = float(np.cov(p, y, bias=True)[0, 1] / (pv + 1e-12))
        b = float(np.mean(y) - a * np.mean(p))
        a = float(np.clip(a, 0.25, 1.75))
        b = float(np.clip(b, -3.0, 3.0))
        out = a * pred + b

        obs_std = float(np.std(y))
        out_std = float(np.std(out))
        if out_std > 3.0 * max(obs_std, 1e-6):
            center = float(np.mean(out))
            out = center + (out - center) * (3.0 * max(obs_std, 1e-6) / out_std)
        return out

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
        mask = np.zeros((n, m), dtype=bool)
        shifts = self._connected_shifts(k, m)
        for i in range(n):
            for s in shifts:
                mask[i, (i + s) % m] = True
        return mask

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
