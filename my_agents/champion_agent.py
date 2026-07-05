import time
from collections import deque

import numpy as np


def _is_connected_bipartite(mask):
    n, m = mask.shape
    total = n + m
    rows_i, cols_j = np.where(mask)
    adj = [[] for _ in range(total)]
    for r, c in zip(rows_i.tolist(), cols_j.tolist()):
        adj[r].append(n + c)
        adj[n + c].append(r)
    visited = [False] * total
    queue = deque([0])
    visited[0] = True
    count = 1
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    return count == total


def _sample_config_model(n, k, rng):
    stubs = np.repeat(np.arange(n), k)
    rng.shuffle(stubs)
    grid = stubs.reshape(n, k)
    row_sets = [set(grid[i].tolist()) for i in range(n)]
    max_repairs = 200 * n
    repairs = 0
    while repairs < max_repairs:
        bad_i = -1
        for i in range(n):
            if len(row_sets[i]) < k:
                bad_i = i
                break
        if bad_i == -1:
            break

        row = grid[bad_i]
        seen = set()
        dup_pos = -1
        for p in range(k):
            if row[p] in seen:
                dup_pos = p
                break
            seen.add(int(row[p]))
        c = int(row[dup_pos])

        resolved = False
        for _ in range(100):
            i2 = int(rng.integers(n))
            if i2 == bad_i:
                continue
            p2 = int(rng.integers(k))
            c2 = int(grid[i2, p2])
            if c2 in row_sets[bad_i]:
                continue
            if c in row_sets[i2]:
                continue
            grid[bad_i, dup_pos] = c2
            grid[i2, p2] = c
            row_sets[bad_i] = set(grid[bad_i].tolist())
            row_sets[i2] = set(grid[i2].tolist())
            resolved = True
            break
        repairs += 1
        if not resolved:
            return None

    if any(len(s) < k for s in row_sets):
        return None
    mask = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    mask[rows, grid.ravel()] = True
    return mask


def _circulant_regular_mask(n, k):
    mask = np.zeros((n, n), dtype=bool)
    for offset in range(k):
        for r in range(n):
            mask[r, (r + offset) % n] = True
    return mask


def _permuted_circulant(n, k, rng):
    base = _circulant_regular_mask(n, k)
    row_perm = rng.permutation(n)
    col_perm = rng.permutation(n)
    return base[np.ix_(row_perm, col_perm)]


def _random_regular_mask(n, k, seed):
    if k < 1 or k > n:
        raise ValueError("bad k")
    if k == 1 and n > 1:
        raise ValueError("no connected 1-regular mask")
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    for _ in range(50):
        mask = _sample_config_model(n, k, rng)
        if mask is not None and _is_connected_bipartite(mask):
            return mask
    return _permuted_circulant(n, k, rng)


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
        t0 = time.perf_counter()
        timeout = float(budget.get("attack_timeout_s", 1.0))
        deadline = t0 + 0.55 * timeout
        X = np.asarray(X, dtype=float)
        Z = np.asarray(Z, dtype=float)
        n = X.shape[0]
        k = int(k)
        try:
            out = self._clustered_attack(X, Z, k, seed, deadline)
            if out.shape != (n, n):
                return _random_regular_mask(n, k, seed)
            if not (np.all(out.sum(axis=1) == k) and np.all(out.sum(axis=0) == k)):
                return _random_regular_mask(n, k, seed)
            if not _is_connected_bipartite(out):
                return _random_regular_mask(n, k, seed)
            return out
        except Exception:
            return _random_regular_mask(n, k, seed)

    def _clustered_attack(self, X, Z, k, seed, deadline):
        n = X.shape[0]
        if n == 0 or k <= 0 or k > n or Z.shape[0] != n:
            return _random_regular_mask(n, max(1, min(k, max(n, 1))), seed)

        max_c = 5
        clusters = 1
        for c in range(max_c, 0, -1):
            if n // c > k + max(3, k // 3):
                clusters = c
                break

        if clusters <= 1 or time.perf_counter() > deadline:
            return _random_regular_mask(n, k, seed)

        try:
            Xc = X - X.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
            proj_x = X @ vt[0]
        except Exception:
            proj_x = X[:, 0]
        try:
            Zc = Z - Z.mean(axis=0, keepdims=True)
            _, _, vtz = np.linalg.svd(Zc, full_matrices=False)
            proj_z = Z @ vtz[0]
        except Exception:
            proj_z = Z[:, 0]

        row_order = np.argsort(proj_x)
        col_order = np.argsort(proj_z)

        sizes = [n // clusters + (1 if i < n % clusters else 0) for i in range(clusters)]
        row_groups = []
        col_groups = []
        start = 0
        for size in sizes:
            row_groups.append(row_order[start:start + size])
            col_groups.append(col_order[start:start + size])
            start += size

        mask = np.zeros((n, n), dtype=bool)
        for c_idx in range(clusters):
            if time.perf_counter() > deadline:
                return _random_regular_mask(n, k, seed)
            rows_c = row_groups[c_idx]
            cols_c = col_groups[c_idx]
            size_c = len(rows_c)
            sub_seed = (int(seed) * 1_000_003 + c_idx * 7919 + 11) % (2**31 - 1)
            sub = _random_regular_mask(size_c, k, sub_seed)
            mask[np.ix_(rows_c, cols_c)] = sub

        for c_idx in range(clusters - 1):
            if time.perf_counter() > deadline:
                break
            rows_a = row_groups[c_idx]
            cols_a = col_groups[c_idx]
            rows_b = row_groups[c_idx + 1]
            cols_b = col_groups[c_idx + 1]
            r1 = rows_a[0]
            r2 = rows_b[min(1, len(rows_b) - 1)]
            c1_cands = cols_a[mask[r1, cols_a]]
            c2_cands = cols_b[mask[r2, cols_b]]

            done = False
            for c1 in c1_cands:
                for c2 in c2_cands:
                    if not mask[r1, c2] and not mask[r2, c1]:
                        mask[r1, c1] = False
                        mask[r2, c2] = False
                        mask[r1, c2] = True
                        mask[r2, c1] = True
                        done = True
                        break
                if done:
                    break

        return mask

    def _fit_bilinear(self, X, Z, rows, cols, y, alpha):
        n, d = X.shape
        m = Z.shape[0]
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
