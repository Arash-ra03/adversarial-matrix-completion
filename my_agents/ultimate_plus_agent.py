"""Matrix Arena competition agent ("ultimate_plus").

Built from the strongest observed agent family:
  * ultimate_agent's robust bilinear expert,
  * v5_plus_agent's pure direct-low-rank expert,
  * a budget-adaptive clustered attack.

solve()
-------
Models the observed entries as a sum of complementary signals:
  (1) additive row/column bias        (median-polish sweeps)
  (2) inductive bilinear term         X @ W @ Z.T   (ridge on Kronecker
      features)
  (2b) ROBUST bilinear term           the same model refit with a Huber-IRLS
      re-weighting, so the handful of very large-magnitude ("spiky"-regime)
      observations cannot dominate W; the plain and robust fits are exposed
      as two separate stacked experts so the blend can lean on whichever is
      more accurate on a given instance.
  (3) direct latent low-rank term     P @ Q.T       fit directly to observed
      centered values, kept as a separate stacked expert for low-rank-heavy
      regimes.
  (4) residual latent low-rank term   P @ Q.T       (vectorised ALS on the
      bilinear residual - handles arbitrary/ragged observation patterns via
      padded batched linear solves, no python loop over rows or columns)
  (5) weak nonlinear correction       sum_l a_l tanh(x_i.p_l * z_j.q_l)
      (ridge on random tanh features, fit on the remaining residual)

Hyperparameters (ridge alpha, ALS rank/lambda, nonlinear alpha) are chosen by
a held-out grid search, then all signals are combined via an out-of-fold
(K-fold) ridge stacking regression so the blend weights are estimated
honestly (no leakage), and finally every component is refit on ALL observed
entries at the selected hyperparameters before applying the learned blend
weights - so no observed data is wasted in the final answer. Wall-clock is
checked throughout against a conservative fraction of solve_timeout_s so the
function always returns a valid array well inside the hard deadline.

attack()
--------
Builds a valid k-regular, connected observation mask that is deliberately
poorly-mixing: rows/columns are ordered along their dominant feature
direction and split into several blocks; each block gets an internally
random k-regular bipartite graph, and consecutive blocks are joined by the
minimum number of degree-preserving edge swaps needed for global
connectivity. Validity (k-regularity) cannot be weakened - the rules force it
exactly - so the only lever available to make the opponent's reconstruction
harder is the *conductance* (graph bottleneck / algebraic connectivity) of
the observation graph, and this construction minimises it while remaining
technically connected: each block is an internally well-observed but nearly
isolated sub-problem, so a global model has almost no cross-block signal to
generalise with. This is also consistent with matrix-completion sampling
theory (Candes & Recht; Candes & Tao): recoverability guarantees rely on
observations being spread with the matrix's coherence/leverage structure, and
a low-conductance, clustered sampling pattern is exactly the kind of
structured deviation from that assumption that makes the unseen entries
harder to extrapolate. Falls back to an honest random-regular mask whenever
anything is atypical, so it can never return an invalid mask.

Notes:
  * The direct-low-rank expert is intentionally guarded by time checks and
    out-of-fold stacking. If it does not help on a given mask/regime, the stack
    can give it little weight.
  * The attack remains valid-by-construction and validates before returning.
"""
from __future__ import annotations
import time
from collections import deque
import numpy as np


# ---------------------------------------------------------------------------
# Self-contained k-regular bipartite mask helpers (no external dependency;
# mirrors the grader's own generator so behaviour/guarantees match exactly).
# ---------------------------------------------------------------------------

def _is_connected_bipartite(mask: np.ndarray) -> bool:
    n, m = mask.shape
    total = n + m
    rows_i, cols_j = np.where(mask)
    adj = [[] for _ in range(total)]
    for r, c in zip(rows_i.tolist(), cols_j.tolist()):
        adj[r].append(n + c)
        adj[n + c].append(r)
    visited = [False] * total
    dq = deque([0])
    visited[0] = True
    count = 1
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                dq.append(v)
    return count == total


def _sample_config_model(n: int, k: int, rng) -> np.ndarray | None:
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
        seen: set = set()
        dup_pos = -1
        for p in range(k):
            if row[p] in seen:
                dup_pos = p
                break
            seen.add(int(row[p]))
        c = int(row[dup_pos])
        resolved = False
        for _try in range(100):
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


def _circulant_mask(n: int, k: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    for offset in range(k):
        for r in range(n):
            mask[r, (r + offset) % n] = True
    return mask


def _permuted_circulant(n: int, k: int, rng) -> np.ndarray:
    base = _circulant_mask(n, k)
    row_perm = rng.permutation(n)
    col_perm = rng.permutation(n)
    return base[np.ix_(row_perm, col_perm)]


def _random_regular_mask(n: int, k: int, seed: int) -> np.ndarray:
    if k < 1 or k > n:
        raise ValueError("bad k")
    if k == 1 and n > 1:
        raise ValueError("no connected 1-regular mask")
    rng = np.random.default_rng(seed)
    for _ in range(50):
        mask = _sample_config_model(n, k, rng)
        if mask is None:
            continue
        if _is_connected_bipartite(mask):
            return mask
    return _permuted_circulant(n, k, rng)


# ---------------------------------------------------------------------------
# Vectorised (padded) ALS - handles arbitrary / ragged observation patterns.
# ---------------------------------------------------------------------------

def _padded_index(rr: np.ndarray, cc: np.ndarray, yy: np.ndarray, n_side: int):
    """Return (padded_cols, padded_vals, weight), each shape (n_side, maxk).

    Rows with fewer than maxk observations are zero-padded; `weight` is 1
    where real and 0 where padding, so padded slots never contribute.
    """
    order = np.argsort(rr, kind="stable")
    rr_s, cc_s, yy_s = rr[order], cc[order], yy[order]
    counts = np.bincount(rr_s, minlength=n_side)
    maxk = int(counts.max()) if len(counts) else 0
    if maxk == 0:
        return (np.zeros((n_side, 1), dtype=int),
                np.zeros((n_side, 1)),
                np.zeros((n_side, 1)))
    offsets = np.concatenate(([0], np.cumsum(counts)))[:-1]
    pos_in_row = np.arange(len(rr_s)) - offsets[rr_s]
    padded_cols = np.zeros((n_side, maxk), dtype=int)
    padded_vals = np.zeros((n_side, maxk), dtype=float)
    weight = np.zeros((n_side, maxk), dtype=float)
    padded_cols[rr_s, pos_in_row] = cc_s
    padded_vals[rr_s, pos_in_row] = yy_s
    weight[rr_s, pos_in_row] = 1.0
    return padded_cols, padded_vals, weight


def _als_fit(rr, cc, yy, n, m, rank, lam, n_iter, rng):
    """ALS for Y[i,j] ~= P[i].Q[j], vectorised over ragged observations."""
    row_cols, row_vals, row_w = _padded_index(rr, cc, yy, n)
    col_rows, col_vals, col_w = _padded_index(cc, rr, yy, m)

    P = rng.standard_normal((n, rank)) * 0.01
    Q = rng.standard_normal((m, rank)) * 0.01
    eye_idx = np.arange(rank)

    for _ in range(n_iter):
        Q_obs = Q[row_cols]
        Q_obs_w = Q_obs * row_w[:, :, None]
        A = np.einsum("nkr,nks->nrs", Q_obs_w, Q_obs)
        A[:, eye_idx, eye_idx] += lam
        b = np.einsum("nkr,nk->nr", Q_obs_w, row_vals)
        P = np.linalg.solve(A, b)

        P_obs = P[col_rows]
        P_obs_w = P_obs * col_w[:, :, None]
        A2 = np.einsum("mkr,mks->mrs", P_obs_w, P_obs)
        A2[:, eye_idx, eye_idx] += lam
        b2 = np.einsum("mkr,mk->mr", P_obs_w, col_vals)
        Q = np.linalg.solve(A2, b2)

    return P, Q


def _fit_bias(rr, cc, yy, n, m, n_sweeps=8):
    mean = float(yy.mean())
    row_eff = np.zeros(n)
    col_eff = np.zeros(m)
    for _ in range(n_sweeps):
        resid = yy - mean - row_eff[rr] - col_eff[cc]
        sums = np.zeros(n); cnts = np.zeros(n)
        np.add.at(sums, rr, resid); np.add.at(cnts, rr, 1.0)
        row_eff += np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)

        resid = yy - mean - row_eff[rr] - col_eff[cc]
        sums = np.zeros(m); cnts = np.zeros(m)
        np.add.at(sums, cc, resid); np.add.at(cnts, cc, 1.0)
        col_eff += np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)
    return mean, row_eff, col_eff


def _fit_bilinear(X, Z, rr, cc, yy, alpha, d):
    Xi = X[rr]; Zj = Z[cc]
    Phi = (Xi[:, :, None] * Zj[:, None, :]).reshape(len(rr), d * d)
    A = Phi.T @ Phi
    A[np.arange(d * d), np.arange(d * d)] += alpha
    b = Phi.T @ yy
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w.reshape(d, d)


def _fit_bilinear_robust(X, Z, rr, cc, yy, alpha, d, n_iter=4, delta=1.345):
    """IRLS bilinear ridge with a Huber weight, down-weighting extreme
    (high-leverage / spiky) observations so a handful of outlier cells
    cannot dominate the fitted interaction matrix W."""
    Xi = X[rr]; Zj = Z[cc]
    Phi = (Xi[:, :, None] * Zj[:, None, :]).reshape(len(rr), d * d)
    diag = np.arange(d * d)
    w_obs = np.ones(len(rr))
    coef = np.zeros(d * d)
    for _ in range(int(n_iter)):
        Phiw = Phi * w_obs[:, None]
        A = Phiw.T @ Phi
        A[diag, diag] += alpha
        b = Phiw.T @ yy
        try:
            coef = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(A, b, rcond=None)[0]
        resid = yy - Phi @ coef
        scale = 1.4826 * float(np.median(np.abs(resid))) + 1e-6
        z = np.abs(resid) / (delta * scale)
        w_obs = np.where(z <= 1.0, 1.0, 1.0 / np.maximum(z, 1e-6))
    return coef.reshape(d, d)


def _fit_nonlinear(XP, ZQ, rr, cc, resid, alpha_nl, L):
    Phi = np.tanh(XP[rr] * ZQ[cc])
    A = Phi.T @ Phi
    A[np.arange(L), np.arange(L)] += alpha_nl
    b = Phi.T @ resid
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _nl_full_grid(XP, ZQ, a_coef, n, m):
    Y_nl = np.zeros((n, m))
    chunk = max(1, (2_000_000 // max(m, 1)))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        block = np.tanh(XP[s:e, None, :] * ZQ[None, :, :])
        Y_nl[s:e] = block @ a_coef
    return Y_nl


class Agent:
    # ------------------------------------------------------------------
    # SOLVE
    # ------------------------------------------------------------------
    def solve(self, X, Z, Y_obs, mask, budget, seed):
        t0 = time.perf_counter()
        timeout = float(budget.get("solve_timeout_s", 5.0))
        deadline = t0 + 0.55 * timeout

        n, d = X.shape
        m = Z.shape[0]
        rng = np.random.default_rng(seed)

        rows_idx, cols_idx = np.where(mask)
        N = len(rows_idx)
        if N == 0:
            return np.zeros((n, m), dtype=float)
        y_flat = Y_obs[rows_idx, cols_idx].astype(float)
        mean_val = float(y_flat.mean())

        if N < 20:
            return np.full((n, m), mean_val, dtype=float)

        rank_budget = int(budget.get("rank", 8))
        r_nl = int(budget.get("nonlinear_rank", 2))
        bname = budget.get("name", "small")
        L = int(np.clip(6 * r_nl, 8, 48))
        Pp = rng.standard_normal((d, L)) / np.sqrt(d)
        Qd = rng.standard_normal((d, L)) / np.sqrt(d)
        XP = X @ Pp
        ZQ = Z @ Qd

        # ----------------------------------------------------------------
        # Phase 1: quick single-split grid search for regularisation strength
        # ----------------------------------------------------------------
        perm = rng.permutation(N)
        n_val = int(np.clip(int(N * 0.2), 8, N - 8))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        tr_r, tr_c, tr_y = rows_idx[tr_idx], cols_idx[tr_idx], y_flat[tr_idx]
        va_r, va_c, va_y = rows_idx[val_idx], cols_idx[val_idx], y_flat[val_idx]

        alpha_grid = [0.02, 0.08, 0.3, 1.0, 3.0, 10.0]
        best_alpha, best_err = alpha_grid[2], np.inf
        Xi_tr = X[tr_r]; Zj_tr = Z[tr_c]
        Phi_tr = (Xi_tr[:, :, None] * Zj_tr[:, None, :]).reshape(len(tr_r), d * d)
        A_base = Phi_tr.T @ Phi_tr
        b_base = Phi_tr.T @ tr_y
        diag = np.arange(d * d)
        Xi_va = X[va_r]; Zj_va = Z[va_c]
        for a in alpha_grid:
            if time.perf_counter() > deadline:
                break
            A = A_base.copy(); A[diag, diag] += a
            try:
                w = np.linalg.solve(A, b_base)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(A, b_base, rcond=None)[0]
            Wc = w.reshape(d, d)
            pred = np.einsum("nd,de,ne->n", Xi_va, Wc, Zj_va)
            err = float(np.mean((pred - va_y) ** 2))
            if err < best_err:
                best_err, best_alpha = err, a

        W_screen = _fit_bilinear(X, Z, tr_r, tr_c, tr_y, best_alpha, d)
        resid_tr_screen = tr_y - np.einsum("nd,de,ne->n", Xi_tr, W_screen, Zj_tr)

        rank_grid = sorted(set([max(2, rank_budget - 2), rank_budget, rank_budget + 3]))
        lam_grid = [0.05, 0.2, 0.6, 2.0]
        best_rank, best_lam, best_err2 = rank_budget, lam_grid[1], np.inf
        Y_bil_screen_va = np.einsum("nd,de,ne->n", X[va_r], W_screen, Z[va_c])
        for rk in rank_grid:
            for lam in lam_grid:
                if time.perf_counter() > deadline:
                    break
                try:
                    Pc, Qc = _als_fit(tr_r, tr_c, resid_tr_screen, n, m, rk, lam, 24, rng)
                except Exception:
                    continue
                pred = Y_bil_screen_va + np.sum(Pc[va_r] * Qc[va_c], axis=1)
                err = float(np.mean((pred - va_y) ** 2))
                if err < best_err2:
                    best_err2, best_rank, best_lam = err, rk, lam
            if time.perf_counter() > deadline:
                break

        resid2_tr_screen = resid_tr_screen  # placeholder overwritten below if ALS succeeds
        alpha_nl_grid = [0.5, 3.0, 15.0]
        best_alpha_nl = alpha_nl_grid[1]
        if time.perf_counter() < deadline:
            try:
                Pc, Qc = _als_fit(tr_r, tr_c, resid_tr_screen, n, m, best_rank, best_lam, 40, rng)
                joint_va = Y_bil_screen_va + np.sum(Pc[va_r] * Qc[va_c], axis=1)
                resid2_tr = tr_y - (np.einsum("nd,de,ne->n", Xi_tr, W_screen, Zj_tr) +
                                    np.sum(Pc[tr_r] * Qc[tr_c], axis=1))
                best_err3 = np.inf
                for anl in alpha_nl_grid:
                    if time.perf_counter() > deadline:
                        break
                    a_coef = _fit_nonlinear(XP, ZQ, tr_r, tr_c, resid2_tr, anl, L)
                    pred_nl_va = np.sum(np.tanh(XP[va_r] * ZQ[va_c]) * a_coef, axis=1)
                    err = float(np.mean((joint_va + pred_nl_va - va_y) ** 2))
                    if err < best_err3:
                        best_err3, best_alpha_nl = err, anl
            except Exception:
                pass

        # ----------------------------------------------------------------
        # Phase 2: K-fold out-of-fold stacking (honest blend weights)
        # ----------------------------------------------------------------
        K = 4 if N >= 200 else 3
        fold_of = np.empty(N, dtype=int)
        fold_of[perm] = np.arange(N) % K

        names = [
            "mean",
            "bias",
            "direct_lr",
            "bilinear",
            "robust_bilinear",
            "joint",
            "full",
        ]
        oof = {nm: np.zeros(N) for nm in names}
        n_iter_fold = {"small": 60, "medium": 60, "large": 45}.get(bname, 55)

        fold_ok = True
        for kf in range(K):
            if time.perf_counter() > deadline:
                fold_ok = False
                break
            va_mask_f = fold_of == kf
            tr_mask_f = ~va_mask_f
            fr_r, fr_c, fr_y = rows_idx[tr_mask_f], cols_idx[tr_mask_f], y_flat[tr_mask_f]
            fv_r, fv_c = rows_idx[va_mask_f], cols_idx[va_mask_f]

            mean_f = float(fr_y.mean())
            oof["mean"][va_mask_f] = mean_f

            _, row_eff_f, col_eff_f = _fit_bias(fr_r, fr_c, fr_y, n, m)
            oof["bias"][va_mask_f] = mean_f + row_eff_f[fv_r] + col_eff_f[fv_c]

            if time.perf_counter() < deadline:
                try:
                    Pd, Qd = _als_fit(
                        fr_r, fr_c, fr_y - mean_f, n, m,
                        best_rank, max(best_lam, 0.2),
                        max(22, n_iter_fold // 2), rng
                    )
                    oof["direct_lr"][va_mask_f] = (
                        mean_f + np.sum(Pd[fv_r] * Qd[fv_c], axis=1)
                    )
                except Exception:
                    oof["direct_lr"][va_mask_f] = oof["bias"][va_mask_f]
            else:
                oof["direct_lr"][va_mask_f] = oof["bias"][va_mask_f]

            Wf = _fit_bilinear(X, Z, fr_r, fr_c, fr_y, best_alpha, d)
            bil_tr_f = np.einsum("nd,de,ne->n", X[fr_r], Wf, Z[fr_c])
            bil_va_f = np.einsum("nd,de,ne->n", X[fv_r], Wf, Z[fv_c])
            oof["bilinear"][va_mask_f] = bil_va_f

            try:
                Wrf = _fit_bilinear_robust(X, Z, fr_r, fr_c, fr_y, best_alpha, d)
                oof["robust_bilinear"][va_mask_f] = np.einsum("nd,de,ne->n", X[fv_r], Wrf, Z[fv_c])
            except Exception:
                oof["robust_bilinear"][va_mask_f] = bil_va_f

            resid_f = fr_y - bil_tr_f
            try:
                Pf, Qf = _als_fit(fr_r, fr_c, resid_f, n, m, best_rank, best_lam, n_iter_fold, rng)
                joint_va_f = bil_va_f + np.sum(Pf[fv_r] * Qf[fv_c], axis=1)
            except Exception:
                joint_va_f = bil_va_f
                Pf = Qf = None
            oof["joint"][va_mask_f] = joint_va_f

            if Pf is not None:
                try:
                    resid2_f = fr_y - (bil_tr_f + np.sum(Pf[fr_r] * Qf[fr_c], axis=1))
                    a_coef_f = _fit_nonlinear(XP, ZQ, fr_r, fr_c, resid2_f, best_alpha_nl, L)
                    nl_va_f = np.sum(np.tanh(XP[fv_r] * ZQ[fv_c]) * a_coef_f, axis=1)
                    oof["full"][va_mask_f] = joint_va_f + nl_va_f
                except Exception:
                    oof["full"][va_mask_f] = joint_va_f
            else:
                oof["full"][va_mask_f] = joint_va_f

        # ----------------------------------------------------------------
        # Phase 3: refit every component on ALL observed data
        # ----------------------------------------------------------------
        mean_all, row_eff_all, col_eff_all = _fit_bias(rows_idx, cols_idx, y_flat, n, m)
        bias_full = mean_all + row_eff_all[:, None] + col_eff_all[None, :]
        n_iter_final = {"small": 160, "medium": 150, "large": 100}.get(bname, 140)

        direct_lr_full = bias_full
        if time.perf_counter() < deadline:
            try:
                Pd_all, Qd_all = _als_fit(
                    rows_idx, cols_idx, y_flat - mean_all, n, m,
                    best_rank, max(best_lam, 0.2),
                    max(36, n_iter_final // 2), rng
                )
                direct_lr_full = mean_all + Pd_all @ Qd_all.T
            except Exception:
                direct_lr_full = bias_full

        W_all = _fit_bilinear(X, Z, rows_idx, cols_idx, y_flat, best_alpha, d)
        Y_bil = X @ W_all @ Z.T

        try:
            W_rob_all = _fit_bilinear_robust(X, Z, rows_idx, cols_idx, y_flat, best_alpha, d)
            Y_bil_robust = X @ W_rob_all @ Z.T
        except Exception:
            Y_bil_robust = Y_bil

        resid_all = y_flat - Y_bil[rows_idx, cols_idx]
        try:
            n_reps = 2 if time.perf_counter() < deadline - 0.02 else 1
            accY = np.zeros((n, m))
            reps_done = 0
            for _ in range(n_reps):
                if time.perf_counter() > deadline:
                    break
                P_all, Q_all = _als_fit(rows_idx, cols_idx, resid_all, n, m,
                                         best_rank, best_lam, n_iter_final, rng)
                accY += P_all @ Q_all.T
                reps_done += 1
            Y_lr = accY / max(reps_done, 1) if reps_done > 0 else np.zeros((n, m))
        except Exception:
            Y_lr = np.zeros((n, m))
        Y_joint = Y_bil + Y_lr

        Y_full = Y_joint
        if time.perf_counter() < deadline:
            try:
                resid2_all = y_flat - Y_joint[rows_idx, cols_idx]
                a_coef_all = _fit_nonlinear(XP, ZQ, rows_idx, cols_idx, resid2_all, best_alpha_nl, L)
                Y_nl = _nl_full_grid(XP, ZQ, a_coef_all, n, m)
                Y_full = Y_joint + Y_nl
            except Exception:
                pass

        full_grids = {
            "mean": np.full((n, m), mean_val, dtype=float),
            "bias": bias_full,
            "direct_lr": direct_lr_full,
            "bilinear": Y_bil,
            "robust_bilinear": Y_bil_robust,
            "joint": Y_joint,
            "full": Y_full,
        }

        # ----------------------------------------------------------------
        # Blend: ridge-regress true y (all N observed points) on OOF preds
        # ----------------------------------------------------------------
        M = np.stack([oof[nm] for nm in names], axis=1)
        M1 = np.concatenate([M, np.ones((N, 1))], axis=1)
        K1 = M1.shape[1]
        A = M1.T @ M1
        A[np.arange(K1), np.arange(K1)] += 1e-2 * N
        b = M1.T @ y_flat
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]

        Y_hat = sum(w[i] * full_grids[names[i]] for i in range(len(names))) + w[-1]
        Y_hat = np.nan_to_num(Y_hat, nan=mean_val, posinf=8.0, neginf=-8.0)
        Y_hat = np.clip(Y_hat, -8.0, 8.0)
        return Y_hat.astype(float)

    # ------------------------------------------------------------------
    # ATTACK
    # ------------------------------------------------------------------
    def attack(self, X, Z, k, budget, seed):
        t0 = time.perf_counter()
        timeout = float(budget.get("attack_timeout_s", 1.0))
        deadline = t0 + 0.55 * timeout
        n = X.shape[0]
        try:
            out = self._clustered_attack(X, Z, k, seed, deadline)
            row_sums = out.sum(axis=1)
            col_sums = out.sum(axis=0)
            if out.shape != (n, n) or not (np.all(row_sums == k) and np.all(col_sums == k)):
                return _random_regular_mask(n, k, seed)
            if not _is_connected_bipartite(out):
                return _random_regular_mask(n, k, seed)
            return out
        except Exception:
            return _random_regular_mask(n, k, seed)

    def _clustered_attack(self, X, Z, k, seed, deadline):
        n = X.shape[0]

        # Budget-adaptive cap. Small cannot safely support many clusters
        # because each cluster still needs more than k nodes. Medium/large can
        # use more clusters, lowering cross-block conductance.
        if n <= 64:
            max_c = 4
        elif n <= 112:
            max_c = 8
        else:
            max_c = 11
        C = 1
        for c in range(max_c, 0, -1):
            if n // c > k + max(2, k // 4):
                C = c
                break

        if C <= 1 or time.perf_counter() > deadline:
            return _random_regular_mask(n, k, seed)

        try:
            Xc = X - X.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            proj_x = X @ Vt[0]
        except Exception:
            proj_x = X[:, 0]
        try:
            Zc = Z - Z.mean(axis=0, keepdims=True)
            _, _, VtZ = np.linalg.svd(Zc, full_matrices=False)
            proj_z = Z @ VtZ[0]
        except Exception:
            proj_z = Z[:, 0]

        row_order = np.argsort(proj_x)
        col_order = np.argsort(proj_z)

        sizes = [n // C + (1 if i < n % C else 0) for i in range(C)]
        row_groups, col_groups = [], []
        start = 0
        for s in sizes:
            row_groups.append(row_order[start:start + s])
            col_groups.append(col_order[start:start + s])
            start += s

        mask = np.zeros((n, n), dtype=bool)
        for c_idx in range(C):
            if time.perf_counter() > deadline:
                return _random_regular_mask(n, k, seed)
            rows_c = row_groups[c_idx]
            cols_c = col_groups[c_idx]
            size_c = len(rows_c)
            sub = _random_regular_mask(size_c, k, seed=(seed * 1_000_003 + c_idx * 7919 + 11) % (2**31 - 1))
            mask[np.ix_(rows_c, cols_c)] = sub

        for c_idx in range(C - 1):
            if time.perf_counter() > deadline:
                break
            rows_a, cols_a = row_groups[c_idx], col_groups[c_idx]
            rows_b, cols_b = row_groups[c_idx + 1], col_groups[c_idx + 1]
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
