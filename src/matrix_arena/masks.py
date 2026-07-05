"""Mask generation and validation for Matrix Arena."""
from __future__ import annotations
import numpy as np
from collections import deque


def validate_mask(mask: np.ndarray, k: int) -> tuple[bool, str]:
    """Return (is_valid, error_message). Empty string means valid."""
    if not isinstance(mask, np.ndarray):
        return False, f"mask must be np.ndarray, got {type(mask)}"
    if mask.dtype != bool:
        return False, f"mask.dtype must be bool, got {mask.dtype}"
    if mask.ndim != 2:
        return False, f"mask must be 2-D, got {mask.ndim}-D"
    n, m = mask.shape
    if n != m:
        return False, f"mask must be square (n==m), got ({n}, {m})"
    row_sums = mask.sum(axis=1)
    if not np.all(row_sums == k):
        bad = np.where(row_sums != k)[0]
        return False, f"row sum != {k} at rows {bad[:5].tolist()}"
    col_sums = mask.sum(axis=0)
    if not np.all(col_sums == k):
        bad = np.where(col_sums != k)[0]
        return False, f"col sum != {k} at cols {bad[:5].tolist()}"
    if not is_connected_bipartite(mask):
        return False, "bipartite graph induced by mask is not connected"
    return True, ""


def is_connected_bipartite(mask: np.ndarray) -> bool:
    """BFS over the bipartite graph.

    Nodes 0..n-1 are row nodes; n..n+m-1 are column nodes.
    Edge R_i -- C_j iff mask[i, j] == True.
    """
    n, m = mask.shape
    total = n + m
    adj: list[list[int]] = [[] for _ in range(total)]
    rows_i, cols_j = np.where(mask)
    for r, c in zip(rows_i.tolist(), cols_j.tolist()):
        adj[r].append(n + c)
        adj[n + c].append(r)
    visited = [False] * total
    queue: deque[int] = deque([0])
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


def generate_random_regular_mask(n: int, k: int, seed: int) -> np.ndarray:
    """Generate a random k-regular bipartite adjacency matrix (n x n, bool).

    Uses the bipartite configuration model with conflict repair: each column is
    given ``k`` "stubs", the stubs are shuffled across rows, and within-row
    duplicate columns are repaired by conflict-free swaps. The result is exactly
    k-regular (every row and column has degree ``k``) and simple, and it is
    genuinely seed-dependent.

    A naive "k independent permutations" construction collides constantly for
    moderate ``k`` and silently degenerates to one fixed pattern; pure
    edge-disjoint rejection sampling is infeasible (the t-th matching is
    accepted with probability ~e^{-t}). The configuration model avoids both.

    Connectivity (guaranteed w.h.p. for ``k >= 2``) is verified. If sampling
    repeatedly fails, a seed-varying *permuted circulant* is returned as a
    guaranteed-valid fallback for ``k >= 2`` (still seed-dependent, never a
    fixed mask).

    Raises ``ValueError`` for degenerate inputs that admit no valid mask: a
    connected 1-regular bipartite graph is impossible for ``n > 1`` (it is
    always ``n`` disjoint edges), and ``k`` must satisfy ``1 <= k <= n``.
    """
    if k < 1 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n; got k={k}, n={n}")
    if k == 1 and n > 1:
        raise ValueError(
            f"no connected 1-regular bipartite mask exists for n={n} > 1"
        )
    rng = np.random.default_rng(seed)
    for _attempt in range(50):
        mask = _sample_config_model(n, k, rng)
        if mask is None:
            continue
        if is_connected_bipartite(mask):
            return mask
    return _permuted_circulant(n, k, rng)


def _sample_config_model(n: int, k: int, rng) -> np.ndarray | None:
    """Configuration-model k-regular bipartite graph with conflict repair.

    Returns an (n, n) bool mask or None if duplicates could not be repaired.
    """
    # Each column appears exactly k times; shuffle and deal k columns per row.
    stubs = np.repeat(np.arange(n), k)
    rng.shuffle(stubs)
    grid = stubs.reshape(n, k)

    row_sets = [set(grid[i].tolist()) for i in range(n)]
    # Rebuild any row that holds duplicates: it has < k distinct columns.
    max_repairs = 200 * n
    repairs = 0
    while repairs < max_repairs:
        # Find a row with a within-row duplicate.
        bad_i = -1
        for i in range(n):
            if len(row_sets[i]) < k:
                bad_i = i
                break
        if bad_i == -1:
            break  # every row has k distinct columns

        row = grid[bad_i]
        # Identify a position holding a duplicated value in this row.
        seen: set[int] = set()
        dup_pos = -1
        for p in range(k):
            if row[p] in seen:
                dup_pos = p
                break
            seen.add(int(row[p]))
        c = int(row[dup_pos])

        # Find a donor (i2, p2) such that swapping resolves the conflict
        # without creating a new within-row duplicate on either side.
        resolved = False
        for _try in range(100):
            i2 = int(rng.integers(n))
            if i2 == bad_i:
                continue
            p2 = int(rng.integers(k))
            c2 = int(grid[i2, p2])
            if c2 in row_sets[bad_i]:
                continue  # would duplicate in bad_i
            if c in row_sets[i2]:
                continue  # would duplicate in i2
            # Perform the swap and update bookkeeping.
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
    # Sanity: column degrees are preserved by construction (k stubs each), but
    # repairs only swap columns between rows, never change column multiplicity.
    return mask


def _circulant_mask(n: int, k: int) -> np.ndarray:
    """Circulant k-regular mask — always connected for k >= 2 and n > k."""
    mask = np.zeros((n, n), dtype=bool)
    for offset in range(k):
        for r in range(n):
            mask[r, (r + offset) % n] = True
    return mask


def _permuted_circulant(n: int, k: int, rng) -> np.ndarray:
    """Randomly relabeled circulant — always valid, connected, seed-varying."""
    base = _circulant_mask(n, k)
    row_perm = rng.permutation(n)
    col_perm = rng.permutation(n)
    return base[np.ix_(row_perm, col_perm)]


def generate_official_mask(
    X: np.ndarray,
    Z: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """Deterministic official mask — random regular, not adversarial.

    X and Z are provided for API compatibility; they are not used since
    the official mask is purely random regular (fair and unbiased).
    """
    n = X.shape[0]
    return generate_random_regular_mask(n, k, seed=seed)
