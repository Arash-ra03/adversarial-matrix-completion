"""Minimal agent template — copy this file and implement your own Agent.

Participants must submit a single Python file containing a class called Agent
with two methods: solve() and attack().

Note on ``seed``: it is a per-call seed provided only so your own randomness can
be deterministic (the grader rewards reproducibility). It is NOT the seed used
to generate the instance — it is decoupled by a one-way function, and the graded
evaluation uses secret high-entropy generation seeds. Do not try to rebuild the
hidden matrix from ``seed`` or by brute-forcing the generator against the visible
``X``: it will not work on the real grader and only hurts your score.
"""
import numpy as np


class Agent:
    def solve(self, X, Z, Y_obs, mask, budget, seed):
        """
        Reconstruct the full matrix Y from partial observations.

        Parameters
        ----------
        X     : np.ndarray, shape (n, d)  — row features
        Z     : np.ndarray, shape (m, d)  — column features
        Y_obs : np.ndarray, shape (n, m)  — observed entries; 0 where hidden
        mask  : np.ndarray, shape (n, m), dtype bool  — True where observed
        budget: dict  — {"name", "n", "m", "d", "rank", "solve_timeout_s", ...}
        seed  : int   — deterministic seed

        Returns
        -------
        Y_hat : np.ndarray, shape (n, m)  — full prediction matrix
        """
        mean = float(Y_obs[mask].mean()) if mask.any() else 0.0
        return np.full_like(Y_obs, mean, dtype=float)

    def attack(self, X, Z, k, budget, seed):
        """
        Choose which entries the opponent may observe.

        You do NOT see Y_gt.  You only see X, Z, k, budget, seed.

        The returned mask must satisfy:
          - shape == (n, n),  dtype bool
          - each row sums to k
          - each column sums to k
          - bipartite graph induced by mask is connected

        An invalid mask is replaced by the official mask and you receive
        an attack penalty.

        Parameters
        ----------
        X      : np.ndarray, shape (n, d)
        Z      : np.ndarray, shape (m, d)
        k      : int  — observations per row and per column
        budget : dict
        seed   : int

        Returns
        -------
        mask : np.ndarray, shape (n, n), dtype bool
        """
        from matrix_arena.masks import generate_random_regular_mask
        return generate_random_regular_mask(X.shape[0], k, seed)
