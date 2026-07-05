"""Baseline: fill all hidden entries with the observed mean."""
import numpy as np
from matrix_arena.masks import generate_random_regular_mask


class Agent:
    def solve(self, X, Z, Y_obs, mask, budget, seed):
        mean = float(Y_obs[mask].mean()) if mask.any() else 0.0
        return np.full_like(Y_obs, mean, dtype=float)

    def attack(self, X, Z, k, budget, seed):
        return generate_random_regular_mask(X.shape[0], k, seed)
