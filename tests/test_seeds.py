"""Tests for seed hygiene — agents must not be able to rebuild the instance."""
import numpy as np

from matrix_arena.instance import generate_instance, SMALL
from matrix_arena.masks import generate_random_regular_mask
from matrix_arena.seeds import resolve_generation_seed, derive_agent_seed
from matrix_arena.grader import run_solve_evaluation


class _SeedCheatAgent:
    """Tries the classic exploit: rebuild Y_gt from the seed it was handed."""

    def solve(self, X, Z, Y_obs, mask, budget, seed):
        _, _, Y = generate_instance(seed, budget)
        return Y

    def attack(self, X, Z, k, budget, seed):
        return generate_random_regular_mask(X.shape[0], k, seed)


def test_agent_seed_is_decoupled_from_generation_seed():
    # The derived seed must differ from the generation seed and be deterministic.
    for g in [0, 1, 7, 42, 100]:
        a = derive_agent_seed(g, "solve")
        assert a != g
        assert a == derive_agent_seed(g, "solve")  # deterministic
    # Different roles → different seeds (no cross-call correlation to exploit).
    assert derive_agent_seed(5, "solve") != derive_agent_seed(5, "attack_A")


def test_resolve_generation_seed_identity_by_default(monkeypatch):
    monkeypatch.delenv("MATRIX_ARENA_SEED_KEY", raising=False)
    assert resolve_generation_seed(3) == 3
    assert resolve_generation_seed(99) == 99


def test_resolve_generation_seed_high_entropy_with_key(monkeypatch):
    monkeypatch.setenv("MATRIX_ARENA_SEED_KEY", "super-secret")
    s = resolve_generation_seed(3)
    assert s != 3
    assert s > 2**32  # 64-bit keyed hash, not brute-forceable from visible X
    # Deterministic given the key (so the hidden grader is reproducible).
    assert resolve_generation_seed(3) == s


def test_seed_cheat_does_not_recover_ground_truth():
    """The headline regression: rebuilding from the handed seed must fail now."""
    res = run_solve_evaluation(
        [_SeedCheatAgent()], ["cheat"], seeds=[0, 1, 2, 3],
        budgets=[SMALL], budget_weights={"small": 1.0},
    )
    # A real reconstruction would score ~0; a wrong matrix scores ~1 (NRMSE).
    assert res["cheat"]["mean_loss"] > 0.5, (
        f"seed cheat still reconstructs Y_gt (loss={res['cheat']['mean_loss']:.3f})"
    )


def test_legitimate_mean_agent_still_scores_normally():
    """The decoupled seed must not break honest agents."""
    class _Mean:
        def solve(self, X, Z, Y_obs, mask, budget, seed):
            m = float(Y_obs[mask].mean()) if mask.any() else 0.0
            return np.full_like(Y_obs, m, dtype=float)

        def attack(self, X, Z, k, budget, seed):
            return generate_random_regular_mask(X.shape[0], k, seed)

    res = run_solve_evaluation([_Mean()], ["mean"], seeds=[0, 1, 2],
                               budgets=[SMALL], budget_weights={"small": 1.0})
    # Mean baseline NRMSE is ~1 by construction, and never crashes.
    assert res["mean"]["crashes"] == 0
    assert 0.5 < res["mean"]["mean_loss"] < 1.5
