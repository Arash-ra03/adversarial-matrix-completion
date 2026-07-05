import numpy as np
import pytest
from matrix_arena.instance import generate_instance, SMALL
from matrix_arena.tournament import run_duel, DuelResult
from matrix_arena.masks import generate_random_regular_mask


class _PerfectAgent:
    """Solves perfectly by knowing Y_gt from closure (test only)."""
    def __init__(self, Y_gt):
        self._Y_gt = Y_gt

    def solve(self, X, Z, Y_obs, mask, budget, seed):
        return self._Y_gt.copy()

    def attack(self, X, Z, k, budget, seed):
        return generate_random_regular_mask(X.shape[0], k, seed)


class _MeanAgent:
    def solve(self, X, Z, Y_obs, mask, budget, seed):
        mean = float(Y_obs[mask].mean()) if mask.any() else 0.0
        return np.full_like(Y_obs, mean)

    def attack(self, X, Z, k, budget, seed):
        return generate_random_regular_mask(X.shape[0], k, seed)


class _BadAttacker:
    """Returns an invalid mask (all zeros)."""
    def solve(self, X, Z, Y_obs, mask, budget, seed):
        return Y_obs.copy()

    def attack(self, X, Z, k, budget, seed):
        return np.zeros((X.shape[0], X.shape[0]), dtype=bool)  # invalid


class _CrashingSolver:
    """Crashes in solve()."""
    def solve(self, X, Z, Y_obs, mask, budget, seed):
        raise RuntimeError("intentional crash")

    def attack(self, X, Z, k, budget, seed):
        return generate_random_regular_mask(X.shape[0], k, seed)


def test_duel_returns_result():
    mean_a = _MeanAgent()
    mean_b = _MeanAgent()
    result = run_duel(mean_a, mean_b, seed=0, budget=SMALL)
    assert isinstance(result, DuelResult)


def test_duel_has_winner_field():
    mean_a = _MeanAgent()
    mean_b = _MeanAgent()
    result = run_duel(mean_a, mean_b, seed=0, budget=SMALL)
    assert result.winner in ("A", "B", "draw")


def test_perfect_agent_beats_mean():
    X, Z, Y_gt = generate_instance(0, SMALL)
    perfect = _PerfectAgent(Y_gt)
    mean_agent = _MeanAgent()
    result = run_duel(perfect, mean_agent, seed=0, budget=SMALL)
    assert result.winner == "A", f"Expected A to win, got {result.winner}, loss_A={result.loss_A:.4f}, loss_B={result.loss_B:.4f}"


def test_duel_deterministic():
    mean_a = _MeanAgent()
    mean_b = _MeanAgent()
    r1 = run_duel(mean_a, mean_b, seed=42, budget=SMALL)
    r2 = run_duel(mean_a, mean_b, seed=42, budget=SMALL)
    assert r1.winner == r2.winner
    assert abs(r1.loss_A - r2.loss_A) < 1e-12
    assert abs(r1.loss_B - r2.loss_B) < 1e-12


def test_invalid_attack_is_flagged():
    bad = _BadAttacker()
    mean_agent = _MeanAgent()
    result = run_duel(bad, mean_agent, seed=0, budget=SMALL)
    assert result.attack_A_valid is False


def test_invalid_attack_applies_penalty():
    bad = _BadAttacker()
    mean_agent = _MeanAgent()
    result = run_duel(bad, mean_agent, seed=0, budget=SMALL)
    assert result.attack_A_penalty > 0


def test_crash_in_solve_does_not_raise():
    crasher = _CrashingSolver()
    mean_agent = _MeanAgent()
    result = run_duel(crasher, mean_agent, seed=0, budget=SMALL)
    assert result.solve_A_crashed is True
    assert result.winner == "B"


def test_round_robin_count():
    from matrix_arena.tournament import run_full_round_robin
    agents = [_MeanAgent(), _MeanAgent(), _MeanAgent()]
    names = ["a", "b", "c"]
    seeds = [0, 1]
    results = run_full_round_robin(agents, names, seeds, SMALL)
    # 3 agents: C(3,2)=3 pairs * 2 seeds = 6 duels
    assert len(results) == 6


def test_sampled_round_robin_deterministic():
    from matrix_arena.tournament import run_sampled_round_robin
    agents = [_MeanAgent() for _ in range(5)]
    names = [f"a{i}" for i in range(5)]
    r1 = run_sampled_round_robin(agents, names, [0], SMALL,
                                  opponents_per_agent=2, tournament_seed=7)
    r2 = run_sampled_round_robin(agents, names, [0], SMALL,
                                  opponents_per_agent=2, tournament_seed=7)
    pairs1 = [(a, b) for a, b, _ in r1]
    pairs2 = [(a, b) for a, b, _ in r2]
    assert pairs1 == pairs2
