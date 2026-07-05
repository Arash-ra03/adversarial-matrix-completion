"""Public grader pipeline: solve-only evaluation loop."""
from __future__ import annotations
import numpy as np
from .instance import generate_instance, BudgetConfig, SMALL, MEDIUM, LARGE
from .masks import generate_official_mask
from .scoring import nrmse_hidden, safe_nrmse, solve_score_instance, INVALID_LOSS
from .api import call_solve
from .seeds import resolve_generation_seed, derive_agent_seed


class _MeanBaseline:
    """Reference mean-fill baseline used internally by the grader."""

    def solve(self, X, Z, Y_obs, mask, budget, seed):
        mean = float(Y_obs[mask].mean()) if mask.any() else 0.0
        return np.full_like(Y_obs, mean, dtype=float)

    def attack(self, X, Z, k, budget, seed):
        from .masks import generate_random_regular_mask
        return generate_random_regular_mask(X.shape[0], k, seed)


def run_solve_evaluation(
    agents: list,
    agent_names: list[str],
    seeds: list[int],
    budgets: list[BudgetConfig],
    budget_weights: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Evaluate agents on solve-only tasks using official observation masks.

    Each agent receives the same official mask as every other agent for fair
    comparison. Returns a per-agent stats dictionary.
    """
    if budget_weights is None:
        budget_weights = {"small": 0.30, "medium": 0.45, "large": 0.25}

    mean_baseline = _MeanBaseline()
    stats: dict[str, dict] = {
        name: {
            "losses": [],
            "scores": [],
            "crashes": 0,
            "timeouts": 0,
        }
        for name in agent_names
    }

    for budget in budgets:
        bname = budget["name"]
        bw = budget_weights.get(bname, 1.0 / max(len(budgets), 1))

        for index in seeds:
            # The generation seed is private to the grader; agents receive only a
            # decoupled per-call seed (see matrix_arena.seeds) so they cannot
            # rebuild Y_gt via generate_instance.
            gen_seed = resolve_generation_seed(index)
            agent_seed = derive_agent_seed(gen_seed, "solve")

            X, Z, Y_gt = generate_instance(gen_seed, budget)
            official_mask = generate_official_mask(X, Z, budget["k"], seed=gen_seed)
            Y_obs = np.where(official_mask, Y_gt, 0.0)

            # Mean baseline prediction for scoring reference
            mean_val = float(Y_gt[official_mask].mean()) if official_mask.any() else 0.0
            Y_mean = np.full_like(Y_gt, mean_val)

            for agent, name in zip(agents, agent_names):
                res = call_solve(agent, X, Z, Y_obs, official_mask, budget, agent_seed)
                s = stats[name]

                if res.crashed:
                    s["crashes"] += 1
                if res.timed_out:
                    s["timeouts"] += 1

                if res.valid:
                    loss = nrmse_hidden(res.Y_hat, Y_gt, official_mask)
                    score = solve_score_instance(Y_gt, res.Y_hat, Y_mean, official_mask)
                else:
                    loss = INVALID_LOSS
                    # Score for a maximally bad prediction
                    bad_pred = np.full_like(Y_gt, INVALID_LOSS)
                    score = solve_score_instance(Y_gt, bad_pred, Y_mean, official_mask)

                s["losses"].append(loss * bw)
                s["scores"].append(score * bw)

    result: dict[str, dict] = {}
    for name in agent_names:
        s = stats[name]
        losses = s["losses"]
        scores = s["scores"]
        result[name] = {
            "mean_loss": float(np.mean(losses)) if losses else INVALID_LOSS,
            "median_loss": float(np.median(losses)) if losses else INVALID_LOSS,
            "mean_score": float(np.mean(scores)) if scores else -999.0,
            "crashes": s["crashes"],
            "timeouts": s["timeouts"],
        }
    return result
