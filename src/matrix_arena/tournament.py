"""Tournament and duel logic for Matrix Arena."""
from __future__ import annotations
import dataclasses
import numpy as np
from .instance import generate_instance, BudgetConfig
from .masks import generate_official_mask
from .scoring import nrmse_hidden, safe_nrmse, INVALID_LOSS
from .api import call_solve, call_attack
from .seeds import resolve_generation_seed, derive_agent_seed

EPS: float = 1e-4
ATTACK_PENALTY: float = 0.5


@dataclasses.dataclass
class DuelResult:
    seed: int
    winner: str           # "A", "B", or "draw"
    loss_A: float
    loss_B: float
    attack_A_valid: bool
    attack_B_valid: bool
    attack_A_penalty: float
    attack_B_penalty: float
    solve_A_crashed: bool
    solve_B_crashed: bool
    solve_A_timed_out: bool
    solve_B_timed_out: bool
    error_A: str
    error_B: str
    # Optional rich record of every intermediate array, populated only when
    # ``run_duel(..., capture=True)``. Kept out of the default path so the
    # round-robin runners stay lightweight. See ``run_duel`` for the schema.
    capture: dict | None = None


def run_duel(
    agent_A,
    agent_B,
    seed: int,
    budget: BudgetConfig,
    capture: bool = False,
) -> DuelResult:
    """Run one duel between agent_A and agent_B on a shared hidden matrix.

    Both agents receive the same X, Z, Y_gt but different observation masks
    chosen by the opponent.

    If ``capture`` is True, the returned ``DuelResult.capture`` is a dict with
    every array needed to visualize the match::

        {
            "X", "Z", "Y_gt",            # shared instance
            "mask_for_A", "mask_for_B",  # opponent-chosen observation masks
            "Y_obs_A", "Y_obs_B",        # masked inputs each agent received
            "Y_hat_A", "Y_hat_B",        # predictions (None if invalid/crashed)
            "raw_loss_A", "raw_loss_B",  # NRMSE before attack penalties
            "regime", "k", "budget_name",
            "attack_A_used_official", "attack_B_used_official",
        }
    """
    # The generation seed is private to the grader; agents only ever receive
    # decoupled, one-way-derived seeds (see matrix_arena.seeds) so they cannot
    # rebuild Y_gt via generate_instance.
    gen_seed = resolve_generation_seed(seed)
    X, Z, Y_gt = generate_instance(gen_seed, budget)
    k = budget["k"]

    # Attack phase — distinct decoupled seeds for A and B.
    attack_seed_A = derive_agent_seed(gen_seed, "attack_A")
    attack_seed_B = derive_agent_seed(gen_seed, "attack_B")

    attack_res_A = call_attack(agent_A, X, Z, k, budget, attack_seed_A)
    attack_res_B = call_attack(agent_B, X, Z, k, budget, attack_seed_B)

    # Fall back to official mask when attack is invalid
    official = generate_official_mask(X, Z, k, seed=gen_seed)
    mask_for_B = attack_res_A.mask if attack_res_A.valid else official
    mask_for_A = attack_res_B.mask if attack_res_B.valid else official

    # Penalty on the agent whose attack was invalid
    # (invalid attack on opponent means opponent's mask is official, not adversarial)
    penalty_A = ATTACK_PENALTY if not attack_res_A.valid else 0.0
    penalty_B = ATTACK_PENALTY if not attack_res_B.valid else 0.0

    # Solve phase — both agents get the same decoupled solve seed.
    solve_seed = derive_agent_seed(gen_seed, "solve")
    Y_obs_A = np.where(mask_for_A, Y_gt, 0.0)
    Y_obs_B = np.where(mask_for_B, Y_gt, 0.0)

    solve_res_A = call_solve(agent_A, X, Z, Y_obs_A, mask_for_A, budget, solve_seed)
    solve_res_B = call_solve(agent_B, X, Z, Y_obs_B, mask_for_B, budget, solve_seed)

    # Compute raw losses
    raw_loss_A = (
        nrmse_hidden(solve_res_A.Y_hat, Y_gt, mask_for_A)
        if solve_res_A.valid
        else INVALID_LOSS
    )
    raw_loss_B = (
        nrmse_hidden(solve_res_B.Y_hat, Y_gt, mask_for_B)
        if solve_res_B.valid
        else INVALID_LOSS
    )

    loss_A = raw_loss_A + penalty_A
    loss_B = raw_loss_B + penalty_B

    # Determine winner with epsilon margin
    if loss_A < loss_B * (1 - EPS):
        winner = "A"
    elif loss_B < loss_A * (1 - EPS):
        winner = "B"
    else:
        winner = "draw"

    capture_data: dict | None = None
    if capture:
        from .instance import get_regime
        capture_data = {
            "X": X,
            "Z": Z,
            "Y_gt": Y_gt,
            "mask_for_A": mask_for_A,
            "mask_for_B": mask_for_B,
            "Y_obs_A": Y_obs_A,
            "Y_obs_B": Y_obs_B,
            "Y_hat_A": solve_res_A.Y_hat if solve_res_A.valid else None,
            "Y_hat_B": solve_res_B.Y_hat if solve_res_B.valid else None,
            "raw_loss_A": raw_loss_A,
            "raw_loss_B": raw_loss_B,
            "regime": get_regime(gen_seed),
            "k": k,
            "budget_name": budget["name"],
            # An attack is "used official" when the opponent's attack was invalid
            # and we fell back to the official mask for this agent.
            "attack_A_used_official": not attack_res_B.valid,
            "attack_B_used_official": not attack_res_A.valid,
        }

    return DuelResult(
        seed=seed,
        winner=winner,
        loss_A=loss_A,
        loss_B=loss_B,
        attack_A_valid=attack_res_A.valid,
        attack_B_valid=attack_res_B.valid,
        attack_A_penalty=penalty_A,
        attack_B_penalty=penalty_B,
        solve_A_crashed=solve_res_A.crashed,
        solve_B_crashed=solve_res_B.crashed,
        solve_A_timed_out=solve_res_A.timed_out,
        solve_B_timed_out=solve_res_B.timed_out,
        error_A=solve_res_A.error_msg,
        error_B=solve_res_B.error_msg,
        capture=capture_data,
    )


def run_full_round_robin(
    agents: list,
    agent_names: list[str],
    seeds: list[int],
    budget: BudgetConfig,
) -> list[tuple[str, str, DuelResult]]:
    """Run every pair (i, j) where i < j for every seed."""
    results: list[tuple[str, str, DuelResult]] = []
    n = len(agents)
    for i in range(n):
        for j in range(i + 1, n):
            for seed in seeds:
                r = run_duel(agents[i], agents[j], seed, budget)
                results.append((agent_names[i], agent_names[j], r))
    return results


def run_sampled_round_robin(
    agents: list,
    agent_names: list[str],
    seeds: list[int],
    budget: BudgetConfig,
    opponents_per_agent: int = 16,
    tournament_seed: int = 0,
) -> list[tuple[str, str, DuelResult]]:
    """Each agent plays a random subset of opponents.

    Deterministic given tournament_seed. Each pair (i, j) with i < j appears
    at most once regardless of sampling.
    """
    rng = np.random.default_rng(tournament_seed)
    n = len(agents)
    scheduled: set[tuple[int, int]] = set()

    for i in range(n):
        candidates = [j for j in range(n) if j != i]
        num_opp = min(opponents_per_agent, len(candidates))
        chosen = rng.choice(candidates, size=num_opp, replace=False)
        for j in chosen:
            pair = (min(i, j), max(i, j))
            scheduled.add(pair)

    results: list[tuple[str, str, DuelResult]] = []
    for (i, j) in sorted(scheduled):
        for seed in seeds:
            r = run_duel(agents[i], agents[j], seed, budget)
            results.append((agent_names[i], agent_names[j], r))
    return results
