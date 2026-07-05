#!/usr/bin/env python3
"""Tournament runner: arena duels among submitted agent files.

Usage:
    python scripts/run_tournament.py --agents baselines/*.py --mode sampled --output leaderboard.csv
"""
from __future__ import annotations
import argparse
import glob
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
from matrix_arena.instance import SMALL, MEDIUM, LARGE
from matrix_arena.tournament import run_full_round_robin, run_sampled_round_robin
from matrix_arena.ranking import (
    compute_arena_points,
    fit_bradley_terry,
    bt_to_elo,
    compute_final_scores,
)
from matrix_arena.utils import load_agent_from_file, format_table, save_csv

BUDGET_MAP = {"small": SMALL, "medium": MEDIUM, "large": LARGE}


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrix Arena Tournament Runner")
    parser.add_argument("--agents", nargs="+", required=True)
    parser.add_argument("--mode", default="sampled", choices=["full", "sampled"])
    parser.add_argument("--opponents-per-agent", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--budgets", nargs="+", default=["small"],
        choices=list(BUDGET_MAP),
    )
    parser.add_argument("--tournament-seed", type=int, default=0)
    parser.add_argument("--output", default="leaderboard.csv")
    args = parser.parse_args()

    # Expand globs
    agent_paths: list[str] = []
    for pat in args.agents:
        expanded = glob.glob(pat)
        agent_paths.extend(expanded if expanded else [pat])
    agent_paths = sorted(set(agent_paths))

    agents, names = [], []
    for path in agent_paths:
        try:
            agent = load_agent_from_file(path)
            name = pathlib.Path(path).stem
            agents.append(agent)
            names.append(name)
            print(f"  ✓ Loaded: {name}")
        except Exception as exc:
            print(f"  ✗ Failed to load {path}: {exc}", file=sys.stderr)

    if len(agents) < 2:
        print("Need at least 2 agents to run a tournament.", file=sys.stderr)
        sys.exit(1)

    n = len(agents)
    seeds = list(range(args.seeds))
    budgets = [BUDGET_MAP[b] for b in args.budgets]

    print(
        f"\nTournament: {n} agents | mode={args.mode} | "
        f"{args.seeds} seeds | budgets={args.budgets}\n"
    )

    all_results: list = []
    for budget in budgets:
        if args.mode == "full":
            res = run_full_round_robin(agents, names, seeds, budget)
        else:
            res = run_sampled_round_robin(
                agents, names, seeds, budget,
                opponents_per_agent=args.opponents_per_agent,
                tournament_seed=args.tournament_seed,
            )
        all_results.extend(res)

    # Aggregate pairwise wins/draws/losses matrices
    name_idx = {name: i for i, name in enumerate(names)}
    wins_mat = np.zeros((n, n))
    draws_mat = np.zeros((n, n))
    losses_mat = np.zeros((n, n))

    for name_a, name_b, duel in all_results:
        i, j = name_idx[name_a], name_idx[name_b]
        if duel.winner == "A":
            wins_mat[i, j] += 1
            losses_mat[j, i] += 1
        elif duel.winner == "B":
            wins_mat[j, i] += 1
            losses_mat[i, j] += 1
        else:
            draws_mat[i, j] += 1
            draws_mat[j, i] += 1

    total_wins = wins_mat.sum(axis=1)
    total_draws = draws_mat.sum(axis=1) / 2   # de-duplicate (stored twice)
    total_losses = losses_mat.sum(axis=1)

    arena_pts = compute_arena_points(total_wins, total_draws, total_losses)
    r_bt = fit_bradley_terry(wins_mat, draws_mat)
    elo = bt_to_elo(r_bt)

    rows = []
    for i, name in enumerate(names):
        rows.append({
            "agent": name,
            "bt_elo": float(elo[i]),
            "arena_points": float(arena_pts[i]),
            "wins": int(total_wins[i]),
            "draws": int(total_draws[i]),
            "losses": int(total_losses[i]),
        })

    rows.sort(key=lambda x: x["bt_elo"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    # Format for display
    display_rows = [
        {
            "rank": str(r["rank"]),
            "agent": r["agent"],
            "bt_elo": f"{r['bt_elo']:.1f}",
            "arena_pts": f"{r['arena_points']:.3f}",
            "wins": str(r["wins"]),
            "draws": str(r["draws"]),
            "losses": str(r["losses"]),
        }
        for r in rows
    ]
    cols = ["rank", "agent", "bt_elo", "arena_pts", "wins", "draws", "losses"]
    print(format_table(display_rows, cols))

    save_csv(display_rows, args.output)
    print(f"\nLeaderboard saved to {args.output}")


if __name__ == "__main__":
    main()
