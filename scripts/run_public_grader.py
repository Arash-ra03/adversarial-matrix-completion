#!/usr/bin/env python3
"""Public grader: solve-only evaluation for one or more agent files.

Usage:
    python scripts/run_public_grader.py --agents baselines/mean_agent.py baselines/ridge_agent.py --seeds 5
"""
from __future__ import annotations
import argparse
import sys
import pathlib

# Add src/ to path so the package is importable regardless of install state
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from matrix_arena.instance import SMALL, MEDIUM, LARGE
from matrix_arena.grader import run_solve_evaluation
from matrix_arena.utils import load_agent_from_file, format_table, save_csv

BUDGET_MAP = {"small": SMALL, "medium": MEDIUM, "large": LARGE}


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrix Arena Public Grader")
    parser.add_argument(
        "--agents", nargs="+", required=True,
        help="Path(s) to agent .py files",
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of evaluation seeds (default: 5)",
    )
    parser.add_argument(
        "--budgets", nargs="+", default=["small", "medium"],
        choices=list(BUDGET_MAP),
        help="Budget levels to evaluate (default: small medium)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save leaderboard CSV",
    )
    args = parser.parse_args()

    # Load agents
    agents, names = [], []
    for path in args.agents:
        try:
            agent = load_agent_from_file(path)
            name = pathlib.Path(path).stem
            agents.append(agent)
            names.append(name)
            print(f"  ✓ Loaded: {name}")
        except Exception as exc:
            print(f"  ✗ Failed to load {path}: {exc}", file=sys.stderr)

    if not agents:
        print("No valid agents loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    seeds = list(range(args.seeds))
    budgets = [BUDGET_MAP[b] for b in args.budgets]
    print(
        f"\nEvaluating {len(agents)} agent(s) | "
        f"{args.seeds} seeds | "
        f"budgets: {args.budgets}\n"
    )

    results = run_solve_evaluation(agents, names, seeds, budgets)

    # Build table rows sorted by mean_score descending
    rows = []
    for name in names:
        r = results[name]
        rows.append({
            "agent": name,
            "mean_score": f"{r['mean_score']:.4f}",
            "mean_loss": f"{r['mean_loss']:.4f}",
            "median_loss": f"{r['median_loss']:.4f}",
            "crashes": str(r["crashes"]),
            "timeouts": str(r["timeouts"]),
        })
    rows.sort(key=lambda x: float(x["mean_score"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = str(rank)

    cols = ["rank", "agent", "mean_score", "mean_loss", "median_loss", "crashes", "timeouts"]
    print(format_table(rows, cols))

    if args.output:
        save_csv(rows, args.output)
        print(f"\nLeaderboard saved to {args.output}")


if __name__ == "__main__":
    main()
