#!/usr/bin/env python3
"""Render match visualizations for Matrix Arena.

Examples
--------
Solve-only (one agent under the official mask)::

    python scripts/visualize_match.py solve \
        --agent baselines/hybrid_agent.py --seed 0 --budget small \
        --output solve.png

Duel (two agents on the same hidden matrix, each under the other's attack)::

    python scripts/visualize_match.py duel \
        --agent-a baselines/hybrid_agent.py \
        --agent-b baselines/low_rank_agent.py \
        --seed 0 --budget small --output duel.png

Regime gallery (the hidden instance family)::

    python scripts/visualize_match.py gallery --budget small --output gallery.png
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from matrix_arena.instance import SMALL, MEDIUM, LARGE
from matrix_arena.utils import load_agent_from_file

BUDGET_MAP = {"small": SMALL, "medium": MEDIUM, "large": LARGE}


def _check_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            'matplotlib is required. Install it with:\n'
            '    pip install -e ".[viz]"   (or:  pip install matplotlib)',
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrix Arena match visualizer")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_solve = sub.add_parser("solve", help="visualize one agent's reconstruction")
    p_solve.add_argument("--agent", required=True, help="path to agent .py file")
    p_solve.add_argument("--seed", type=int, default=0)
    p_solve.add_argument("--budget", choices=list(BUDGET_MAP), default="small")
    p_solve.add_argument("--output", default="solve.png")

    p_duel = sub.add_parser("duel", help="visualize a head-to-head duel")
    p_duel.add_argument("--agent-a", required=True, help="path to agent A .py file")
    p_duel.add_argument("--agent-b", required=True, help="path to agent B .py file")
    p_duel.add_argument("--seed", type=int, default=0)
    p_duel.add_argument("--budget", choices=list(BUDGET_MAP), default="small")
    p_duel.add_argument("--output", default="duel.png")

    p_gal = sub.add_parser("gallery", help="visualize the regime family")
    p_gal.add_argument("--budget", choices=list(BUDGET_MAP), default="small")
    p_gal.add_argument("--output", default="gallery.png")

    args = parser.parse_args()
    _check_matplotlib()

    from matrix_arena import viz  # imported after the matplotlib check

    budget = BUDGET_MAP[args.budget]

    if args.mode == "solve":
        agent = load_agent_from_file(args.agent)
        label = pathlib.Path(args.agent).stem
        viz.visualize_solve(agent, args.seed, budget, agent_label=label, path=args.output)
        print(f"Saved solve visualization to {args.output}")

    elif args.mode == "duel":
        agent_a = load_agent_from_file(args.agent_a)
        agent_b = load_agent_from_file(args.agent_b)
        name_a = pathlib.Path(args.agent_a).stem
        name_b = pathlib.Path(args.agent_b).stem
        viz.visualize_duel(agent_a, agent_b, args.seed, budget,
                           name_A=name_a, name_B=name_b, path=args.output)
        print(f"Saved duel visualization to {args.output}")

    elif args.mode == "gallery":
        viz.visualize_regime_gallery(budget, path=args.output)
        print(f"Saved regime gallery to {args.output}")


if __name__ == "__main__":
    main()
