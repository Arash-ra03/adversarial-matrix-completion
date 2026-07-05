"""Utility helpers: agent loading, table formatting, CSV/JSON output."""
from __future__ import annotations
import csv
import importlib.util
import json
import sys
from pathlib import Path


def load_agent_from_file(path: str | Path):
    """Dynamically load an Agent class from a Python file and return an instance."""
    path = Path(path).resolve()
    module_name = f"_arena_agent_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load agent from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "Agent"):
        raise AttributeError(f"File {path} must define class Agent")
    return mod.Agent()


def format_table(rows: list[dict], columns: list[str]) -> str:
    """Return a fixed-width ASCII table string."""
    col_widths = {c: len(c) for c in columns}
    for row in rows:
        for c in columns:
            col_widths[c] = max(col_widths[c], len(str(row.get(c, ""))))
    sep = "+" + "+".join("-" * (col_widths[c] + 2) for c in columns) + "+"
    header = "|" + "|".join(f" {c:<{col_widths[c]}} " for c in columns) + "|"
    lines = [sep, header, sep]
    for row in rows:
        line = (
            "|"
            + "|".join(f" {str(row.get(c, '')):<{col_widths[c]}} " for c in columns)
            + "|"
        )
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def save_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(data, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
