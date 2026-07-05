"""Match visualization for Matrix Arena.

Renders the full story of a reconstruction task as a row of heatmaps:

    ground truth  ->  observation mask  ->  what the agent sees  ->
    prediction    ->  hidden-cell error

Two entry points cover the competition's two phases:

* :func:`visualize_solve` -- a single agent reconstructing under the official
  mask (the solve-only phase).
* :func:`visualize_duel` -- two agents solving the *same* hidden matrix under
  each other's attack masks, stacked as two rows for side-by-side comparison.

matplotlib is an optional dependency (``pip install -e ".[viz]"``); importing
this module without matplotlib raises a clear, actionable error. The core
library and its test suite never import this module, so they stay numpy-only.

The renderer uses matplotlib's object-oriented ``Figure`` API rather than
``pyplot``; it touches no global state and works in fully headless
environments without configuring a backend.
"""
from __future__ import annotations

import numpy as np

from .instance import generate_instance, get_regime, BudgetConfig
from .masks import generate_official_mask
from .scoring import nrmse_hidden
from .api import call_solve

try:  # pragma: no cover - exercised indirectly by the import-guard test
    from matplotlib.figure import Figure
    from matplotlib import colormaps
    _HAVE_MPL = True
except ImportError:  # pragma: no cover
    Figure = None  # type: ignore[assignment]
    _HAVE_MPL = False

# Colormaps. Signed data (≈ N(0,1)) uses a diverging map centered at 0; the
# absolute-error map uses a perceptually uniform sequential map.
_SIGNED_CMAP = "RdBu_r"
_ERROR_CMAP = "magma"
_MASK_CMAP = "Greys"
_BAD_COLOR = "#cccccc"  # neutral gray for "hidden / not shown" cells


def _require_mpl() -> None:
    if not _HAVE_MPL:
        raise ImportError(
            "matplotlib is required for matrix_arena.viz. "
            'Install it with:  pip install -e ".[viz]"   (or: pip install matplotlib)'
        )


def _signed_scale(Y_ref: np.ndarray) -> float:
    """Symmetric color limit for signed heatmaps, robust to outliers."""
    v = float(np.percentile(np.abs(Y_ref), 99.0))
    return v if v > 1e-8 else 1.0


def _draw_signed(ax, M: np.ndarray, vmax: float, title: str):
    """Heatmap for a fully-known signed matrix (ground truth / prediction)."""
    im = ax.imshow(M, cmap=_SIGNED_CMAP, vmin=-vmax, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _draw_mask(ax, mask: np.ndarray, title: str):
    """Binary heatmap of the observation pattern (dark = observed)."""
    im = ax.imshow(mask.astype(float), cmap=_MASK_CMAP, vmin=0.0, vmax=1.0,
                   aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _draw_visible(ax, Y_gt: np.ndarray, mask: np.ndarray, vmax: float, title: str):
    """What the agent actually receives: observed values colored, hidden gray.

    This is the literal input to ``solve`` -- only ``mask==True`` cells carry
    real values; everything else is unknown and drawn in neutral gray.
    """
    cmap = colormaps[_SIGNED_CMAP].copy()
    cmap.set_bad(_BAD_COLOR)
    shown = np.ma.masked_array(Y_gt, mask=~mask)
    im = ax.imshow(shown, cmap=cmap, vmin=-vmax, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _draw_error(ax, Y_hat, Y_gt: np.ndarray, mask: np.ndarray, title: str):
    """Absolute error on hidden cells only (observed cells drawn gray).

    Scoring uses hidden entries exclusively, so observed cells are masked out
    of the error view. If ``Y_hat`` is None (invalid / crashed solve), a clear
    placeholder is drawn instead.
    """
    cmap = colormaps[_ERROR_CMAP].copy()
    cmap.set_bad(_BAD_COLOR)
    hidden = ~mask

    if Y_hat is None:
        ax.text(0.5, 0.5, "no prediction\n(invalid / crashed)",
                ha="center", va="center", fontsize=9, color="firebrick",
                transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(_BAD_COLOR)
        return None

    err = np.abs(np.asarray(Y_hat, dtype=float) - Y_gt)
    err_shown = np.ma.masked_array(err, mask=~hidden)
    if hidden.any():
        # A misbehaving agent can return NaN/Inf; restrict the color scale to
        # finite errors so vmax stays finite. The guard is written so it also
        # fires for NaN (``not (nan > 1e-8)`` is True), which a naive
        # ``vmax <= 1e-8`` check would silently skip and blank the whole panel.
        finite = err[hidden]
        finite = finite[np.isfinite(finite)]
        vmax = float(np.percentile(finite, 99.0)) if finite.size else 1.0
        if not (vmax > 1e-8):
            vmax = 1.0
    else:
        vmax = 1.0
    im = ax.imshow(err_shown, cmap=cmap, vmin=0.0, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _draw_invalid_panel(ax, title: str):
    ax.text(0.5, 0.5, "no prediction\n(invalid / crashed)",
            ha="center", va="center", fontsize=9, color="firebrick",
            transform=ax.transAxes)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(_BAD_COLOR)


# ----------------------------------------------------------------------------
# Public figure builders
# ----------------------------------------------------------------------------

def plot_solve_row(
    Y_gt: np.ndarray,
    mask: np.ndarray,
    Y_hat,
    *,
    agent_label: str = "agent",
    subtitle: str = "",
):
    """Build a single-row figure for one reconstruction.

    Parameters
    ----------
    Y_gt : (n, m) ground-truth matrix.
    mask : (n, m) bool, True where observed.
    Y_hat : (n, m) prediction, or None for an invalid / crashed solve.
    agent_label : name shown in the figure header.
    subtitle : optional extra header line (e.g. regime / seed info).

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_mpl()
    vmax = _signed_scale(Y_gt)

    fig = Figure(figsize=(16, 3.8), layout="constrained")
    axes = fig.subplots(1, 5)

    _draw_signed(axes[0], Y_gt, vmax, "Ground truth $Y^*$")
    _draw_mask(axes[1], mask, "Observation mask (black=observed)")
    _draw_visible(axes[2], Y_gt, mask, vmax, "What the agent sees $Y_{obs}$")

    if Y_hat is None:
        _draw_invalid_panel(axes[3], f"Prediction ({agent_label})")
        nrmse = float("nan")
    else:
        _draw_signed(axes[3], np.asarray(Y_hat, dtype=float), vmax,
                     f"Prediction $\\hat{{Y}}$ ({agent_label})")
        nrmse = nrmse_hidden(np.asarray(Y_hat, dtype=float), Y_gt, mask)

    err_title = "Hidden error  NRMSE=nan" if np.isnan(nrmse) else f"Hidden error  NRMSE={nrmse:.3f}"
    _draw_error(axes[4], None if Y_hat is None else np.asarray(Y_hat, dtype=float),
                Y_gt, mask, err_title)

    header = f"Solve — {agent_label}"
    if subtitle:
        header += f"   |   {subtitle}"
    fig.suptitle(header, fontsize=12)
    return fig


def plot_duel(capture: dict, *, name_A: str = "A", name_B: str = "B",
              winner: str = "?", loss_A: float | None = None,
              loss_B: float | None = None):
    """Build a two-row figure comparing two agents on the same hidden matrix.

    ``capture`` is the dict produced by ``run_duel(..., capture=True)``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_mpl()
    Y_gt = capture["Y_gt"]
    vmax = _signed_scale(Y_gt)

    fig = Figure(figsize=(16, 8.0), layout="constrained")
    axes = fig.subplots(2, 5)

    rows = [
        (0, name_A, capture["mask_for_A"], capture["Y_hat_A"],
         capture["attack_A_used_official"]),
        (1, name_B, capture["mask_for_B"], capture["Y_hat_B"],
         capture["attack_B_used_official"]),
    ]
    for r, label, mask, Y_hat, used_official in rows:
        atk_note = " [official fallback]" if used_official else ""
        _draw_signed(axes[r][0], Y_gt, vmax, "Ground truth $Y^*$")
        _draw_mask(axes[r][1], mask, f"Mask given to {label}{atk_note}")
        _draw_visible(axes[r][2], Y_gt, mask, vmax, f"What {label} sees")
        if Y_hat is None:
            _draw_invalid_panel(axes[r][3], f"{label} prediction")
            nrmse = float("nan")
        else:
            _draw_signed(axes[r][3], np.asarray(Y_hat, dtype=float), vmax,
                         f"{label} prediction")
            nrmse = nrmse_hidden(np.asarray(Y_hat, dtype=float), Y_gt, mask)
        err_title = "Hidden error  NRMSE=nan" if np.isnan(nrmse) else f"Hidden error  NRMSE={nrmse:.3f}"
        _draw_error(axes[r][4], None if Y_hat is None else np.asarray(Y_hat, dtype=float),
                    Y_gt, mask, err_title)

    reg = capture.get("regime", {})
    la = f"{loss_A:.4f}" if loss_A is not None else "?"
    lb = f"{loss_B:.4f}" if loss_B is not None else "?"
    win_name = {"A": name_A, "B": name_B, "draw": "draw"}.get(winner, winner)
    header = (
        f"Duel: {name_A}  vs  {name_B}    "
        f"regime={reg.get('name', '?')}  k={capture.get('k', '?')}  "
        f"budget={capture.get('budget_name', '?')}"
    )
    subhead = f"loss({name_A})={la}   loss({name_B})={lb}   →  winner: {win_name}"
    fig.suptitle(header + "\n" + subhead, fontsize=12)
    return fig


def plot_regime_gallery(budget: BudgetConfig, seeds=None):
    """Build a gallery of ground-truth matrices, one per regime.

    Helpful for understanding the hidden instance family. ``seeds`` defaults to
    ``range(len(REGIME_NAMES))`` so each panel shows a distinct regime.
    """
    _require_mpl()
    from .instance import REGIME_NAMES
    if seeds is None:
        seeds = list(range(len(REGIME_NAMES)))

    n = len(seeds)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig = Figure(figsize=(4.2 * ncols, 4.2 * nrows), layout="constrained")
    axes = np.atleast_1d(fig.subplots(nrows, ncols)).ravel()

    for ax in axes[n:]:
        ax.axis("off")

    for ax, seed in zip(axes, seeds):
        _, _, Y_gt = generate_instance(seed, budget)
        reg = get_regime(seed)
        vmax = _signed_scale(Y_gt)
        _draw_signed(ax, Y_gt, vmax, f"seed={seed}  ·  {reg['name']}")

    fig.suptitle(f"Regime gallery (budget={budget['name']})", fontsize=13)
    return fig


# ----------------------------------------------------------------------------
# High-level convenience: run an agent / a duel and render it
# ----------------------------------------------------------------------------

def visualize_solve(agent, seed: int, budget: BudgetConfig, *,
                    agent_label: str = "agent", path: str | None = None):
    """Run ``agent.solve`` under the official mask and render the result.

    The agent is invoked through :func:`matrix_arena.api.call_solve`, so
    timeouts, crashes and invalid outputs are handled gracefully (the
    prediction panel shows a placeholder instead of erroring).

    Returns the matplotlib Figure. If ``path`` is given, the figure is also
    saved there.
    """
    _require_mpl()
    X, Z, Y_gt = generate_instance(seed, budget)
    mask = generate_official_mask(X, Z, budget["k"], seed=seed)
    Y_obs = np.where(mask, Y_gt, 0.0)

    res = call_solve(agent, X, Z, Y_obs, mask, budget, seed)
    Y_hat = res.Y_hat if res.valid else None

    reg = get_regime(seed)
    status = "ok"
    if res.crashed:
        status = "CRASHED"
    elif res.timed_out:
        status = "TIMEOUT"
    elif not res.valid:
        status = "INVALID"
    subtitle = (f"seed={seed}  regime={reg['name']}  budget={budget['name']}  "
                f"k={budget['k']}  status={status}")

    fig = plot_solve_row(Y_gt, mask, Y_hat, agent_label=agent_label, subtitle=subtitle)
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig


def visualize_duel(agent_A, agent_B, seed: int, budget: BudgetConfig, *,
                   name_A: str = "A", name_B: str = "B", path: str | None = None):
    """Run a captured duel and render both agents side by side.

    Returns the matplotlib Figure. If ``path`` is given, the figure is saved.
    """
    _require_mpl()
    from .tournament import run_duel
    result = run_duel(agent_A, agent_B, seed, budget, capture=True)
    fig = plot_duel(result.capture, name_A=name_A, name_B=name_B,
                    winner=result.winner, loss_A=result.loss_A, loss_B=result.loss_B)
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig


def visualize_regime_gallery(budget: BudgetConfig, *, path: str | None = None, seeds=None):
    """Render the regime gallery and optionally save it. Returns the Figure."""
    _require_mpl()
    fig = plot_regime_gallery(budget, seeds=seeds)
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig
