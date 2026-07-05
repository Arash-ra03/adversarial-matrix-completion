"""Tests for the match visualization module.

These tests skip cleanly when matplotlib is not installed, so the core test
suite never requires it.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

# Skip the whole module unless matplotlib is available.
pytest.importorskip("matplotlib")

from matrix_arena.instance import generate_instance, get_regime, REGIME_NAMES, SMALL
from matrix_arena.masks import generate_official_mask
from matrix_arena.tournament import run_duel
from matrix_arena import viz


def _load_baseline(name: str):
    path = pathlib.Path(__file__).parent.parent / "baselines" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_viz_baseline_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_viz_baseline_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod.Agent()


# --- get_regime metadata ---------------------------------------------------

def test_get_regime_matches_seed_selection():
    reg = get_regime(0)
    assert reg["index"] == 0
    assert reg["name"] == REGIME_NAMES[0]
    # seed 6 wraps back to regime 0
    assert get_regime(6)["index"] == get_regime(0)["index"]


def test_get_regime_has_weights():
    reg = get_regime(3)
    for key in ["w_bilinear", "w_latent", "w_nonlinear", "noise_std"]:
        assert key in reg


# --- plot_solve_row --------------------------------------------------------

def test_plot_solve_row_returns_figure_with_five_panels():
    X, Z, Y_gt = generate_instance(0, SMALL)
    mask = generate_official_mask(X, Z, SMALL["k"], seed=0)
    Y_hat = Y_gt + 0.1
    fig = viz.plot_solve_row(Y_gt, mask, Y_hat, agent_label="test")
    assert len(fig.axes) == 5
    # Use the public accessor and assert the header actually carries the label.
    assert "test" in fig.get_suptitle()


def test_plot_solve_row_handles_invalid_prediction():
    X, Z, Y_gt = generate_instance(0, SMALL)
    mask = generate_official_mask(X, Z, SMALL["k"], seed=0)
    # None prediction (crash / invalid) must not raise
    fig = viz.plot_solve_row(Y_gt, mask, None, agent_label="crashed")
    assert len(fig.axes) == 5


def test_visualize_solve_saves_png(tmp_path):
    agent = _load_baseline("hybrid_agent")
    out = tmp_path / "solve.png"
    fig = viz.visualize_solve(agent, 0, SMALL, agent_label="hybrid", path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    assert fig is not None


# --- plot_duel -------------------------------------------------------------

def test_plot_duel_returns_two_row_figure():
    a = _load_baseline("hybrid_agent")
    b = _load_baseline("low_rank_agent")
    result = run_duel(a, b, seed=0, budget=SMALL, capture=True)
    assert result.capture is not None
    fig = viz.plot_duel(result.capture, name_A="hybrid", name_B="low_rank",
                        winner=result.winner, loss_A=result.loss_A, loss_B=result.loss_B)
    # 2 rows x 5 cols
    assert len(fig.axes) == 10


def test_visualize_duel_saves_png(tmp_path):
    a = _load_baseline("ridge_agent")
    b = _load_baseline("mean_agent")
    out = tmp_path / "duel.png"
    viz.visualize_duel(a, b, 0, SMALL, name_A="ridge", name_B="mean", path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_capture_default_is_none():
    """run_duel without capture=True must not populate capture (lightweight path)."""
    a = _load_baseline("mean_agent")
    b = _load_baseline("mean_agent")
    result = run_duel(a, b, seed=0, budget=SMALL)
    assert result.capture is None


def test_capture_contains_expected_keys():
    a = _load_baseline("hybrid_agent")
    b = _load_baseline("ridge_agent")
    cap = run_duel(a, b, seed=1, budget=SMALL, capture=True).capture
    # Assert the full published schema — plot_duel hard-indexes these keys, so a
    # silently dropped key must fail here rather than KeyError at render time.
    for key in ["X", "Z", "Y_gt", "mask_for_A", "mask_for_B",
                "Y_obs_A", "Y_obs_B", "Y_hat_A", "Y_hat_B",
                "raw_loss_A", "raw_loss_B", "regime", "k", "budget_name",
                "attack_A_used_official", "attack_B_used_official"]:
        assert key in cap, f"missing capture key: {key}"
    # Both agents see the same ground truth
    assert cap["Y_gt"].shape == (SMALL["n"], SMALL["m"])


def test_duel_gives_each_agent_a_distinct_mask():
    """Core regression: when both attacks are valid, each agent must receive a
    DIFFERENT observation mask. A prior bug made every mask identical, which
    silently nullified the entire attack phase."""
    from matrix_arena.masks import validate_mask
    a = _load_baseline("mean_agent")
    b = _load_baseline("mean_agent")
    cap = run_duel(a, b, seed=4, budget=SMALL, capture=True).capture
    assert not np.array_equal(cap["mask_for_A"], cap["mask_for_B"])
    assert validate_mask(cap["mask_for_A"], SMALL["k"])[0]
    assert validate_mask(cap["mask_for_B"], SMALL["k"])[0]


def test_error_panel_survives_nonfinite_prediction():
    """A misbehaving agent (NaN/Inf prediction) must not blank the error panel:
    the color scale must stay finite rather than silently collapsing."""
    X, Z, Y_gt = generate_instance(0, SMALL)
    mask = generate_official_mask(X, Z, SMALL["k"], seed=0)
    Y_hat = Y_gt.copy()
    hidden_idx = np.argwhere(~mask)
    Y_hat[tuple(hidden_idx[0])] = np.nan
    Y_hat[tuple(hidden_idx[1])] = np.inf
    fig = viz.plot_solve_row(Y_gt, mask, Y_hat, agent_label="garbage")
    err_ax = fig.axes[4]
    images = err_ax.get_images()
    assert images, "error panel should contain an image"
    _, vmax = images[0].get_clim()
    assert np.isfinite(vmax), "error color scale must stay finite under NaN/Inf"


def test_duel_visualizes_invalid_attacker(tmp_path):
    """An agent with an invalid attack still renders (official-fallback note)."""
    class _BadAttacker:
        def solve(self, X, Z, Y_obs, mask, budget, seed):
            mean = float(Y_obs[mask].mean()) if mask.any() else 0.0
            return np.full_like(Y_obs, mean, dtype=float)

        def attack(self, X, Z, k, budget, seed):
            return np.zeros((X.shape[0], X.shape[0]), dtype=bool)  # invalid

    good = _load_baseline("mean_agent")
    out = tmp_path / "duel_invalid.png"
    viz.visualize_duel(_BadAttacker(), good, 0, SMALL,
                       name_A="bad", name_B="mean", path=str(out))
    assert out.exists()


# --- regime gallery --------------------------------------------------------

def test_regime_gallery_saves_png(tmp_path):
    out = tmp_path / "gallery.png"
    fig = viz.visualize_regime_gallery(SMALL, path=str(out))
    assert out.exists()
    assert len(fig.axes) >= len(REGIME_NAMES)
