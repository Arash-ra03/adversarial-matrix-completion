import importlib.util
import sys
import pathlib
import numpy as np
import pytest
from matrix_arena.instance import generate_instance, SMALL
from matrix_arena.masks import generate_official_mask
from matrix_arena.scoring import validate_prediction, nrmse_hidden
from matrix_arena.masks import validate_mask


def _load(name: str):
    path = pathlib.Path(__file__).parent.parent / "baselines" / f"{name}.py"
    module_name = f"_test_baseline_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod.Agent()


@pytest.fixture(scope="module")
def instance():
    X, Z, Y_gt = generate_instance(0, SMALL)
    mask = generate_official_mask(X, Z, SMALL["k"], seed=0)
    Y_obs = np.where(mask, Y_gt, 0.0)
    return X, Z, Y_gt, Y_obs, mask


def _check_solve(agent, instance):
    X, Z, Y_gt, Y_obs, mask = instance
    Y_hat = agent.solve(X, Z, Y_obs, mask, SMALL, seed=0)
    ok, msg = validate_prediction(Y_hat, expected_shape=Y_gt.shape)
    assert ok, f"invalid prediction from {type(agent).__module__}: {msg}"
    return Y_hat


def _check_attack(agent, instance):
    X, Z, _, _, _ = instance
    att_mask = agent.attack(X, Z, SMALL["k"], SMALL, seed=0)
    ok, msg = validate_mask(att_mask, SMALL["k"])
    assert ok, f"invalid attack mask from {type(agent).__module__}: {msg}"
    return att_mask


def test_mean_agent_solve(instance):
    _check_solve(_load("mean_agent"), instance)


def test_mean_agent_attack(instance):
    _check_attack(_load("mean_agent"), instance)


def test_ridge_agent_solve(instance):
    _check_solve(_load("ridge_agent"), instance)


def test_ridge_agent_attack(instance):
    _check_attack(_load("ridge_agent"), instance)


def test_low_rank_agent_solve(instance):
    _check_solve(_load("low_rank_agent"), instance)


def test_low_rank_agent_attack(instance):
    _check_attack(_load("low_rank_agent"), instance)


def test_hybrid_agent_solve(instance):
    _check_solve(_load("hybrid_agent"), instance)


def test_hybrid_agent_attack(instance):
    _check_attack(_load("hybrid_agent"), instance)


def test_random_attack_agent_solve(instance):
    _check_solve(_load("random_attack_agent"), instance)


def test_random_attack_agent_attack(instance):
    _check_attack(_load("random_attack_agent"), instance)


def test_ridge_not_worse_than_mean(instance):
    """Ridge should not catastrophically fail vs mean baseline."""
    X, Z, Y_gt, Y_obs, mask = instance
    mean_agent = _load("mean_agent")
    ridge_agent = _load("ridge_agent")
    Y_mean = mean_agent.solve(X, Z, Y_obs, mask, SMALL, seed=0)
    Y_ridge = ridge_agent.solve(X, Z, Y_obs, mask, SMALL, seed=0)
    loss_mean = nrmse_hidden(Y_mean, Y_gt, mask)
    loss_ridge = nrmse_hidden(Y_ridge, Y_gt, mask)
    assert loss_ridge <= loss_mean * 2.0, (
        f"ridge ({loss_ridge:.4f}) is more than 2x worse than mean ({loss_mean:.4f})"
    )


def test_hybrid_beats_mean_on_bilinear_regime(instance):
    """Hybrid should beat mean on seed 0 (bilinear-heavy regime)."""
    X, Z, Y_gt, Y_obs, mask = instance
    mean_agent = _load("mean_agent")
    hybrid_agent = _load("hybrid_agent")
    Y_mean = mean_agent.solve(X, Z, Y_obs, mask, SMALL, seed=0)
    Y_hybrid = hybrid_agent.solve(X, Z, Y_obs, mask, SMALL, seed=0)
    loss_mean = nrmse_hidden(Y_mean, Y_gt, mask)
    loss_hybrid = nrmse_hidden(Y_hybrid, Y_gt, mask)
    assert loss_hybrid <= loss_mean * 1.5, (
        f"hybrid ({loss_hybrid:.4f}) much worse than mean ({loss_mean:.4f})"
    )


def test_all_baselines_deterministic(instance):
    """Same seed must produce the same prediction."""
    X, Z, Y_gt, Y_obs, mask = instance
    for name in ["mean_agent", "ridge_agent", "low_rank_agent", "hybrid_agent"]:
        agent = _load(name)
        Y1 = agent.solve(X, Z, Y_obs, mask, SMALL, seed=7)
        Y2 = agent.solve(X, Z, Y_obs, mask, SMALL, seed=7)
        np.testing.assert_array_equal(Y1, Y2, err_msg=f"{name} is not deterministic")
