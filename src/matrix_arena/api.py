"""Wrappers that call agent methods with timeout protection and output validation."""
from __future__ import annotations
import signal
import time
import traceback
import numpy as np
from .masks import validate_mask
from .scoring import validate_prediction, INVALID_LOSS


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout()


def _call_with_timeout(fn, timeout_s: float, *args, **kwargs):
    """Call fn(*args, **kwargs) with a SIGALRM-based wall-clock timeout (Unix only).

    If timeout_s <= 0, calls fn directly with no timeout.
    """
    if timeout_s <= 0:
        return fn(*args, **kwargs)
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        result = fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    return result


class SolveResult:
    __slots__ = ("Y_hat", "loss", "valid", "timed_out", "crashed", "error_msg", "elapsed_s")

    def __init__(self):
        self.Y_hat: np.ndarray | None = None
        self.loss: float = INVALID_LOSS
        self.valid: bool = False
        self.timed_out: bool = False
        self.crashed: bool = False
        self.error_msg: str = ""
        self.elapsed_s: float = 0.0


class AttackResult:
    __slots__ = ("mask", "valid", "timed_out", "crashed", "error_msg", "elapsed_s")

    def __init__(self):
        self.mask: np.ndarray | None = None
        self.valid: bool = False
        self.timed_out: bool = False
        self.crashed: bool = False
        self.error_msg: str = ""
        self.elapsed_s: float = 0.0


def call_solve(
    agent,
    X: np.ndarray,
    Z: np.ndarray,
    Y_obs: np.ndarray,
    mask: np.ndarray,
    budget: dict,
    seed: int,
) -> SolveResult:
    """Call agent.solve() with timeout and output validation."""
    result = SolveResult()
    timeout_s = float(budget.get("solve_timeout_s", 5.0))
    t0 = time.perf_counter()
    try:
        Y_hat = _call_with_timeout(
            agent.solve, timeout_s, X, Z, Y_obs, mask, budget, seed
        )
        result.elapsed_s = time.perf_counter() - t0
        ok, msg = validate_prediction(Y_hat, expected_shape=Y_obs.shape)
        if ok:
            result.Y_hat = Y_hat
            result.valid = True
        else:
            result.error_msg = f"invalid prediction: {msg}"
    except _Timeout:
        result.elapsed_s = time.perf_counter() - t0
        result.timed_out = True
        result.error_msg = f"solve timed out after {timeout_s}s"
    except Exception:
        result.elapsed_s = time.perf_counter() - t0
        result.crashed = True
        result.error_msg = traceback.format_exc()
    return result


def call_attack(
    agent,
    X: np.ndarray,
    Z: np.ndarray,
    k: int,
    budget: dict,
    seed: int,
) -> AttackResult:
    """Call agent.attack() with timeout and mask validation."""
    result = AttackResult()
    timeout_s = float(budget.get("attack_timeout_s", 1.0))
    t0 = time.perf_counter()
    try:
        mask = _call_with_timeout(agent.attack, timeout_s, X, Z, k, budget, seed)
        result.elapsed_s = time.perf_counter() - t0
        ok, msg = validate_mask(mask, k)
        if ok:
            result.mask = mask
            result.valid = True
        else:
            result.error_msg = f"invalid mask: {msg}"
    except _Timeout:
        result.elapsed_s = time.perf_counter() - t0
        result.timed_out = True
        result.error_msg = f"attack timed out after {timeout_s}s"
    except Exception:
        result.elapsed_s = time.perf_counter() - t0
        result.crashed = True
        result.error_msg = traceback.format_exc()
    return result
