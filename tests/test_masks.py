import numpy as np
import pytest
from matrix_arena.masks import (
    validate_mask, is_connected_bipartite,
    generate_random_regular_mask, generate_official_mask,
)
from matrix_arena.instance import SMALL


def test_random_regular_mask_row_sum():
    n, k = 48, 10
    mask = generate_random_regular_mask(n, k, seed=0)
    assert mask.shape == (n, n)
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask.sum(axis=1), k)


def test_random_regular_mask_varies_with_seed():
    """Regression: the generator must produce genuinely different masks per seed.

    A buggy "k independent permutations" construction collided constantly and
    always fell back to one fixed circulant pattern, which silently broke the
    entire attack phase (all agents got identical masks).
    """
    n, k = 48, 10
    m_a = generate_random_regular_mask(n, k, seed=1)
    m_b = generate_random_regular_mask(n, k, seed=2)
    m_c = generate_random_regular_mask(n, k, seed=3)
    assert not np.array_equal(m_a, m_b)
    assert not np.array_equal(m_a, m_c)
    # Each is still a valid mask.
    for m in (m_a, m_b, m_c):
        ok, msg = validate_mask(m, k)
        assert ok, msg


@pytest.mark.parametrize("n,k", [(48, 10), (96, 12), (160, 16)])
def test_random_regular_mask_valid_all_budgets(n, k):
    """Valid k-regular connected masks across every official budget size."""
    mask = generate_random_regular_mask(n, k, seed=7)
    ok, msg = validate_mask(mask, k)
    assert ok, msg


def test_random_regular_mask_deterministic():
    mask1 = generate_random_regular_mask(64, 8, seed=11)
    mask2 = generate_random_regular_mask(64, 8, seed=11)
    np.testing.assert_array_equal(mask1, mask2)


def test_random_regular_mask_rejects_degenerate_k():
    """k=1 admits no connected mask for n>1, and k must be in [1, n]."""
    with pytest.raises(ValueError):
        generate_random_regular_mask(8, 1, seed=0)   # 1-regular is disconnected
    with pytest.raises(ValueError):
        generate_random_regular_mask(8, 9, seed=0)   # k > n is impossible
    with pytest.raises(ValueError):
        generate_random_regular_mask(8, 0, seed=0)   # k < 1 is nonsensical


def test_random_regular_mask_col_sum():
    n, k = 48, 10
    mask = generate_random_regular_mask(n, k, seed=0)
    np.testing.assert_array_equal(mask.sum(axis=0), k)


def test_random_regular_mask_connected():
    mask = generate_random_regular_mask(48, 10, seed=0)
    assert is_connected_bipartite(mask)


def test_validate_mask_valid():
    mask = generate_random_regular_mask(48, 10, seed=0)
    ok, msg = validate_mask(mask, 10)
    assert ok, msg


def test_validate_mask_wrong_row_sum():
    n, k = 6, 2
    mask = np.zeros((n, n), dtype=bool)
    mask[0, 0] = True  # only 1 entry in row 0
    ok, msg = validate_mask(mask, k)
    assert not ok
    assert "row" in msg.lower()


def test_validate_mask_wrong_dtype():
    n, k = 6, 2
    mask = np.zeros((n, n), dtype=float)
    ok, msg = validate_mask(mask, k)
    assert not ok


def test_official_mask_is_valid():
    from matrix_arena.instance import generate_instance
    X, Z, _ = generate_instance(0, SMALL)
    mask = generate_official_mask(X, Z, SMALL["k"], seed=0)
    ok, msg = validate_mask(mask, SMALL["k"])
    assert ok, msg


def test_determinism():
    from matrix_arena.instance import generate_instance
    X, Z, _ = generate_instance(0, SMALL)
    m1 = generate_official_mask(X, Z, SMALL["k"], seed=5)
    m2 = generate_official_mask(X, Z, SMALL["k"], seed=5)
    np.testing.assert_array_equal(m1, m2)


def test_disconnected_mask_rejected():
    n, k = 8, 2
    # Two disjoint k-regular blocks of size 4
    mask = np.zeros((n, n), dtype=bool)
    for i in range(4):
        for offset in range(k):
            mask[i, (i + offset) % 4] = True
    for i in range(4, 8):
        for offset in range(k):
            mask[i, 4 + (i + offset) % 4] = True
    assert not is_connected_bipartite(mask)
    ok, msg = validate_mask(mask, k)
    assert not ok
    assert "connect" in msg.lower()
