import numpy as np
import pytest
import mvhg


# ==============================================================================
# 1. Parameter Validation & Error Handling (Python Wrapper Guardrails)
# ==============================================================================

class TestHypergeometricValidation:
    """Test ValueError triggers in mvhg.hypergeometric."""

    def test_invalid_K_out_of_bounds(self):
        """K must satisfy 0 <= K <= N."""
        with pytest.raises(ValueError, match="K must satisfy 0 ≤ K ≤ N"):
            mvhg.hypergeometric(N=100, K=150, n=20)
        
        with pytest.raises(ValueError, match="K must satisfy 0 ≤ K ≤ N"):
            mvhg.hypergeometric(N=100, K=-5, n=20)

    def test_invalid_n_out_of_bounds(self):
        """n must satisfy 0 <= n <= N."""
        with pytest.raises(ValueError, match="n must satisfy 0 ≤ n ≤ N"):
            mvhg.hypergeometric(N=100, K=50, n=101)
            
        with pytest.raises(ValueError, match="n must satisfy 0 ≤ n ≤ N"):
            mvhg.hypergeometric(N=100, K=50, n=-1)

    def test_invalid_num_samples(self):
        """num_samples must be positive."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            mvhg.hypergeometric(N=100, K=50, n=20, num_samples=0)

        with pytest.raises(ValueError, match="num_samples must be positive"):
            mvhg.hypergeometric(N=100, K=50, n=20, num_samples=-10)


class TestMultivariateHypergeometricValidation:
    """Test ValueError triggers in mvhg.multivariate_hypergeometric."""

    def test_negative_category_sizes(self):
        """Ns elements must be non-negative."""
        with pytest.raises(ValueError, match="All entries in Ns must be nonnegative"):
            mvhg.multivariate_hypergeometric(Ns=[50, -10, 20], N=60, Na=10)

    def test_sum_Ns_mismatch_N(self):
        """Sum of Ns must strictly equal N."""
        with pytest.raises(ValueError, match="Sum of Ns must equal N"):
            mvhg.multivariate_hypergeometric(Ns=[50, 30, 20], N=150, Na=10)

    def test_invalid_Na_out_of_bounds(self):
        """Na must satisfy 0 <= Na <= N."""
        with pytest.raises(ValueError, match="Na must satisfy 0 ≤ Na ≤ N"):
            mvhg.multivariate_hypergeometric(Ns=[50, 30, 20], N=100, Na=105)

        with pytest.raises(ValueError, match="Na must satisfy 0 ≤ Na ≤ N"):
            mvhg.multivariate_hypergeometric(Ns=[50, 30, 20], N=100, Na=-5)

    def test_invalid_num_samples(self):
        """num_samples must be positive."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            mvhg.multivariate_hypergeometric(Ns=[50, 50], N=100, Na=10, num_samples=0)


# ==============================================================================
# 2. Output Shapes and Dtypes
# ==============================================================================

def test_univariate_default_shape_and_dtype():
    """Default num_samples=None should return a 1D array of shape (1,) with C integer dtype."""
    res = mvhg.hypergeometric(N=100, K=30, n=10)
    assert isinstance(res, np.ndarray)
    assert res.shape == (1,)
    assert np.issubdtype(res.dtype, np.integer)


def test_univariate_custom_samples_shape():
    """Requesting M samples should return shape (M,)."""
    num_samples = 250
    res = mvhg.hypergeometric(N=100, K=30, n=10, num_samples=num_samples)
    assert res.shape == (num_samples,)


def test_multivariate_default_shape_and_dtype():
    """Default num_samples=None should return shape (1, len(Ns))."""
    Ns = [40, 30, 20, 10]
    res = mvhg.multivariate_hypergeometric(Ns=Ns, N=100, Na=15)
    assert isinstance(res, np.ndarray)
    assert res.shape == (1, len(Ns))
    assert np.issubdtype(res.dtype, np.integer)


def test_multivariate_custom_samples_shape():
    """Requesting M samples should return shape (M, len(Ns))."""
    Ns = [40, 30, 30]
    num_samples = 75
    res = mvhg.multivariate_hypergeometric(Ns=Ns, N=100, Na=20, num_samples=num_samples)
    assert res.shape == (num_samples, len(Ns))


# ==============================================================================
# 3. Mathematical Invariants & Boundary Edge Cases
# ==============================================================================

def test_univariate_bounds():
    """Draws must fall within [max(0, n + K - N), min(K, n)]."""
    N, K, n = 100, 40, 30
    samples = mvhg.hypergeometric(N=N, K=K, n=n, num_samples=500)
    
    k_min = max(0, n + K - N)
    k_max = min(K, n)
    assert np.all(samples >= k_min)
    assert np.all(samples <= k_max)


def test_univariate_edge_cases():
    """Test boundary conditions: n=0, K=0, n=N."""
    # Zero draws -> 0 successes
    assert np.all(mvhg.hypergeometric(N=100, K=50, n=0, num_samples=10) == 0)
    # Zero success states in population -> 0 successes
    assert np.all(mvhg.hypergeometric(N=100, K=0, n=20, num_samples=10) == 0)
    # Draw entire population -> exactly K successes
    assert np.all(mvhg.hypergeometric(N=100, K=35, n=100, num_samples=10) == 35)


def test_multivariate_sum_invariant():
    """The sum of category draws across each sample row MUST equal Na."""
    Ns = [50, 30, 15, 5]
    N, Na = 100, 40
    num_samples = 300
    
    samples = mvhg.multivariate_hypergeometric(Ns=Ns, N=N, Na=Na, num_samples=num_samples)
    row_sums = np.sum(samples, axis=1)
    
    np.testing.assert_array_equal(row_sums, np.full(num_samples, Na))


def test_multivariate_category_bounds():
    """No category draw can exceed its population size or be negative."""
    Ns = np.array([40, 30, 20, 10])
    N, Na = 100, 70
    samples = mvhg.multivariate_hypergeometric(Ns=Ns, N=N, Na=Na, num_samples=200)

    assert np.all(samples >= 0)
    assert np.all(samples <= Ns)


def test_multivariate_edge_cases():
    """Test multivariate boundary conditions."""
    Ns = [40, 30, 20, 10]
    N = 100
    
    # Draw 0 items -> all categories receive 0
    zero_draws = mvhg.multivariate_hypergeometric(Ns=Ns, N=N, Na=0, num_samples=5)
    np.testing.assert_array_equal(zero_draws, np.zeros((5, 4)))
    
    # Draw full population -> output exactly matches Ns
    full_draws = mvhg.multivariate_hypergeometric(Ns=Ns, N=N, Na=N, num_samples=5)
    for row in full_draws:
        np.testing.assert_array_equal(row, Ns)


# ==============================================================================
# 4. Reproducibility & Random Seed Handling
# ==============================================================================

def test_univariate_seed_reproducibility():
    """Passing identical seed parameters must generate identical outputs."""
    res1 = mvhg.hypergeometric(N=100, K=40, n=25, num_samples=50, seed=12345)
    res2 = mvhg.hypergeometric(N=100, K=40, n=25, num_samples=50, seed=12345)
    np.testing.assert_array_equal(res1, res2)


def test_multivariate_seed_reproducibility():
    """Passing identical seed parameters must generate identical outputs."""
    Ns = [60, 40, 20]
    res1 = mvhg.multivariate_hypergeometric(Ns=Ns, N=120, Na=30, num_samples=50, seed=999)
    res2 = mvhg.multivariate_hypergeometric(Ns=Ns, N=120, Na=30, num_samples=50, seed=999)
    np.testing.assert_array_equal(res1, res2)


# ==============================================================================
# 5. Statistical Expectation & Law of Large Numbers
# ==============================================================================

def test_univariate_mean_convergence():
    """Sample mean E[X] should converge to n * (K / N)."""
    N, K, n = 200, 80, 50
    expected_mean = n * (K / N)  # 50 * (80 / 200) = 20.0
    
    samples = mvhg.hypergeometric(N=N, K=K, n=n, num_samples=50_000, seed=42)
    empirical_mean = np.mean(samples)
    
    np.testing.assert_allclose(empirical_mean, expected_mean, atol=1e-1)


def test_multivariate_mean_convergence():
    """Sample mean vector E[X_j] should converge to Na * (Ns_j / N)."""
    Ns = np.array([50, 30, 20], dtype=np.int64)
    N = 100
    Na = 40
    expected_means = Na * (Ns / N)  # [20.0, 12.0, 8.0]
    
    samples = mvhg.multivariate_hypergeometric(Ns=Ns, N=N, Na=Na, num_samples=50_000, seed=42)
    empirical_means = np.mean(samples, axis=0)
    
    np.testing.assert_allclose(empirical_means, expected_means, atol=1e-1)