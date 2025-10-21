## @file mvhg.py
#  @author Sean Svihla

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

import mvhg._mvhg


def hypergeometric(
    N: int,
    K: int,
    n: int,
    num_samples: Optional[int] = None,
    num_max_iter: Optional[int] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Sample from the univariate hypergeometric distribution using the compiled C++ implementation.

    Parameters
    ----------
    N : int
        Total population size.
    K : int
        Number of "success" states in the population.
    n : int
        Number of draws (sample size).
    num_samples : int, default = 1
        Number of random variates to generate.
    num_max_iter: Optional[int], default=1000
        Maximum number of iterations of the rejection sampler.
    seed : Optional[int], default=None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (num_samples,) containing draws from Hypergeometric(N, K, n).

    Raises
    ------
    ValueError
        If any parameter violates logical constraints: 0 ≤ K ≤ N, 0 ≤ n ≤ N, num_samples > 0.
    """
    if not (0 <= K <= N):
        raise ValueError(f"K must satisfy 0 ≤ K ≤ N (got K={K}, N={N})")
    if not (0 <= n <= N):
        raise ValueError(f"n must satisfy 0 ≤ n ≤ N (got n={n}, N={N})")
    if num_samples is None:
        num_samples = 1
    elif num_samples <= 0:
        raise ValueError(f"num_samples must be positive (got {num_samples})")
    if num_max_iter is None:
        num_max_iter = 1000

    samples = mvhg._mvhg.hypergeometric(N, K, n, num_samples, num_max_iter, seed)
    return samples


def multivariate_hypergeometric(
    Ns: ArrayLike,
    N: int,
    Na: int,
    num_samples: Optional[int] = None,
    num_max_iter: Optional[int] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Sample from the multivariate hypergeometric distribution using the compiled C++ implementation.

    Parameters
    ----------
    Ns : ArrayLike
        Sequence of category sizes (e.g., number of items per category). Must sum to N.
    N : int
        Total population size.
    Na : int
        Number of draws (sample size).
    num_samples : int, default = 1
        Number of random variates to generate.
    num_max_iter: Optional[int], default=1000
        Maximum number of iterations of the rejection sampler.
    seed : Optional[int], default=None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (num_samples, len(Ns)) containing draws from the multivariate hypergeometric.

    Raises
    ------
    ValueError
        If Ns does not sum to N, or if any parameter violates logical constraints.
    """
    Ns = np.asarray(Ns, dtype=np.int64)
    if np.any(Ns < 0):
        raise ValueError("All entries in Ns must be nonnegative.")
    if Ns.sum() != N:
        raise ValueError(f"Sum of Ns must equal N (sum={Ns.sum()}, N={N})")
    if not (0 <= Na <= N):
        raise ValueError(f"Na must satisfy 0 ≤ Na ≤ N (got Na={Na}, N={N})")
    if num_samples is None:
        num_samples = 1
    elif num_samples <= 0:
        raise ValueError(f"num_samples must be positive (got {num_samples})")
    if num_max_iter is None:
        num_max_iter = 1000

    samples = mvhg._mvhg.multivariate_hypergeometric(Ns, N, Na, num_samples, num_max_iter, seed)
    return samples
