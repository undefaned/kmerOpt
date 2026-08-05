"""Haseman-Elston regression for k-mer heritability estimation."""

import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler


def haseman_elston(X: np.ndarray, y: np.ndarray,
                   n_markers_list: List[int] = None,
                   n_bootstrap: int = 100,
                   random_state: int = 42) -> Dict[int, Tuple[float, float]]:
    """Estimate h² via Haseman-Elston regression across marker counts.

    The Haseman-Elston regression estimates h² by regressing the
    cross-product of phenotypes (Yi * Yj) on the genetic relationship
    matrix (GRM) computed from k-mers:

        h² = cov(Yi*Yj, Gij) / var(Gij) / var(y)

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_kmers)
        LD-pruned k-mer dosage matrix.
    y : np.ndarray, shape (n_samples,)
        Phenotype values (will be standardized internally).
    n_markers_list : list of int
        Marker counts to test.
    n_bootstrap : int
        Bootstrap resamples for SE.
    random_state : int
        Random seed.

    Returns
    -------
    h2_curve : dict
        {n_markers: (h2_mean, h2_se)}
    """
    if n_markers_list is None:
        n_markers_list = [200, 500, 1000, 2000, 3000, 5000]

    Xs = StandardScaler().fit_transform(X)
    ys = StandardScaler().fit_transform(y.reshape(-1,1)).ravel()

    n_samples = Xs.shape[0]
    n_available = Xs.shape[1]
    t_idx = np.triu_indices(n_samples, k=1)
    yy = np.outer(ys, ys)[t_idx]
    vp = np.var(ys)

    rng = np.random.RandomState(random_state)
    h2_curve = {}

    for n_markers in n_markers_list:
        if n_markers > n_available:
            break

        h2_vals = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_available, n_markers, replace=False)
            G = Xs[:, idx] @ Xs[:, idx].T / n_markers
            gg = G[t_idx]
            sigma2_g = max(0, np.cov(yy, gg)[0,1] / np.var(gg))
            h2_vals.append(min(sigma2_g / vp, 0.99))

        h2_curve[n_markers] = (np.mean(h2_vals), np.std(h2_vals))

    return h2_curve


def variance_components(X: np.ndarray, y: np.ndarray,
                         n_markers: int = 2000,
                         random_state: int = 42) -> Dict[str, float]:
    """Decompose phenotypic variance into genetic and residual components.

    Parameters
    ----------
    X : np.ndarray
        LD-pruned k-mer dosage matrix.
    y : np.ndarray
        Phenotype values.
    n_markers : int
        Number of markers for GRM construction.
    random_state : int
        Random seed.

    Returns
    -------
    components : dict
        {'V_P': ..., 'V_G': ..., 'V_E': ..., 'h2': ...}
    """
    Xs = StandardScaler().fit_transform(X)
    ys = StandardScaler().fit_transform(y.reshape(-1,1)).ravel()

    rng = np.random.RandomState(random_state)
    idx = rng.choice(Xs.shape[1], min(n_markers, Xs.shape[1]), replace=False)

    G = Xs[:, idx] @ Xs[:, idx].T / len(idx)
    t_idx = np.triu_indices(len(y), k=1)
    yy = np.outer(ys, ys)[t_idx]
    gg = G[t_idx]

    V_G = max(0, np.cov(yy, gg)[0,1] / np.var(gg))
    V_P = np.var(ys)
    V_E = V_P - V_G

    return {
        'V_P': V_P, 'V_G': V_G, 'V_E': V_E,
        'h2': V_G / V_P if V_P > 0 else 0
    }
