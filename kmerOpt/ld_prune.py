"""LD pruning module."""

import numpy as np
from sklearn.preprocessing import StandardScaler


def ld_prune(X: np.ndarray, r2_threshold: float = 0.7,
             max_kept: int = 5000, n_jobs: int = 1) -> np.ndarray:
    """Greedy LD pruning of k-mer dosage matrix.

    Algorithm:
        1. Sort markers by variance (descending)
        2. Iterate: keep marker if |r| < r2_threshold with all kept markers
        3. Stop when max_kept reached or all markers processed

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_kmers)
        K-mer dosage matrix (standardized internally).
    r2_threshold : float
        Pearson |r| threshold. Default 0.7.
        - 0.3-0.5: conservative (few markers retained)
        - 0.5-0.7: moderate (recommended for GWAS)
        - 0.7-0.9: liberal (keeps more markers)
    max_kept : int
        Maximum number of markers to retain.
    n_jobs : int
        Reserved for future parallelization.

    Returns
    -------
    kept_indices : np.ndarray
        Indices of LD-pruned markers.
    """
    Xs = StandardScaler().fit_transform(X)
    variance = np.var(Xs, axis=0)
    var_order = np.argsort(-variance)

    kept = []
    for i in var_order:
        if not kept:
            kept.append(i)
            continue
        # Check correlation with recently kept markers (use last 50 for speed)
        check_set = kept[-min(50, len(kept)):]
        corrs = np.abs([np.corrcoef(Xs[:, j], Xs[:, i])[0,1] for j in check_set])
        if corrs.max() < r2_threshold:
            kept.append(i)
        if len(kept) >= max_kept:
            break

    return np.array(kept)


def compute_ld_matrix(X: np.ndarray, indices: list = None) -> np.ndarray:
    """Compute pairwise LD (r) matrix for selected k-mers.

    Parameters
    ----------
    X : np.ndarray
        Standardized k-mer dosage matrix.
    indices : list, optional
        Subset of indices. If None, use all columns.

    Returns
    -------
    R : np.ndarray, shape (m, m)
        Pairwise Pearson correlation matrix.
    """
    if indices is not None:
        X = X[:, indices]
    return np.corrcoef(X.T)
