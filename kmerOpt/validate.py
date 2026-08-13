"""ANOVA tertile validation module."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


def anova_tertile(X: np.ndarray, y: np.ndarray,
                  kmer_ids: Optional[list] = None,
                  kmer_indices: Optional[np.ndarray] = None,
                  n_top: int = 100, alpha: float = 0.05) -> pd.DataFrame:
    """Validate k-mer markers via tertile ANOVA.

    For each of the top-n markers (by variance), split samples into
    three equal-sized groups based on k-mer dosage (tertile), then
    test whether the three groups have significantly different
    phenotype means.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_kmers)
        Raw (unstandardized) k-mer dosage matrix.
    y : np.ndarray, shape (n_samples,)
        Phenotype values.
    kmer_ids : list, optional
        K-mer names.
    kmer_indices : np.ndarray, optional
        Which columns of X to use (e.g., LD-pruned subset).
        If None, use all.
    n_top : int
        Number of top-variance markers to validate.
    alpha : float
        Significance threshold for calling a marker "valid".

    Returns
    -------
    results : pd.DataFrame
        Columns: kmer, p_value, eta_squared, valid, f_statistic
    """
    if kmer_indices is not None:
        X = X[:, kmer_indices]

    if kmer_ids is None:
        kmer_ids = [f"kmer_{i}" for i in range(X.shape[1])]

    variance = np.var(X, axis=0)
    top_idx = np.argsort(-variance)[:min(n_top, X.shape[1])]

    results = []
    for idx in top_idx:
        kmer_vals = X[:, idx]
        try:
            tertile = pd.qcut(kmer_vals, 3, labels=['low','mid','high'],
                              duplicates='drop')
            actual_labels = tertile.categories.tolist()
        except (ValueError, IndexError):
            results.append({'kmer': kmer_ids[idx], 'p_value': 1.0,
                           'eta_squared': 0.0, 'f_statistic': 0.0,
                           'valid': False})
            continue

        groups = [y[tertile == g] for g in actual_labels]
        if len(groups) >= 2 and all(len(g) >= 3 for g in groups):
            f_stat, p_val = stats.f_oneway(*groups)
            grand_mean = y.mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
            ss_total = sum((y - grand_mean)**2)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0
        else:
            f_stat, p_val, eta_sq = 0.0, 1.0, 0.0

        results.append({
            'kmer': kmer_ids[idx],
            'p_value': p_val,
            'eta_squared': eta_sq,
            'f_statistic': f_stat,
            'valid': p_val < alpha
        })

    return pd.DataFrame(results)


def count_valid_markers(df: pd.DataFrame) -> dict:
    """Count valid and invalid markers from ANOVA results.

    Returns
    -------
    dict with keys: total, valid, invalid, valid_pct
    """
    total = len(df)
    valid = df['valid'].sum()
    return {
        'total': total,
        'valid': valid,
        'invalid': total - valid,
        'valid_pct': valid / total * 100 if total > 0 else 0
    }
