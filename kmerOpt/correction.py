"""Multiple testing correction for k-mer GWAS.

K-mer GWAS has unique multiple-testing challenges:
1. Adjacent k-mers share (k-1)/k nucleotide overlap → tests are not independent
2. Conventional Bonferroni (α / M) is overly conservative

References:
    Chen & Liu et al. (2026) KMERIA. Nature Genetics 58, 1711-1721.
    Benjamini & Hochberg (1995) JRSS-B 57:289-300.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def modified_bonferroni_threshold(alpha: float = 0.05,
                                  n_tests: int = 50000,
                                  kmer_length: int = 31) -> float:
    """Modified Bonferroni threshold accounting for k-mer overlap.

    Standard Bonferroni:     P < α / M
    Modified Bonferroni:     P < α × (k / M)

    where k = k-mer length and M = total k-mers tested.
    This adjusts for the fact that adjacent k-mers share k-1 out of k
    nucleotides, so the effective number of independent tests is ~M/k.

    Parameters
    ----------
    alpha : float
        Family-wise error rate (default 0.05).
    n_tests : int
        Total number of k-mers tested (M).
    kmer_length : int
        K-mer length (default 31).

    Returns
    -------
    threshold : float
        Modified Bonferroni significance threshold.
    """
    return alpha * kmer_length / n_tests


def standard_bonferroni_threshold(alpha: float = 0.05,
                                   n_tests: int = 50000) -> float:
    """Standard Bonferroni correction: P < α / M."""
    return alpha / n_tests


def benjamini_hochberg(p_values: np.ndarray,
                        alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : np.ndarray
        Array of P-values.
    alpha : float
        FDR threshold.

    Returns
    -------
    rejected : np.ndarray (bool)
        Whether each hypothesis is rejected.
    threshold : float
        The BH critical value used.
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    n = len(p_values)

    # Sort P-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH critical values: (i / n) * alpha
    ranks = np.arange(1, n + 1)
    bh_critical = (ranks / n) * alpha

    # Find largest i where p_i ≤ (i/n)α
    below = sorted_p <= bh_critical
    if below.any():
        max_idx = np.where(below)[0][-1]
        threshold = sorted_p[max_idx]
        rejected = p_values <= threshold
    else:
        threshold = 0.0
        rejected = np.zeros(n, dtype=bool)

    return rejected, threshold


def dual_correction(p_values: np.ndarray,
                    n_tests: int,
                    kmer_length: int = 31,
                    alpha: float = 0.05,
                    fdr_level: float = 0.05) -> pd.DataFrame:
    """Dual correction strategy (KMERIA approach):
    1. Benjamini-Hochberg FDR (Padj < 0.05)
    2. Modified Bonferroni (P < α × k/M)

    Both criteria must be met for significance.

    Parameters
    ----------
    p_values : np.ndarray
        Array of raw P-values.
    n_tests : int
        Total number of tests (M).
    kmer_length : int
        K-mer length.
    alpha : float
        FWER for modified Bonferroni.
    fdr_level : float
        FDR level for BH correction.

    Returns
    -------
    results : pd.DataFrame
        Columns: p_value, bonf_sig, bh_sig, dual_sig, padj_bh
    """
    p_values = np.asarray(p_values, dtype=np.float64)

    # Modified Bonferroni
    bonf_threshold = modified_bonferroni_threshold(alpha, n_tests, kmer_length)
    bonf_sig = p_values < bonf_threshold

    # Benjamini-Hochberg FDR
    bh_rejected, bh_threshold = benjamini_hochberg(p_values, fdr_level)

    # Compute adjusted p-values (BH)
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    padj = np.ones(n)
    padj[sorted_idx] = np.minimum.accumulate(
        p_values[sorted_idx] * n / np.arange(1, n + 1)
    )
    # Ensure monotonicity
    padj[sorted_idx] = np.maximum.accumulate(padj[sorted_idx][::-1])[::-1]
    padj = np.minimum(padj, 1.0)

    # Dual significance
    dual_sig = bonf_sig & bh_rejected

    return pd.DataFrame({
        'p_value': p_values,
        'bonf_threshold': bonf_threshold,
        'bonf_sig': bonf_sig,
        'bh_sig': bh_rejected,
        'dual_sig': dual_sig,
        'padj_bh': padj,
    })


def count_significant(results: pd.DataFrame) -> dict:
    """Count significant k-mers under different correction strategies.

    Parameters
    ----------
    results : pd.DataFrame
        Output from dual_correction().

    Returns
    -------
    counts : dict
        {method: n_significant}
    """
    return {
        'modified_bonferroni': int(results['bonf_sig'].sum()),
        'bh_fdr': int(results['bh_sig'].sum()),
        'dual': int(results['dual_sig'].sum()),
        'total': len(results),
    }


def compare_thresholds(n_tests: int = 50000,
                        kmer_length: int = 31,
                        alpha: float = 0.05) -> dict:
    """Compare different multiple-testing thresholds.

    Parameters
    ----------
    n_tests : int
        Number of tests.
    kmer_length : int
        K-mer length.
    alpha : float
        Significance level.

    Returns
    -------
    thresholds : dict
        {method: threshold_value}
    """
    return {
        'nominal': alpha,
        'standard_bonferroni': standard_bonferroni_threshold(alpha, n_tests),
        'modified_bonferroni': modified_bonferroni_threshold(alpha, n_tests, kmer_length),
        'effective_tests': n_tests / kmer_length,
        'bonf_vs_modified_ratio': kmer_length,  # modified is k times more lenient
    }
