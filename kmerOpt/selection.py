"""Selection signal detection from k-mer PAV patterns."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, List, Tuple


def compute_pav_frequencies(pav_matrix: np.ndarray,
                             sample_labels: List[str]) -> pd.DataFrame:
    """Compute PAV frequency by population group.

    Parameters
    ----------
    pav_matrix : np.ndarray, shape (n_samples, n_kmers)
        Binary presence/absence matrix.
    sample_labels : list of str
        Population labels for each sample.

    Returns
    -------
    pd.DataFrame with per-kmer PAV frequencies by population.
    """
    labels = np.array(sample_labels)
    unique_pops = np.unique(labels)

    results = []
    for pop in unique_pops:
        idx = labels == pop
        if idx.sum() == 0:
            continue
        pop_pav = pav_matrix[idx].mean(axis=0)
        for i in range(pav_matrix.shape[1]):
            results.append({'population': pop, 'kmer_idx': i, 'pav_freq': pop_pav[i]})

    return pd.DataFrame(results)


def tajima_d(allele_counts: np.ndarray, n_samples: int) -> float:
    """Compute Tajima's D from allele counts.

    Tajima's D = (pi - theta_W) / sqrt(Var(pi - theta_W))

    Parameters
    ----------
    allele_counts : np.ndarray
        Derived allele counts per SNP.
    n_samples : int
        Number of individuals.

    Returns
    -------
    float : Tajima's D statistic.
    """
    n = n_samples
    p = allele_counts / (2 * n)  # allele frequencies

    # Pi (nucleotide diversity)
    pi = np.mean(2 * p * (1 - p) * n / (n - 1))

    # Watterson's theta
    S = np.sum((allele_counts > 0) & (allele_counts < 2 * n))  # segregating sites
    if S == 0:
        return 0.0

    a1 = np.sum(1.0 / np.arange(1, n))
    theta_w = S / a1

    # Variance (approximate, Tajima 1989)
    a2 = np.sum(1.0 / (np.arange(1, n) ** 2))
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n ** 2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - 1.0 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / (a1 ** 2)
    e1 = c1 / a1
    e2 = c2 / (a1 ** 2 + a2)
    var_d = e1 * S + e2 * S * (S - 1)

    if var_d <= 0:
        return 0.0

    return (pi - theta_w) / np.sqrt(var_d)


def fst_weir_cockerham(pav_jpn: np.ndarray, pav_ind: np.ndarray) -> float:
    """Weir & Cockerham's Fst for two populations from PAV data.

    Parameters
    ----------
    pav_jpn : np.ndarray, shape (n_jpn, n_kmers)
    pav_ind : np.ndarray, shape (n_ind, n_kmers)

    Returns
    -------
    float : Mean Fst across all k-mers.
    """
    fst_values = []
    for j in range(pav_jpn.shape[1]):
        p1 = pav_jpn[:, j].mean()
        p2 = pav_ind[:, j].mean()
        p_bar = (p1 + p2) / 2

        msg = 2 * p_bar * (1 - p_bar)
        if msg > 0:
            msg_between = (p1 - p2) ** 2 / 2
            fst_values.append(msg_between / msg if msg > 0 else 0)

    return np.mean(fst_values) if fst_values else 0.0


def selection_scan(pav_matrix: np.ndarray,
                   sample_labels: List[str],
                   window_size: int = 50) -> pd.DataFrame:
    """Scan for selection signals across k-mer windows.

    Parameters
    ----------
    pav_matrix : np.ndarray
    sample_labels : list of str
    window_size : int
        Number of k-mers per sliding window.

    Returns
    -------
    pd.DataFrame with window-level selection statistics.
    """
    labels = np.array(sample_labels)
    pops = np.unique(labels)

    if len(pops) < 2:
        return pd.DataFrame()  # need at least 2 populations

    idx_p1 = labels == pops[0]
    idx_p2 = labels == pops[1]

    results = []
    n_kmers = pav_matrix.shape[1]

    for start in range(0, n_kmers, window_size // 2):
        end = min(start + window_size, n_kmers)
        window_pav = pav_matrix[:, start:end]

        # Per-population PAV
        pav1 = window_pav[idx_p1].mean()
        pav2 = window_pav[idx_p2].mean()

        # Fst
        fst = fst_weir_cockerham(
            window_pav[idx_p1], window_pav[idx_p2]
        ) if window_pav.shape[1] > 0 else 0

        # PAV frequency difference
        pav_diff = abs(pav1 - pav2) if window_pav.shape[1] > 0 else 0

        results.append({
            'window_start': start,
            'window_end': end,
            f'pav_{pops[0]}': pav1,
            f'pav_{pops[1]}': pav2,
            'pav_diff': pav_diff,
            'fst': fst
        })

    return pd.DataFrame(results)
