"""Ploidy-aware normalization and quantile dosage encoding for k-mer data.

References:
    Chen & Liu et al. (2026) KMERIA. Nature Genetics 58, 1711-1721.
"""

import numpy as np
from typing import Optional, Tuple


def ploidy_aware_normalize(X: np.ndarray,
                           sequencing_depth: np.ndarray,
                           ploidy: np.ndarray,
                           min_depth: float = 1.0) -> np.ndarray:
    """Ploidy-aware depth correction for k-mer occurrence counts.

    Normalizes raw k-mer counts to a standardized haploid depth scale,
    correcting for differences in sequencing depth and ploidy across samples.

    f_ij' = round(f_ij / d_i^haploid)
    where d_i^haploid = D_i / p_i

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_kmers)
        Raw k-mer occurrence count (KOC) matrix.
    sequencing_depth : np.ndarray, shape (n_samples,)
        Per-sample sequencing depth (average coverage).
    ploidy : np.ndarray, shape (n_samples,)
        Per-sample ploidy level (2 for diploid, 4 for tetraploid, etc.).
        Can be an integer array or float (for mixed-ploidy populations).
    min_depth : float
        Minimum haploid depth to avoid division by zero.

    Returns
    -------
    X_norm : np.ndarray, shape (n_samples, n_kmers)
        Depth- and ploidy-corrected k-mer matrix (rounded to integers).
    """
    X = np.asarray(X, dtype=np.float64)
    sequencing_depth = np.asarray(sequencing_depth, dtype=np.float64)
    ploidy = np.asarray(ploidy, dtype=np.float64)

    # Haploid depth per sample
    haploid_depth = sequencing_depth / ploidy
    haploid_depth = np.maximum(haploid_depth, min_depth)

    # Correct each sample by its haploid depth
    X_norm = X / haploid_depth[:, np.newaxis]
    X_norm = np.round(X_norm).astype(np.int32)

    return X_norm


def quantile_dosage_encode(X: np.ndarray,
                           lower_quantile: float = 0.05,
                           upper_quantile: float = 0.95) -> np.ndarray:
    """Encode k-mer abundance as continuous allele dosage (0-2 scale).

    Replaces raw k-mer counts with a 0-2 dosage scale using quantile-based
    thresholding, eliminating depth-dependent bias:

    - Values ≤ lower_quantile → 0  (absent / reference-like)
    - Values ≥ upper_quantile → 2  (full dosage / homozygous alt)
    - Intermediate values → linearly interpolated between 0 and 2

    This preserves continuous dosage information for CNV/PAV while
    standardizing across samples with different sequencing depths.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_kmers)
        Ploidy-normalized k-mer matrix.
    lower_quantile : float
        Lower quantile threshold (default 0.05 = 5th percentile).
    upper_quantile : float
        Upper quantile threshold (default 0.95 = 95th percentile).

    Returns
    -------
    X_dosage : np.ndarray, shape (n_samples, n_kmers)
        Continuous dosage matrix (values in [0, 2]).
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_kmers = X.shape

    # Per-k-mer quantiles (column-wise, like StandardScaler)
    lower = np.percentile(X, lower_quantile * 100, axis=0, method='linear')
    upper = np.percentile(X, upper_quantile * 100, axis=0, method='linear')

    # Avoid division by zero (monomorphic k-mers)
    denom = upper - lower
    denom[denom == 0] = 1.0

    # Linear mapping: [lower, upper] → [0, 2]
    X_dosage = 2.0 * (X - lower[np.newaxis, :]) / denom[np.newaxis, :]

    # Clip to [0, 2]
    X_dosage = np.clip(X_dosage, 0.0, 2.0)

    return X_dosage.astype(np.float32)


def binary_encode(X: np.ndarray, threshold_quantile: float = 0.20) -> np.ndarray:
    """Binary (0/1) encoding for pure-line / presence-absence scenarios.

    Suitable for homozygous inbred lines (like our rice panel) where
    continuous dosage is unnecessary and binary PAV encoding matches
    the biological ground truth.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_kmers)
        Ploidy-normalized k-mer matrix.
    threshold_quantile : float
        Values below this quantile → 0, above → 1.

    Returns
    -------
    X_binary : np.ndarray, shape (n_samples, n_kmers)
        Binary encoded matrix ({0, 1}).
    """
    X = np.asarray(X, dtype=np.float64)
    thresholds = np.percentile(X, threshold_quantile * 100, axis=0, method='linear')
    X_binary = (X > thresholds[np.newaxis, :]).astype(np.int8)
    return X_binary


def mixed_ploidy_normalize(X: np.ndarray,
                           sequencing_depth: np.ndarray,
                           ploidy: np.ndarray,
                           encode: str = 'quantile',
                           lower_quantile: float = 0.05,
                           upper_quantile: float = 0.95) -> np.ndarray:
    """Full normalization pipeline for mixed-ploidy populations.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_kmers)
        Raw KOC matrix.
    sequencing_depth : np.ndarray, shape (n_samples,)
        Per-sample sequencing depth.
    ploidy : np.ndarray, shape (n_samples,)
        Per-sample ploidy level.
    encode : str
        'quantile' for continuous 0-2 dosage, 'binary' for 0/1 PAV.
    lower_quantile : float
        Lower quantile for dosage encoding.
    upper_quantile : float
        Upper quantile for dosage encoding.

    Returns
    -------
    X_encoded : np.ndarray
        Normalized and encoded k-mer matrix.
    """
    # Step 1: Ploidy-aware depth correction
    X_norm = ploidy_aware_normalize(X, sequencing_depth, ploidy)

    # Step 2: Encode
    if encode == 'binary':
        X_encoded = binary_encode(X_norm)
    elif encode == 'quantile':
        X_encoded = quantile_dosage_encode(X_norm, lower_quantile, upper_quantile)
    else:
        raise ValueError(f"Unknown encoding: {encode}. Use 'quantile' or 'binary'.")

    return X_encoded


def estimate_haploid_depth(total_reads: np.ndarray,
                           ploidy: np.ndarray,
                           genome_size_bp: int = 374_000_000) -> np.ndarray:
    """Estimate haploid sequencing depth from total read counts.

    Parameters
    ----------
    total_reads : np.ndarray, shape (n_samples,)
        Total number of reads per sample.
    ploidy : np.ndarray, shape (n_samples,)
        Per-sample ploidy level.
    genome_size_bp : int
        Reference genome size in base pairs.

    Returns
    -------
    haploid_depth : np.ndarray
        Estimated haploid depth per sample.
    """
    # Rough estimate assuming 150bp reads
    avg_read_length = 150
    total_bases = total_reads * avg_read_length
    coverage = total_bases / genome_size_bp
    haploid_depth = coverage / ploidy
    return haploid_depth
