"""Core module: KmerSelector — the main class for k-mer selection."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings("ignore")


class KmerSelector:
    """Select optimal k-mer subset with quantitative genetics criteria.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_kmers)
        K-mer dosage matrix (continuous or binary).
    y : np.ndarray, shape (n_samples,)
        Phenotype values.
    kmer_ids : list of str, optional
        K-mer sequences or identifiers.

    Attributes
    ----------
    n_samples_ : int
    n_kmers_ : int
    selected_indices_ : np.ndarray
        Indices of selected k-mers after LD pruning.
    h2_curve_ : dict
        {n_kmers: (h2, se)} for each tested marker count.
    optimal_n_ : int
        Optimal number of LD-pruned k-mers.
    optimal_h2_ : float
        h² at optimal k-mer count.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, kmer_ids: Optional[List[str]] = None):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.kmer_ids = kmer_ids or [f"kmer_{i}" for i in range(X.shape[1])]

        # Validate
        assert self.X.shape[0] == len(self.y), "X and y must have same n_samples"
        assert self.X.ndim == 2, "X must be 2D"

        self.n_samples_ = self.X.shape[0]
        self.n_kmers_ = self.X.shape[1]

        # Standardize
        self._scaler_X = StandardScaler()
        self.Xs_ = self._scaler_X.fit_transform(self.X)
        self.ys_ = StandardScaler().fit_transform(self.y.reshape(-1,1)).ravel()

        # State
        self.selected_indices_ = None
        self.h2_curve_ = {}
        self.optimal_n_ = None
        self.optimal_h2_ = None

    def ld_prune(self, r2_threshold: float = 0.7, max_kept: int = 5000,
                 n_jobs: int = 1) -> np.ndarray:
        """Greedy LD pruning: keep high-variance markers, drop those
        correlated with already-kept markers above |r| > r2_threshold.

        Parameters
        ----------
        r2_threshold : float
            Pearson |r| threshold. Default 0.7.
        max_kept : int
            Maximum number of markers to retain.
        n_jobs : int
            Number of parallel jobs (1 = sequential).

        Returns
        -------
        kept : np.ndarray
            Indices of LD-pruned markers.
        """
        variance = np.var(self.Xs_, axis=0)
        var_order = np.argsort(-variance)  # descending

        kept = []
        for i in var_order:
            if not kept:
                kept.append(i)
                continue
            # Check correlation with recently kept markers
            recent = kept[-min(50, len(kept)):]
            corrs = np.abs([np.corrcoef(self.Xs_[:, j], self.Xs_[:, i])[0,1]
                           for j in recent])
            if corrs.max() < r2_threshold:
                kept.append(i)
            if len(kept) >= max_kept:
                break

        self.selected_indices_ = np.array(kept)
        return self.selected_indices_

    def estimate_heritability(self, n_markers_list: List[int],
                              n_bootstrap: int = 100, n_pcs: int = 0,
                              random_state: int = 42) -> Dict[int, Tuple[float, float]]:
        """Estimate h² using Haseman-Elston regression at multiple marker counts.

        Parameters
        ----------
        n_markers_list : list of int
            Marker counts to test (e.g. [200, 500, 1000, 2000]).
        n_bootstrap : int
            Number of bootstrap resamples for SE estimation.
        n_pcs : int
            Number of PCs to regress out before h² estimation.
        random_state : int
            Random seed for bootstrap.

        Returns
        -------
        h2_curve : dict
            {n_markers: (h2_mean, h2_se)}
        """
        if self.selected_indices_ is None:
            raise ValueError("Run ld_prune() first!")

        Xp = self.Xs_[:, self.selected_indices_]
        n_available = Xp.shape[1]

        # Regress out PCs if requested
        if n_pcs > 0:
            pcs = PCA(min(n_pcs, self.n_samples_)).fit_transform(self.Xs_)
            # Residualize y
            from sklearn.linear_model import LinearRegression
            y_resid = self.ys_ - LinearRegression().fit(pcs, self.ys_).predict(pcs)
        else:
            y_resid = self.ys_

        # Pre-compute for efficiency
        t_idx = np.triu_indices(self.n_samples_, k=1)
        yy = np.outer(y_resid, y_resid)[t_idx]
        vp = np.var(y_resid)

        rng = np.random.RandomState(random_state)
        h2_curve = {}

        for n_markers in n_markers_list:
            if n_markers > n_available:
                break

            h2_vals = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n_available, n_markers, replace=False)
                G = Xp[:, idx] @ Xp[:, idx].T / n_markers
                gg = G[t_idx]
                sigma2_g = max(0, np.cov(yy, gg)[0,1] / np.var(gg))
                h2 = min(sigma2_g / vp, 0.99)
                h2_vals.append(h2)

            h2_curve[n_markers] = (np.mean(h2_vals), np.std(h2_vals))

        self.h2_curve_ = h2_curve
        return h2_curve

    def find_optimal(self, plateau_threshold: float = 0.005,
                     min_increment: float = 0.001) -> int:
        """Detect h² plateau: the smallest n where adding more markers
        increases h² by less than plateau_threshold.

        Parameters
        ----------
        plateau_threshold : float
            If h² increase < this value, consider plateau reached.
        min_increment : float
            Minimum h² increase for a meaningful gain.

        Returns
        -------
        optimal_n : int
            Optimal number of LD-pruned markers.
        """
        if not self.h2_curve_:
            raise ValueError("Run estimate_heritability() first!")

        sorted_n = sorted(self.h2_curve_.keys())
        h2_vals = np.array([self.h2_curve_[n][0] for n in sorted_n])

        # Find where increments drop below threshold
        increments = np.diff(h2_vals)
        plateau_idx = np.where(increments < plateau_threshold)[0]

        if len(plateau_idx) > 0:
            optimal_n = sorted_n[plateau_idx[0]]
        else:
            # No clear plateau — take the largest n
            optimal_n = sorted_n[-1]

        self.optimal_n_ = optimal_n
        self.optimal_h2_ = self.h2_curve_[optimal_n][0]
        return optimal_n

    def anova_validate(self, n_top: int = 100, alpha: float = 0.05) -> pd.DataFrame:
        """Tertile ANOVA validation for top k-mer markers.

        Parameters
        ----------
        n_top : int
            Number of top markers (by variance) to validate.
        alpha : float
            Significance threshold for ANOVA.

        Returns
        -------
        results : pd.DataFrame
            Columns: kmer, p_value, effect_size, valid (bool)
        """
        if self.selected_indices_ is None:
            raise ValueError("Run ld_prune() first!")

        Xp = self.Xs_[:, self.selected_indices_]
        var_order = np.argsort(-np.var(Xp, axis=0))
        top_idx = var_order[:min(n_top, len(var_order))]

        results = []
        for idx in top_idx:
            kmer_vals = self.X[:, self.selected_indices_[idx]]
            # Tertile grouping
            tertile = pd.qcut(kmer_vals, 3, labels=['low','mid','high'], duplicates='drop')
            groups = [self.y[tertile == g] for g in ['low','mid','high']]

            if all(len(g) >= 3 for g in groups):
                f_stat, p_val = stats.f_oneway(*groups)
                # Effect size (eta-squared)
                grand_mean = self.y.mean()
                ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
                ss_total = sum((self.y - grand_mean)**2)
                eta_sq = ss_between / ss_total if ss_total > 0 else 0
            else:
                p_val, eta_sq = 1.0, 0.0

            results.append({
                'kmer': self.kmer_ids[self.selected_indices_[idx]],
                'p_value': p_val,
                'eta_squared': eta_sq,
                'valid': p_val < alpha
            })

        return pd.DataFrame(results)

    def summary(self) -> str:
        """Generate a summary report."""
        lines = ["=" * 60, "kmerOpt Selection Summary", "=" * 60]
        lines.append(f"Samples: {self.n_samples_}, K-mers: {self.n_kmers_}")

        if self.selected_indices_ is not None:
            lines.append(f"LD-pruned markers: {len(self.selected_indices_)}")

        if self.h2_curve_:
            lines.append("\nHeritability curve:")
            for n, (h2, se) in sorted(self.h2_curve_.items()):
                marker = " ← OPTIMAL" if n == self.optimal_n_ else ""
                lines.append(f"  n={n:>5}: h² = {h2:.4f} ± {se:.4f}{marker}")

        if self.optimal_h2_ is not None:
            lines.append(f"\nOptimal h²: {self.optimal_h2_:.4f} (n={self.optimal_n_})")

        return "\n".join(lines)


def select_kmer(X: np.ndarray, y: np.ndarray,
                r2_threshold: float = 0.7,
                n_markers_list: Optional[List[int]] = None,
                n_bootstrap: int = 100,
                n_top_anova: int = 100) -> KmerSelector:
    """Convenience function: full k-mer selection pipeline.

    Parameters
    ----------
    X : np.ndarray
        K-mer dosage matrix.
    y : np.ndarray
        Phenotype values.
    r2_threshold : float
        LD pruning threshold (Pearson |r|).
    n_markers_list : list, optional
        Marker counts for h² curve. Default: [200,500,1000,2000,3000,5000].
    n_bootstrap : int
        Bootstrap resamples for h² SE.
    n_top_anova : int
        Top markers for ANOVA validation.

    Returns
    -------
    selector : KmerSelector
        Fitted selector with all results.
    """
    if n_markers_list is None:
        n_markers_list = [200, 500, 1000, 2000, 3000, 5000]

    ks = KmerSelector(X, y)
    ks.ld_prune(r2_threshold=r2_threshold)
    ks.estimate_heritability(n_markers_list, n_bootstrap=n_bootstrap)
    ks.find_optimal()

    return ks
