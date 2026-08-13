#!/usr/bin/env python3
"""kmerOpt command-line interface."""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from .core import KmerSelector, select_kmer
from .ld_prune import ld_prune
from .heritability import haseman_elston, variance_components
from .validate import anova_tertile, count_valid_markers
from .mapping import map_to_genome
from .correction import (dual_correction, count_significant,
                          compare_thresholds, modified_bonferroni_threshold)
from .normalize import mixed_ploidy_normalize, estimate_haploid_depth


def load_koc(koc_dir: str, pheno_path: str, prefix: str = "rice_koc",
             maf_min: float = 0.05, max_kmers: int = 50000,
             sample_pattern: str = "_k31") -> tuple:
    """Load KMERIA KOC matrix and align with phenotype.

    Returns
    -------
    X, y, kmer_ids, sample_ids
    """
    import glob
    import csv

    # Load phenotype
    ph = pd.read_csv(pheno_path, sep='\t')
    pheno_map = dict(zip(ph['IID'], ph['HD']))

    # Get KOC files
    koc_files = sorted(glob.glob(f"{koc_dir}/{prefix}.*.txt"))
    if not koc_files:
        raise FileNotFoundError(f"No KOC files found: {koc_dir}/{prefix}.*.txt")

    print(f"Found {len(koc_files)} KOC chunk(s)", file=sys.stderr)

    # Read header from first chunk
    with open(koc_files[0]) as f:
        header = f.readline().strip().split('\t')
    sids = [h.replace(sample_pattern, '') for h in header[1:]]
    cidx = [i for i, s in enumerate(sids) if s in pheno_map]

    if not cidx:
        raise ValueError("No samples overlap between KOC and phenotype!")

    y = np.array([pheno_map[sids[i]] for i in cidx], dtype=np.float32)
    print(f"  {len(cidx)} samples matched", file=sys.stderr)

    # Stream KOC → select top by variance using heap
    import heapq
    heap = []
    total = 0

    for kf in koc_files:
        with open(kf) as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            for row in reader:
                total += 1
                vals = np.array([float(row[i+1]) for i in cidx], dtype=np.float32)
                af = (vals > 0).mean()
                if min(af, 1 - af) < maf_min:
                    continue
                var = np.var(vals)
                if var <= 0:
                    continue
                kmer = row[0]
                if len(heap) < max_kmers:
                    heapq.heappush(heap, (var, kmer))
                elif var > heap[0][0]:
                    heapq.heapreplace(heap, (var, kmer))

    selected = sorted(heap, key=lambda x: -x[0])
    kmer_list = [s[1] for s in selected]
    kmer_set = set(kmer_list)
    kmer_idx = {k: i for i, k in enumerate(kmer_list)}

    print(f"  Scanned {total} k-mers, selected {len(kmer_list)}", file=sys.stderr)

    # Build dosage matrix
    X = np.zeros((len(cidx), len(kmer_list)), dtype=np.float32)
    for kf in koc_files:
        with open(kf) as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                kmer = row[0]
                if kmer in kmer_set:
                    ki = kmer_idx[kmer]
                    for j in range(len(cidx)):
                        X[j, ki] = float(row[cidx[j]+1])

    return X, y, kmer_list, [sids[i] for i in cidx]


def main():
    parser = argparse.ArgumentParser(
        description="kmerOpt: K-mer selection with quantitative genetics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kmeropt select --koc koc/ --pheno pheno.txt --out results/
  kmeropt prune --X dosage.csv --y pheno.csv --r2 0.7
  kmeropt heritability --X pruned.csv --y pheno.csv
  kmeropt validate --X dosage.csv --y pheno.csv --n-top 100
  kmeropt correct --pvals gwas_results.csv --n-tests 50000
  kmeropt normalize --X raw_koc.csv --depth depth.csv --ploidy 4
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # select: full pipeline
    p = subparsers.add_parser('select', help='Full k-mer selection pipeline')
    p.add_argument('--koc', required=True, help='KOC directory')
    p.add_argument('--pheno', required=True, help='Phenotype file (tab-sep, IID+Trait)')
    p.add_argument('--out', default='.', help='Output directory')
    p.add_argument('--r2', type=float, default=0.7, help='LD pruning threshold (default: 0.7)')
    p.add_argument('--maf', type=float, default=0.05, help='MAF filter (default: 0.05)')
    p.add_argument('--max-kmers', type=int, default=50000, help='Max kmers to load (default: 50000)')
    p.add_argument('--bootstrap', type=int, default=100, help='Bootstrap replicates (default: 100)')
    p.add_argument('--ploidy', type=float, default=None, help='Ploidy level for normalization')
    p.add_argument('--encode', default='standard', choices=['standard','quantile','binary'],
                   help='Encoding: standard, quantile (0-2), or binary (0/1)')
    p.add_argument('--depth', type=str, default=None,
                   help='Per-sample sequencing depth file (sample<TAB>depth)')

    # prune: LD pruning only
    p = subparsers.add_parser('prune', help='LD pruning only')
    p.add_argument('--X', required=True, help='Dosage matrix CSV')
    p.add_argument('--y', required=True, help='Phenotype CSV')
    p.add_argument('--r2', type=float, default=0.7, help='LD threshold')
    p.add_argument('--out', default='.', help='Output directory')

    # heritability
    p = subparsers.add_parser('heritability', help='Heritability estimation only')
    p.add_argument('--X', required=True, help='LD-pruned dosage matrix CSV')
    p.add_argument('--y', required=True, help='Phenotype CSV')
    p.add_argument('--out', default='.', help='Output directory')
    p.add_argument('--bootstrap', type=int, default=100)

    # validate
    p = subparsers.add_parser('validate', help='ANOVA validation only')
    p.add_argument('--X', required=True, help='Raw dosage matrix CSV')
    p.add_argument('--y', required=True, help='Phenotype CSV')
    p.add_argument('--n-top', type=int, default=100, help='Top N markers to validate')
    p.add_argument('--out', default='.', help='Output directory')

    # correction: multiple testing correction
    p = subparsers.add_parser('correct', help='Multiple testing correction')
    p.add_argument('--pvals', required=True, help='CSV with P-values (column: p_value)')
    p.add_argument('--n-tests', type=int, default=50000, help='Number of tests')
    p.add_argument('--kmer-length', type=int, default=31, help='K-mer length')
    p.add_argument('--alpha', type=float, default=0.05, help='FWER alpha')
    p.add_argument('--out', default='.', help='Output directory')

    # normalize: ploidy-aware normalization
    p = subparsers.add_parser('normalize', help='Ploidy-aware normalization')
    p.add_argument('--X', required=True, help='Raw KOC matrix CSV')
    p.add_argument('--depth', required=True, help='Per-sample depth CSV')
    p.add_argument('--ploidy', type=float, default=2, help='Ploidy level')
    p.add_argument('--encode', default='quantile', choices=['quantile','binary'],
                   help='Dosage encoding')
    p.add_argument('--out', default='.', help='Output directory')

    # thresholds: compare correction methods
    p = subparsers.add_parser('thresholds', help='Compare multiple testing thresholds')
    p.add_argument('--n-tests', type=int, default=50000)
    p.add_argument('--kmer-length', type=int, default=31)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'select':
        os.makedirs(args.out, exist_ok=True)

        print("Loading KOC...", file=sys.stderr)
        X, y, kmer_ids, sample_ids = load_koc(
            args.koc, args.pheno, maf_min=args.maf, max_kmers=args.max_kmers)

        # Setup normalization parameters
        seq_depth = None
        ploidy_arr = None
        encode = args.encode if args.ploidy else 'standard'

        if args.ploidy:
            ploidy_arr = np.full(len(y), args.ploidy, dtype=np.float64)
            if args.depth:
                depth_df = pd.read_csv(args.depth, sep='\t', header=None)
                seq_depth = depth_df.iloc[:, 1].values if depth_df.shape[1] >= 2 else None
            print(f"Ploidy-aware normalization: {args.ploidy}x, encode={encode}", file=sys.stderr)

        print(f"Running LD pruning (|r| < {args.r2})...", file=sys.stderr)
        ks = KmerSelector(X, y, kmer_ids,
                          sequencing_depth=seq_depth,
                          ploidy=ploidy_arr,
                          encode=encode)
        kept = ks.ld_prune(r2_threshold=args.r2)
        print(f"  {len(kept)} markers retained", file=sys.stderr)

        print("Estimating heritability...", file=sys.stderr)
        h2_curve = ks.estimate_heritability(
            [200, 500, 1000, 2000, 3000, min(5000, len(kept))],
            n_bootstrap=args.bootstrap)
        optimal = ks.find_optimal()

        print("ANOVA validation...", file=sys.stderr)
        anova_df = ks.anova_validate(n_top=100)

        # Save results
        pd.DataFrame({
            'n_markers': list(h2_curve.keys()),
            'h2': [v[0] for v in h2_curve.values()],
            'se': [v[1] for v in h2_curve.values()]
        }).to_csv(f"{args.out}/heritability_curve.csv", index=False)

        anova_df.to_csv(f"{args.out}/anova_validation.csv", index=False)

        # Save selected k-mers
        selected_kmers = [kmer_ids[i] for i in kept]
        pd.DataFrame({'kmer': selected_kmers}).to_csv(
            f"{args.out}/selected_kmers.txt", index=False, header=False)

        # Summary
        print(f"\n{'='*50}")
        print(f"kmerOpt Summary")
        print(f"{'='*50}")
        print(f"Input: {X.shape[0]} samples x {X.shape[1]} k-mers")
        print(f"LD-pruned: {len(kept)} markers")
        print(f"Optimal n: {optimal} (h² = {ks.optimal_h2_:.4f})")
        print(f"Valid markers (ANOVA): {anova_df['valid'].sum()}/{len(anova_df)}")
        print(f"Results saved to: {args.out}/")

    elif args.command == 'prune':
        X = pd.read_csv(args.X, index_col=0).values
        y = pd.read_csv(args.y, index_col=0).values.ravel()
        kept = ld_prune(X, r2_threshold=args.r2)
        pd.DataFrame(X[:, kept]).to_csv(f"{args.out}/pruned_matrix.csv", index=False)
        print(f"Pruned: {len(kept)} markers retained")

    elif args.command == 'heritability':
        X = pd.read_csv(args.X, index_col=0).values
        y = pd.read_csv(args.y, index_col=0).values.ravel()
        h2_curve = haseman_elston(X, y, n_bootstrap=args.bootstrap)
        for n, (h2, se) in sorted(h2_curve.items()):
            print(f"n={n:>5}: h² = {h2:.4f} ± {se:.4f}")

    elif args.command == 'validate':
        X = pd.read_csv(args.X, index_col=0).values
        y = pd.read_csv(args.y, index_col=0).values.ravel()
        df = anova_tertile(X, y, n_top=args.n_top)
        counts = count_valid_markers(df)
        print(f"Valid: {counts['valid']}/{counts['total']} ({counts['valid_pct']:.1f}%)")
        df.to_csv(f"{args.out}/anova_results.csv", index=False)

    elif args.command == 'correct':
        pval_df = pd.read_csv(args.pvals)
        p_col = 'p_value' if 'p_value' in pval_df.columns else pval_df.columns[0]
        p_values = pval_df[p_col].values

        results = dual_correction(p_values, n_tests=args.n_tests,
                                   kmer_length=args.kmer_length,
                                   alpha=args.alpha)
        counts = count_significant(results)

        print(f"Multiple Testing Correction Results")
        print(f"{'='*45}")
        print(f"  Total tests:         {counts['total']}")
        print(f"  Modified Bonferroni: {counts['modified_bonferroni']} sig "
              f"(P < {modified_bonferroni_threshold(args.alpha, args.n_tests, args.kmer_length):.2e})")
        print(f"  BH FDR (5%):         {counts['bh_fdr']} sig")
        print(f"  Dual significant:    {counts['dual']}")
        print(f"  Standard Bonferroni: P < {args.alpha/args.n_tests:.2e}")

        os.makedirs(args.out, exist_ok=True)
        results.to_csv(f"{args.out}/correction_results.csv", index=False)

    elif args.command == 'normalize':
        X = pd.read_csv(args.X, index_col=0).values
        depth_df = pd.read_csv(args.depth, index_col=0)
        depth = depth_df.values.ravel()
        ploidy = np.full(X.shape[0], args.ploidy, dtype=np.float64)

        X_enc = mixed_ploidy_normalize(X, depth, ploidy, encode=args.encode)
        print(f"Normalized: {X.shape[0]} samples x {X.shape[1]} k-mers")
        print(f"  Ploidy: {args.ploidy}x, Encoding: {args.encode}")
        print(f"  Range: [{X_enc.min():.3f}, {X_enc.max():.3f}]")

        os.makedirs(args.out, exist_ok=True)
        pd.DataFrame(X_enc).to_csv(f"{args.out}/normalized_matrix.csv", index=False)

    elif args.command == 'thresholds':
        thresh = compare_thresholds(args.n_tests, args.kmer_length)
        print("Multiple Testing Thresholds Comparison")
        print(f"{'='*45}")
        for k, v in thresh.items():
            if isinstance(v, float):
                print(f"  {k:<25}: {v:.4e}")
            else:
                print(f"  {k:<25}: {v:.1f}")


if __name__ == '__main__':
    main()
