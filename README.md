# kmerOpt

[![PyPI version](https://img.shields.io/pypi/v/kmerOpt.svg)](https://pypi.org/project/kmerOpt/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**K-mer selection, annotation & interpretation for GWAS.**

KMERIA gives you a KOC matrix. kmerOpt tells you **which k-mers matter,
how many you need, and what they represent biologically.**

## The Problem

K-mer GWAS is powerful but the downstream is a black box:

- **No guidance** on how many k-mers to use
- **No LD pruning** — k-mers are highly correlated around SVs
- **No ploidy-aware normalization** — raw counts biased by sequencing depth
- **Overly conservative multiple testing** — k-mers share k-1/k nucleotide overlap
- **No variant classification** — every paper's Discussion starts with "likely represents..."
- **No automated interpretation** — researchers hand-curate gene lists

kmerOpt solves all of this in one package.

## What kmerOpt Does

```
KMERIA KOC matrix + phenotype
       |
       |-- Normalization (v0.3.0) --------
       |   |-- Ploidy-aware depth correction
       |   |-- Quantile dosage encoding (0-2)
       |
       |-- Selection ---------------------
       |   |-- LD Pruning (greedy, |r| threshold)
       |   |-- h^2 Curve (Haseman-Elston + bootstrap SE)
       |   |-- Plateau Detection -> Optimal n
       |   |-- ANOVA Tertile Validation
       |
       |-- Correction (v0.3.0) ----------
       |   |-- Modified Bonferroni (alpha x k/M)
       |   |-- Benjamini-Hochberg FDR
       |   |-- Dual correction (both)
       |
       |-- Annotation -------------------
       |   |-- Genome Mapping + Variant Classification
       |   |-- Gene Annotation (GFF)
       |   |-- GO/KEGG Enrichment
       |   |-- Selection Signal Scan (Fst, PAV freq)
       |
       v
  Optimal k-mer set + Biological interpretation + Figures
```

## Installation

```bash
pip install kmerOpt
```

Requires Python ≥ 3.9. Dependencies (numpy, pandas, scipy, scikit-learn, matplotlib)
are installed automatically.

### From source

```bash
git clone https://github.com/undefaned/kmerOpt.git
cd kmerOpt
pip install -e .
```

## Input Data

kmerOpt takes two inputs, the same ones produced by any k-mer GWAS workflow
(e.g. [KMERIA](https://www.nature.com/articles/s41588-026-02641-8), kmersGWAS):

- **KOC matrix** — a sample × k-mer abundance/count matrix (CSV or `.npy`),
  with sample IDs as row names and k-mer sequences as column names.
- **Phenotype** — a tab-separated file with two columns: `sample` and `trait`
  (one trait per run).

Optional but recommended for normalization: per-sample **sequencing depth**
(a numeric vector aligned with the sample order).

## Quick Start

### Selection with ploidy-aware normalization

```bash
# For tetraploid species (alfalfa, potato):
kmeropt select --koc koc_dir/ --pheno pheno.txt --ploidy 4 --encode quantile --out results/

# For diploid pure lines (rice, sorghum):
kmeropt select --koc koc_dir/ --pheno pheno.txt --out results/
```

```python
from kmerOpt import KmerSelector
import numpy as np

# With ploidy-aware normalization
X = np.load("koc_matrix.npy")
y = np.load("phenotype.npy")
depth = np.load("sequencing_depth.npy")

ks = KmerSelector(X, y, sequencing_depth=depth, ploidy=4.0, encode='quantile')
ks.ld_prune(r2_threshold=0.7)
ks.estimate_heritability([200, 500, 1000, 2000])
optimal_n = ks.find_optimal()
print(ks.summary())
```

### Multiple testing correction

```python
from kmerOpt import dual_correction, compare_thresholds

# Modified Bonferroni is 31x more powerful than standard Bonferroni
th = compare_thresholds(n_tests=50000, kmer_length=31)
# {'standard_bonferroni': 1.0e-06, 'modified_bonferroni': 3.1e-05}

results = dual_correction(p_values, n_tests=50000, kmer_length=31)
```

### Annotation

```python
from kmerOpt import annotate_kmers
ka = annotate_kmers(kmers, genome_fasta='rice.fa', gff_path='rice.gff')
print(ka.variant_summary_)  # {'SNP': 234, 'PAV': 89, 'CNV': 53}
```

## CLI Commands

```bash
kmeropt select       # Full pipeline: KOC -> LD -> h^2 -> ANOVA
kmeropt prune        # LD pruning only
kmeropt heritability # HE regression only
kmeropt validate     # ANOVA tertile validation only
kmeropt correct      # Multiple testing correction
kmeropt normalize    # Ploidy-aware normalization
kmeropt thresholds   # Compare correction thresholds
```

## Output

```
results/
|-- heritability_curve.csv       # n_markers, h2, SE
|-- selected_kmers.txt           # LD-pruned k-mer list
|-- anova_validation.csv         # Tertile ANOVA results
|-- correction_results.csv       # Modified Bonf + FDR
|-- normalized_matrix.csv        # Ploidy-corrected dosage
|-- annotation.csv               # Per-kmer: chr, pos, variant_type, genes
|-- candidate_genes.csv          # Gene-level annotation
|-- enrichment.csv               # GO/KEGG enrichment
```

## New in v0.3.0

- **Ploidy-aware normalization**: Corrects k-mer counts by haploid depth (f / (D/p))
- **Quantile dosage encoding**: Maps raw counts to 0-2 continuous scale
- **Modified Bonferroni**: P < alpha x k/M — 31x more powerful than standard Bonferroni
- **Dual correction**: Benjamini-Hochberg FDR + Modified Bonferroni (KMERIA-style)
- **Mixed-ploidy support**: Per-sample ploidy vectors

## Citation

[Manuscript in preparation]

## References

- Chen S, Liu X et al. (2026) KMERIA. *Nat Genet* 58:1711-1721
- Haseman JK, Elston RC (1972) *Behav Genet* 2:3-19
- Yang J et al. (2010) *Nat Genet* 42:565-569 (GCTA/GREML)
- Voichek & Weigel (2020) *Nat Genet* 52:534-540 (k-mer GWAS)

## License

MIT
