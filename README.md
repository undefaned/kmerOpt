# kmerOpt

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**K-mer selection, annotation & interpretation for GWAS.**

KMERIA gives you a KOC matrix. kmerOpt tells you **which k-mers matter,
how many you need, and what they represent biologically.**

## The Problem

K-mer GWAS is powerful but the downstream is a black box:

- **No guidance** on how many k-mers to use
- **No LD pruning** — k-mers are highly correlated around SVs
- **No variant classification** — every paper's Discussion starts with "likely represents..."
- **No automated interpretation** — researchers hand-curate gene lists

kmerOpt solves all of this in one package.

## What kmerOpt Does

```
KMERIA KOC matrix + phenotype
       │
       ├── Selection ───────────────────────
       │   ├── LD Pruning (greedy, |r| threshold)
       │   ├── h2 Curve (Haseman-Elston + bootstrap SE)
       │   ├── Plateau Detection → Optimal n
       │   └── ANOVA Tertile Validation
       │
       ├── Annotation ──────────────────────
       │   ├── Genome Mapping + Variant Classification
       │   │   (SNP / SV / PAV / CNV)
       │   ├── Gene Annotation (GFF)
       │   ├── GO/KEGG Enrichment
       │   └── Selection Signal Scan (Fst, PAV freq)
       │
       ▼
  Optimal k-mer set + Biological interpretation + Figures
```

## Installation

```bash
pip install kmeropt
```

## Quick Start

### Selection

```bash
kmeropt select --koc koc_dir/ --pheno pheno.txt --out results/
```

```python
from kmerOpt import select_kmer
ks = select_kmer(X, y, r2_threshold=0.7)
print(ks.summary())  # h2=0.66, optimal_n=2000
```

### Annotation

```bash
kmeropt annotate --kmers hits.csv --genome rice.fa --gff rice.gff
```

```python
from kmerOpt import annotate_kmers
ka = annotate_kmers(kmers, genome_fasta='rice.fa', gff_path='rice.gff')
print(ka.variant_summary_)  # {'SNP': 234, 'PAV': 89, 'CNV': 53}
```

## Output

```
results/
├── heritability_curve.csv       # n_markers, h2, SE
├── selected_kmers.txt           # LD-pruned k-mer list
├── anova_validation.csv         # Tertile ANOVA results
├── annotation.csv               # Per-kmer: chr, pos, variant_type, genes
├── candidate_genes.csv          # Gene-level annotation
└── enrichment.csv               # GO/KEGG enrichment
```

## Citation

[Manuscript in preparation]

## References

- Haseman JK, Elston RC (1972) *Behav Genet* 2:3-19
- Yang J et al. (2010) *Nat Genet* 42:565-569
- Voichek & Weigel (2020) *Nat Genet* 52:534-540
- Zhang X et al. (2026) KMERIA. *Nat Genet*

## License

MIT
