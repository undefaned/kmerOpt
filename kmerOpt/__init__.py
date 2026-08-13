"""kmerOpt: K-mer selection, annotation, and interpretation for GWAS.

KMERIA produces a KOC matrix — kmerOpt tells you which k-mers matter,
how many you need, and what they represent biologically.

Core pipeline:
    0. Ploidy-aware normalization (KMERIA-style depth correction)
    1. LD pruning (greedy, |r| threshold)
    2. Haseman-Elston heritability estimation (bootstrap SE)
    3. Plateau detection → optimal k-mer count
    4. ANOVA tertile validation
    5. Multiple testing correction (modified Bonferroni + FDR)
    6. K-mer → reference genome mapping
    7. Variant classification (SNP / SV / PAV / CNV)
    8. Gene annotation + GO/KEGG enrichment
    9. Selection signal detection (Fst / PAV frequency / Tajima's D)

Reference:
    Haseman JK, Elston RC (1972) Behav Genet 2:3-19
    Yang J et al. (2010) Nat Genet 42:565-569 (GCTA/GREML)
    Voichek & Weigel (2020) Nat Genet 52:534-540 (k-mer GWAS)
    Chen & Liu et al. (2026) Nat Genet 58:1711-1721 (KMERIA)
"""

__version__ = "0.3.0"
__author__ = "Shifan"

# Normalization
from .normalize import (ploidy_aware_normalize, quantile_dosage_encode,
                         binary_encode, mixed_ploidy_normalize,
                         estimate_haploid_depth)

# Core selection
from .core import KmerSelector, select_kmer
from .ld_prune import ld_prune
from .heritability import haseman_elston, variance_components
from .validate import anova_tertile, count_valid_markers
from .mapping import map_to_genome, annotate_genes

# Multiple testing correction
from .correction import (modified_bonferroni_threshold,
                          standard_bonferroni_threshold,
                          benjamini_hochberg, dual_correction,
                          count_significant, compare_thresholds)

# Annotation
from .annotate import KmerAnnotator, annotate_kmers
from .enrichment import go_enrichment, pathway_enrichment
from .selection import selection_scan, compute_pav_frequencies, fst_weir_cockerham
