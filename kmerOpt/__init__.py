"""kmerOpt: K-mer selection, annotation, and interpretation for GWAS.

KMERIA produces a KOC matrix — kmerOpt tells you which k-mers matter,
how many you need, and what they represent biologically.

Core pipeline:
    1. LD pruning (greedy, |r| threshold)
    2. Haseman-Elston heritability estimation (bootstrap SE)
    3. Plateau detection → optimal k-mer count
    4. ANOVA tertile validation
    5. K-mer → reference genome mapping
    6. Variant classification (SNP / SV / PAV / CNV)
    7. Gene annotation + GO/KEGG enrichment
    8. Selection signal detection (Fst / PAV frequency / Tajima's D)

Reference:
    Haseman JK, Elston RC (1972) Behav Genet 2:3-19
    Yang J et al. (2010) Nat Genet 42:565-569 (GCTA/GREML)
    Voichek & Weigel (2020) Nat Genet 52:534-540 (k-mer GWAS)
    Zhang X et al. (2026) Nat Genet (KMERIA)
"""

__version__ = "0.2.0"
__author__ = "Shifan"

# Core selection
from .core import KmerSelector, select_kmer
from .ld_prune import ld_prune
from .heritability import haseman_elston, variance_components
from .validate import anova_tertile, count_valid_markers
from .mapping import map_to_genome, annotate_genes

# Annotation
from .annotate import KmerAnnotator, annotate_kmers
from .enrichment import go_enrichment, pathway_enrichment
from .selection import selection_scan, compute_pav_frequencies, fst_weir_cockerham
