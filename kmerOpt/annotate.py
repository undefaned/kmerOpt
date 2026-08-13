"""Core: KmerAnnotator — interpret GWAS-significant k-mers biologically."""

import numpy as np
import pandas as pd
import re, os, sys
from typing import Optional, List, Dict, Tuple
from collections import Counter

from .mapping import (load_genome as _load_genome_fasta,
                      build_kmer_index, reverse_complement, _norm_chr)


class KmerAnnotator:
    """Annotate significant k-mers from GWAS with biological context.

    Parameters
    ----------
    kmer_list : list of (str, float)
        (kmer_sequence, p_value) pairs from GWAS.
    genome_fasta : str
        Path to reference genome FASTA.
    gff_path : str, optional
        Path to GFF annotation.
    protein_fasta : str, optional
        Path to protein FASTA for functional annotation.

    Attributes
    ----------
    hits_ : pd.DataFrame
        Mapped k-mers with chr/pos/variant_type/genes.
    variant_summary_ : dict
        Count of SNPs, SVs, PAVs, CNVs detected.
    gene_list_ : list
        Annotated candidate genes.
    """

    def __init__(self, kmer_list: List[Tuple[str, float]] = None,
                 genome_fasta: str = None, gff_path: str = None,
                 protein_fasta: str = None):
        self.kmer_list = kmer_list or []
        self.genome_fasta = genome_fasta
        self.gff_path = gff_path
        self.protein_fasta = protein_fasta

        # Internal state
        self._genome = None
        self._genes = None
        self._proteins = None
        self.hits_ = None
        self.variant_summary_ = {}
        self.gene_list_ = []

    def load_genome(self) -> Dict[str, str]:
        """Load reference genome into memory.

        Returns
        -------
        dict of {chromosome_name: sequence}
        """
        self._genome = _load_genome_fasta(self.genome_fasta)
        return self._genome

    def load_genes(self, feature_type: str = 'gene') -> pd.DataFrame:
        """Parse GFF annotation."""
        if not self.gff_path:
            raise ValueError("GFF path not set")
        genes = []
        with open(self.gff_path) as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split('\t')
                if len(parts) < 9: continue
                if parts[2] != feature_type: continue
                gid = re.search(r'ID=([^;]+)', parts[8])
                gname = re.search(r'Name=([^;]+)', parts[8])
                src = re.search(r'Source_genome=([^;]+)', parts[8])
                genes.append({
                    'chr': parts[0], 'start': int(parts[3]), 'end': int(parts[4]),
                    'strand': parts[6], 'gene_id': gid.group(1) if gid else '',
                    'gene_name': gname.group(1) if gname else '',
                    'source': src.group(1) if src else '',
                    'chr_norm': _norm_chr(parts[0]),
                })
        self._genes = pd.DataFrame(genes)
        return self._genes

    def load_proteins(self) -> Dict[str, str]:
        """Load protein sequences."""
        if not self.protein_fasta:
            return {}
        prots = {}; cur_id = None; seq = []
        with open(self.protein_fasta) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if cur_id: prots[cur_id] = ''.join(seq)
                    cur_id = line[1:].split()[0]; seq = []
                else: seq.append(line)
            if cur_id: prots[cur_id] = ''.join(seq)
        self._proteins = prots
        return prots

    def map_kmers(self, window_bp: int = 50000, n_jobs: int = 1) -> pd.DataFrame:
        """Map ALL k-mers to genome and annotate with overlapping genes.

        Uses a single-pass inverted index (see ``mapping.build_kmer_index``)
        so the genome is scanned once for the whole query set. A k-mer not
        found on the forward strand is re-checked on its reverse complement
        before being classified as PAV.

        Returns
        -------
        pd.DataFrame with columns: kmer, p_value, chr, pos, variant_type, genes
        """
        if self._genome is None:
            self.load_genome()
        if self._genes is None and self.gff_path:
            self.load_genes()

        # Index both strands so a k-mer reported on the reverse strand still
        # resolves (its reverse complement is what appears in the reference).
        seqs = set()
        for km, _ in self.kmer_list:
            seqs.add(km.upper())
            seqs.add(reverse_complement(km).upper())
        index = build_kmer_index(self._genome, seqs)

        hits = []
        for kmer, pval in self.kmer_list:
            hit = {'kmer': kmer, 'p_value': pval, 'chr': '', 'pos': 0,
                   'variant_type': 'PAV', 'genes': [], 'n_copies': 0}
            positions = index.get(kmer.upper(), [])
            if not positions:
                # k-mer may be stored on the reverse strand; before declaring
                # it absent (PAV), check its reverse complement.
                positions = index.get(reverse_complement(kmer).upper(), [])

            if positions:
                chrom = positions[0][0]
                pos = positions[0][1]
                hit['chr'] = chrom
                hit['pos'] = pos
                hit['n_copies'] = len(positions)
                hit['variant_type'] = self._classify_variant(kmer, len(positions))

                if self._genes is not None:
                    nearby = self._genes[
                        (self._genes['chr_norm'] == _norm_chr(chrom)) &
                        (self._genes['start'] <= pos + window_bp) &
                        (self._genes['end'] >= pos - window_bp)
                    ]
                    hit['genes'] = nearby['gene_id'].tolist() if len(nearby) > 0 else []
            hits.append(hit)

        # Summarize variant distribution
        vtypes = Counter(h['variant_type'] for h in hits)
        self.variant_summary_ = dict(vtypes)

        # Collect gene list
        all_genes = set()
        for h in hits:
            all_genes.update(h['genes'])
        self.gene_list_ = list(all_genes)

        self.hits_ = pd.DataFrame(hits)
        return self.hits_

    def _classify_variant(self, kmer: str, n_copies: int) -> str:
        """Classify a k-mer's mapping pattern by copy number.

        Heuristics (copy-number based; sequence identity is NOT checked):
        - n_copies == 0 → PAV (absent from the reference → presence/absence)
        - n_copies == 1 → unique_mapping (a single exact hit; may tag a SNP or
          small indel, but a unique hit alone does NOT prove a point mutation —
          confirming a SNP requires a mismatch-aware alignment)
        - 2-5 copies → CNV_low_copy; >5 copies → CNV_high_copy
        """
        if n_copies == 0:
            return 'PAV'
        elif n_copies == 1:
            return 'unique_mapping'
        elif 2 <= n_copies <= 5:
            return 'CNV_low_copy'
        else:
            return 'CNV_high_copy'

    def annotate_genes(self, window_bp: int = 50000) -> pd.DataFrame:
        """Detailed gene annotation for hit k-mers."""
        if self.hits_ is None:
            self.map_kmers(window_bp=window_bp)
        if self._genes is None:
            raise ValueError("Load GFF first!")

        gene_annotations = []
        for _, hit in self.hits_.iterrows():
            for gid in hit['genes']:
                gene_row = self._genes[self._genes['gene_id'] == gid]
                if len(gene_row) > 0:
                    g = gene_row.iloc[0]
                    gene_annotations.append({
                        'kmer': hit['kmer'][:40],
                        'p_value': hit['p_value'],
                        'gene_id': g['gene_id'],
                        'gene_name': g['gene_name'],
                        'chr': g['chr'],
                        'start': g['start'],
                        'end': g['end'],
                        'strand': g['strand'],
                        'source': g['source'],
                        'variant_type': hit['variant_type']
                    })

        return pd.DataFrame(gene_annotations)

    def cross_reference_qtls(self, qtl_gene_list: List[str],
                              qtl_label: str = 'HD_genes') -> pd.DataFrame:
        """Cross-reference annotated genes with known QTL gene lists.

        Parameters
        ----------
        qtl_gene_list : list of str
            Known gene names (e.g., from Gramene QTL database).
        qtl_label : str
            Label for this QTL set (e.g., 'heading_date', 'domestication').

        Returns
        -------
        pd.DataFrame with overlap statistics.
        """
        if not self.gene_list_:
            raise ValueError("Run map_kmers() first!")

        overlap = set(self.gene_list_) & set(qtl_gene_list)

        return pd.DataFrame({
            'qtl_set': [qtl_label],
            'qtl_genes_total': [len(qtl_gene_list)],
            'kmer_genes_total': [len(self.gene_list_)],
            'overlap': [len(overlap)],
            'overlap_genes': [', '.join(sorted(overlap))],
            'enrichment_p': [self._hypergeom_p(len(overlap), len(self.gene_list_),
                                                len(qtl_gene_list), 69492)]
        })

    def _hypergeom_p(self, k, n, K, N):
        """Hypergeometric enrichment p-value."""
        from scipy.stats import hypergeom
        return hypergeom.sf(k - 1, N, K, n)

    def summary(self) -> str:
        """Generate annotation summary report."""
        lines = ["=" * 60, "kmerAnnotator Summary", "=" * 60]
        lines.append(f"Input: {len(self.kmer_list)} k-mers")

        if self.variant_summary_:
            lines.append("\nVariant Classification:")
            for vt, n in sorted(self.variant_summary_.items(), key=lambda x: -x[1]):
                pct = n / len(self.kmer_list) * 100
                lines.append(f"  {vt}: {n} ({pct:.1f}%)")

        if self.gene_list_:
            lines.append(f"\nCandidate Genes: {len(self.gene_list_)}")
            lines.append(f"  (within +/-50kb of k-mer hits)")

        if self.hits_ is not None:
            chr_dist = self.hits_['chr'].value_counts()
            lines.append("\nChromosome Distribution:")
            for c, n in chr_dist.head(10).items():
                lines.append(f"  {c}: {n} ({n/len(self.hits_)*100:.1f}%)")

        return "\n".join(lines)


def annotate_kmers(kmer_list: List[Tuple[str, float]],
                    genome_fasta: str, gff_path: str = None,
                    protein_fasta: str = None,
                    window_bp: int = 50000) -> KmerAnnotator:
    """Convenience function: full annotation pipeline.

    Usage:
        ka = annotate_kmers(
            [('AAAAGC...', 1.2e-18), ('TTTCGA...', 5.3e-09)],
            genome_fasta='/path/to/genome.fa',
            gff_path='/path/to/annotation.gff'
        )
        print(ka.summary())
    """
    ka = KmerAnnotator(kmer_list, genome_fasta, gff_path, protein_fasta)
    ka.load_genome()
    if gff_path:
        ka.load_genes()
    if protein_fasta:
        ka.load_proteins()
    ka.map_kmers(window_bp=window_bp)
    return ka
