"""K-mer to reference genome mapping module."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def map_to_genome(kmer_list: List[str], fasta_path: str,
                  max_mismatches: int = 0) -> pd.DataFrame:
    """Map k-mer sequences to a reference genome via exact string match.

    Parameters
    ----------
    kmer_list : list of str
        K-mer sequences (31-bp).
    fasta_path : str
        Path to reference genome FASTA.
    max_mismatches : int
        Maximum allowed mismatches (0 = exact match only).

    Returns
    -------
    hits : pd.DataFrame
        Columns: kmer, chr, pos, strand
    """
    # Load genome
    chroms = {}
    current = None
    seq_chunks = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current:
                    chroms[current] = ''.join(seq_chunks)
                current = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if current:
            chroms[current] = ''.join(seq_chunks)

    # Map k-mers
    hits = []
    for kmer in kmer_list:
        for chr_name, chr_seq in chroms.items():
            pos = chr_seq.find(kmer)
            if pos >= 0:
                try:
                    chr_num = int(chr_name.replace('Chr','').replace('chr',''))
                except ValueError:
                    chr_num = 0
                hits.append({'kmer': kmer, 'chr': chr_num, 'pos': pos + 1})
                break
        # k-mer not found → not in hits

    return pd.DataFrame(hits)


def annotate_genes(hits_df: pd.DataFrame, gff_path: str,
                   window_bp: int = 50000) -> pd.DataFrame:
    """Annotate k-mer hits with overlapping genes from GFF.

    Parameters
    ----------
    hits_df : pd.DataFrame
        From map_to_genome(). Must have chr, pos columns.
    gff_path : str
        Path to GFF annotation file.
    window_bp : int
        Window around each hit to search for genes.

    Returns
    -------
    annotated : pd.DataFrame
        hits_df with added gene_id, gene_name, distance columns.
    """
    import re
    genes = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            gid = re.search(r'ID=([^;]+)', parts[8])
            genes.append({
                'chr': parts[0], 'start': int(parts[3]), 'end': int(parts[4]),
                'gene_id': gid.group(1) if gid else ''
            })

    results = []
    for _, hit in hits_df.iterrows():
        nearby = [g for g in genes
                  if g['chr'].replace('Chr','') == str(hit['chr'])
                  and abs(hit['pos'] - (g['start']+g['end'])/2) < window_bp]
        if nearby:
            best = min(nearby, key=lambda g: abs(hit['pos'] - (g['start']+g['end'])/2))
            dist = abs(hit['pos'] - (best['start']+best['end'])/2)
            results.append({**hit.to_dict(), 'gene_id': best['gene_id'],
                           'distance_bp': dist})
        else:
            results.append({**hit.to_dict(), 'gene_id': '', 'distance_bp': -1})

    return pd.DataFrame(results)
