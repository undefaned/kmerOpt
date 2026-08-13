"""K-mer to reference genome mapping module.

Mapping is done by exact string match against an inverted index, so the
genome is scanned *once* regardless of how many k-mers are queried. This
makes the mapping step O(L + N) instead of O(L x N), where L is the genome
length and N is the number of query k-mers.

For very large genomes or millions of query k-mers, an external aligner
(BWA/bowtie2) or a persistent index (pyfaidx) is faster; this module
favours zero-dependency correctness over raw throughput.
"""

import pandas as pd
from typing import Dict, List, Tuple, Iterable


_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(_COMPLEMENT)[::-1]


def load_genome(fasta_path: str) -> Dict[str, str]:
    """Load a FASTA file into ``{chromosome_name: sequence}``.

    Header lines use the first whitespace-delimited token as the name.
    Sequence lines are stripped and concatenated (soft-wrapped FASTA OK).
    """
    chroms = {}
    current = None
    chunks = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current is not None:
                    chroms[current] = "".join(chunks)
                current = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current is not None:
        chroms[current] = "".join(chunks)
    return chroms


def build_kmer_index(genome: Dict[str, str],
                     kmers: Iterable[str]) -> Dict[str, List[Tuple[str, int]]]:
    """Build an inverted index ``{kmer: [(chrom, 1-based pos), ...]}``.

    Only the query k-mers are indexed, so memory is bounded by the query
    set, not the genome size. Positions are 1-based, matching GFF/BED
    conventions. Matching is case-insensitive (sequences are upper-cased).
    """
    wanted = {k.upper() for k in kmers}
    if not wanted:
        return {}
    k = len(next(iter(wanted)))
    index = {km: [] for km in wanted}
    for chrom, seq in genome.items():
        seq = seq.upper()
        last = len(seq) - k
        for i in range(last + 1):
            km = seq[i:i + k]
            if km in wanted:
                index[km].append((chrom, i + 1))
    return index


def map_to_genome(kmer_list: List[str], fasta_path: str) -> pd.DataFrame:
    """Map k-mer sequences to a reference genome by exact string match.

    Parameters
    ----------
    kmer_list : list of str
        K-mer sequences (e.g. 31-bp).
    fasta_path : str
        Path to reference genome FASTA.

    Returns
    -------
    hits : pd.DataFrame
        Columns: ``kmer, chr, pos, n_copies``. ``pos`` is 1-based;
        ``n_copies`` counts every exact occurrence in the genome. K-mers
        absent from the reference are omitted — use
        :class:`kmerOpt.KmerAnnotator` to classify those as PAV.
    """
    genome = load_genome(fasta_path)
    index = build_kmer_index(genome, kmer_list)

    rows = []
    for km in kmer_list:
        positions = index.get(km.upper(), [])
        if not positions:
            continue
        chrom, pos = positions[0]
        rows.append({"kmer": km, "chr": chrom, "pos": pos,
                     "n_copies": len(positions)})
    return pd.DataFrame(rows)


def annotate_genes(hits_df: pd.DataFrame, gff_path: str,
                   window_bp: int = 50000) -> pd.DataFrame:
    """Annotate k-mer hits with the nearest gene from a GFF file.

    Parameters
    ----------
    hits_df : pd.DataFrame
        From :func:`map_to_genome`. Must have ``chr`` and ``pos`` columns.
    gff_path : str
        Path to GFF annotation.
    window_bp : int
        Maximum distance (bp) from the gene centre to consider a hit.

    Returns
    -------
    annotated : pd.DataFrame
        ``hits_df`` with added ``gene_id`` and ``distance_bp`` columns.
        ``distance_bp == -1`` means no gene within the window.
    """
    import re

    genes = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            gid = re.search(r"ID=([^;]+)", parts[8])
            genes.append({
                "chr": _norm_chr(parts[0]),
                "start": int(parts[3]),
                "end": int(parts[4]),
                "gene_id": gid.group(1) if gid else "",
            })

    results = []
    for _, hit in hits_df.iterrows():
        chrom = _norm_chr(str(hit["chr"]))
        pos = int(hit["pos"])
        nearby = [g for g in genes
                  if g["chr"] == chrom
                  and abs(pos - (g["start"] + g["end"]) / 2) < window_bp]
        if nearby:
            best = min(nearby,
                       key=lambda g: abs(pos - (g["start"] + g["end"]) / 2))
            results.append({**hit.to_dict(), "gene_id": best["gene_id"],
                            "distance_bp":
                                abs(pos - (best["start"] + best["end"]) / 2)})
        else:
            results.append({**hit.to_dict(), "gene_id": "", "distance_bp": -1})

    return pd.DataFrame(results)


def _norm_chr(chrom: str) -> str:
    """Normalize a chromosome name for comparison (strip ``Chr``/``chr``)."""
    s = chrom.strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    return s
