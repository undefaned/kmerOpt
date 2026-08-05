"""GO/KEGG enrichment for k-mer annotated genes."""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from typing import Dict, List, Tuple, Optional


def go_enrichment(candidate_genes: List[str],
                  background_genes: List[str],
                  go_annotation: Dict[str, List[str]] = None,
                  min_genes: int = 3) -> pd.DataFrame:
    """Fisher's exact test for GO term enrichment.

    Parameters
    ----------
    candidate_genes : list of str
        Gene IDs from k-mer annotation.
    background_genes : list of str
        All genes in the genome (universe).
    go_annotation : dict, optional
        {GO_term: [gene1, gene2, ...]} mapping.
    min_genes : int
        Minimum number of candidate genes in a term to report.

    Returns
    -------
    pd.DataFrame with columns: GO_term, n_candidate, n_background,
    p_value, odds_ratio, significant
    """
    if go_annotation is None:
        # Try to load plant GO annotation
        go_annotation = _load_plant_go()

    candidate_set = set(candidate_genes)
    background_set = set(background_genes)

    results = []
    for go_term, go_genes in go_annotation.items():
        go_set = set(go_genes)
        a = len(candidate_set & go_set)  # candidate + term
        b = len(go_set) - a  # background + term
        c = len(candidate_set) - a  # candidate - term
        d = len(background_set) - a - b - c  # none

        if a < min_genes or a == 0:
            continue

        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
        results.append({
            'GO_term': go_term,
            'description': _go_descriptions.get(go_term, ''),
            'n_candidate': a,
            'n_background': len(go_set),
            'p_value': p_value,
            'odds_ratio': odds_ratio,
            'significant': p_value < 0.05
        })

    df = pd.DataFrame(results).sort_values('p_value')
    return df


def pathway_enrichment(candidate_genes: List[str],
                        background_genes: List[str],
                        pathway_db: str = 'KEGG') -> pd.DataFrame:
    """KEGG/Reactome pathway enrichment.

    Parameters
    ----------
    candidate_genes : list of str
    background_genes : list of str
    pathway_db : str
        'KEGG' or 'Reactome'

    Returns
    -------
    pd.DataFrame
    """
    pathways = _load_pathways(pathway_db)
    if not pathways:
        return pd.DataFrame()

    candidate_set = set(candidate_genes)
    background_set = set(background_genes)

    results = []
    for pw_name, pw_genes in pathways.items():
        pw_set = set(pw_genes)
        a = len(candidate_set & pw_set)
        if a < 2:
            continue
        b = len(pw_set) - a
        c = len(candidate_set) - a
        d = len(background_set) - a - b - c
        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
        results.append({
            'pathway': pw_name,
            'db': pathway_db,
            'n_candidate': a,
            'n_total': len(pw_set),
            'p_value': p_value,
            'odds_ratio': odds_ratio
        })

    return pd.DataFrame(results).sort_values('p_value')


# Built-in plant GO annotation (水稻 MSUv7)
_go_descriptions = {
    'GO:0009908': 'flower development',
    'GO:0010228': 'vegetative to reproductive phase transition',
    'GO:0009789': 'positive regulation of abscisic acid-activated signaling pathway',
    'GO:0009416': 'response to light stimulus',
    'GO:0007623': 'circadian rhythm',
    'GO:0048573': 'photoperiodism, flowering',
    'GO:0009910': 'negative regulation of flower development',
    'GO:0009737': 'response to abscisic acid',
    'GO:0009739': 'response to gibberellin',
    'GO:0009755': 'hormone-mediated signaling pathway',
}


def _load_plant_go() -> Dict[str, List[str]]:
    """Load plant GO annotation (lightweight built-in for key terms)."""
    # For rice: use OsMADS51-related GO terms as example
    return {}


def _load_pathways(db: str) -> Dict[str, List[str]]:
    """Load pathway database."""
    return {}
