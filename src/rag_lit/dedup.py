"""Result-level near-duplicate detection (issue #21).

Multiple *versions* of the same paper are already deduplicated by arXiv ID
elsewhere in the pipeline. This module addresses the separate case of
substantively similar papers that were assigned *different* arXiv IDs
(e.g. independent submissions covering very similar work, or near-identical
preprints by different author groups).

Corpus-wide pairwise comparison is infeasible at ~3M papers, so detection is
scoped to the small final result set returned for a single query (typically
top 10-25 papers after RRF fusion). This keeps the comparison O(N^2) on a
tiny N, which is cheap (<=25 -> at most 300 pairs).

Similarity is computed on the same SBERT embeddings (title + abstract,
`all-MiniLM-L6-v2`, L2-normalized) already used for dense retrieval, so no
extra model is introduced.
"""
from typing import Dict, List, Sequence

import numpy as np

# Cosine similarity threshold above which two results in the same result set
# are flagged as likely near-duplicates.
#
# Rationale: with all-MiniLM-L6-v2 (L2-normalized embeddings, so cosine
# similarity == dot product), unrelated abstracts in this domain typically
# score well below 0.5, and topically-related-but-distinct papers (e.g. two
# different papers on "transformer architectures for time series") commonly
# land in the 0.6-0.85 range. Scores above ~0.92 are reserved for cases where
# the title+abstract text is nearly word-for-word identical or differs only
# in minor phrasing -- the regime we want to flag as "possibly the same
# paper under a different arXiv ID", while avoiding false positives between
# merely-similar-topic papers. 0.92 is a conservative starting point that
# favors precision (few false positives) over recall; it can be tuned down
# if evaluation shows true near-duplicates are being missed.
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.92


def find_near_duplicates(
    arxiv_ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> Dict[str, List[str]]:
    """Find pairs of near-duplicate papers within a small result set.

    Args:
        arxiv_ids: arXiv IDs of the papers in the result set, in any order.
        embeddings: parallel list of embedding vectors (one per arxiv_id).
            Vectors are L2-normalized internally, so callers may pass either
            normalized or unnormalized vectors.
        threshold: cosine similarity threshold above which two papers are
            considered near-duplicates of each other.

    Returns:
        A dict mapping each arxiv_id that has at least one near-duplicate to
        the sorted list of arxiv_ids (within `arxiv_ids`) it is a near-duplicate
        of. arxiv_ids with no flagged duplicates are omitted from the dict
        (i.e. absence means "no duplicates found").
    """
    n = len(arxiv_ids)
    if n != len(embeddings):
        raise ValueError("arxiv_ids and embeddings must have the same length")

    if n < 2:
        return {}

    vectors = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for any degenerate all-zero embedding.
    norms[norms == 0] = 1.0
    normalized = vectors / norms

    similarity = normalized @ normalized.T

    duplicates: Dict[str, List[str]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                duplicates.setdefault(arxiv_ids[i], []).append(arxiv_ids[j])
                duplicates.setdefault(arxiv_ids[j], []).append(arxiv_ids[i])

    for arxiv_id, others in duplicates.items():
        duplicates[arxiv_id] = sorted(set(others))

    return duplicates
