from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import numpy as np
import bm25s

from .schemas import Paper
from .keyword_index import tokenize


class BM25Retriever:
    def __init__(self):
        self.arxiv_ids: List[str] = []
        self.paper_id_to_index: Dict[str, int] = {}
        self._bm25: Optional[bm25s.BM25] = None

    def build_index(self, papers: List[Paper]) -> None:
        self.arxiv_ids = [p.arxiv_id for p in papers]
        self.paper_id_to_index = {aid: i for i, aid in enumerate(self.arxiv_ids)}
        corpus_tokens = [tokenize(p.text) for p in papers]
        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens)

    def search(
        self,
        query: str,
        candidate_ids: Optional[Set[str]] = None,
        top_n: int = 100,
    ) -> List[Dict[str, Any]]:
        query_tokens = [tokenize(query)]
        n_docs = len(self.arxiv_ids)

        # When filtering to candidates, retrieve enough to cover the candidate set.
        # Candidates come from keyword pre-filter (max ~50K per config), so 2x covers all.
        k = min(n_docs, max(top_n * 10, len(candidate_ids) * 2) if candidate_ids else top_n)

        doc_indices, scores = self._bm25.retrieve(query_tokens, k=k)

        results = []
        rank = 1
        for idx, score in zip(doc_indices[0], scores[0]):
            aid = self.arxiv_ids[int(idx)]
            if candidate_ids and aid not in candidate_ids:
                continue
            results.append({
                "doc_id": aid,
                "rank": rank,
                "score": float(score),
                "source": "bm25",
            })
            rank += 1
            if rank > top_n:
                break

        return results

    def save(self, path: str) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(out / "index"))
        np.save(str(out / "arxiv_ids.npy"), np.array(self.arxiv_ids, dtype=object))

    @staticmethod
    def load(path: str) -> "BM25Retriever":
        out = Path(path)
        r = BM25Retriever()
        r._bm25 = bm25s.BM25.load(str(out / "index"), load_corpus=False)
        r.arxiv_ids = np.load(str(out / "arxiv_ids.npy"), allow_pickle=True).tolist()
        r.paper_id_to_index = {aid: i for i, aid in enumerate(r.arxiv_ids)}
        return r
