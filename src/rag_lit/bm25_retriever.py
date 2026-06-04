import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from rank_bm25 import BM25Okapi

from .schemas import Paper
from .keyword_index import tokenize


class BM25Retriever:
    def __init__(self):
        self.papers: List[Paper] = []
        self.bm25 = None
        self.paper_id_to_index = {}

    def build_index(self, papers: List[Paper]) -> None:
        self.papers = papers
        self.paper_id_to_index = {
            paper.arxiv_id: i for i, paper in enumerate(papers)
        }

        tokenized = [tokenize(paper.text) for paper in papers]
        self.bm25 = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        candidate_ids: Optional[Set[str]] = None,
        top_n: int = 100,
    ) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        if candidate_ids:
            candidate_indices = [
                self.paper_id_to_index[doc_id]
                for doc_id in candidate_ids
                if doc_id in self.paper_id_to_index
            ]
        else:
            candidate_indices = list(range(len(self.papers)))

        ranked = sorted(
            [(i, scores[i]) for i in candidate_indices],
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for rank, (i, score) in enumerate(ranked[:top_n], start=1):
            results.append({
                "doc_id": self.papers[i].arxiv_id,
                "rank": rank,
                "score": float(score),
                "source": "bm25"
            })

        return results

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BM25Retriever":
        with open(path, "rb") as f:
            return pickle.load(f)