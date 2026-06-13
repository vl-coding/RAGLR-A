from typing import Any, Dict, List, Optional, Set

import chromadb
from sentence_transformers import SentenceTransformer

from .schemas import Paper

# Hard ceiling on adaptive over-fetch, expressed as a multiple of top_n.
_MAX_OVERFETCH_MULTIPLIER = 400


class DenseRetriever:
    def __init__(self, model_name: str, persist_dir: str, skip_filter_threshold: float = 0.4):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("arxiv_papers")
        self.skip_filter_threshold = skip_filter_threshold

    def build_index(self, papers: List[Paper], batch_size: int = 128) -> None:
        for start in range(0, len(papers), batch_size):
            batch = papers[start:start + batch_size]

            texts = [paper.text for paper in batch]
            ids = [paper.arxiv_id for paper in batch]

            metadatas = [
                {
                    "title": paper.title,
                    "year": paper.year,
                    "categories": ",".join(paper.categories),
                    "arxiv_id": paper.arxiv_id,
                }
                for paper in batch
            ]

            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    def search(
        self,
        query_text: str,
        candidate_ids: Optional[Set[str]] = None,
        top_n: int = 100,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
        )[0].tolist()

        total = self.collection.count()
        if total == 0:
            return []

        candidate_set = set(candidate_ids) if candidate_ids else None

        # If the candidate set covers most of the corpus, it isn't meaningfully
        # narrowing the search -- skip post-filtering and return Chroma's
        # unfiltered global top-k, symmetric to pipeline.py's
        # min_prefilter_candidates fallback on the small-candidate-set side.
        if candidate_set is not None and len(candidate_set) / total >= self.skip_filter_threshold:
            candidate_set = None

        # Fetch top_n * 50 results and post-filter by candidate_ids, doubling
        # the pool (up to a hard ceiling or the full corpus) if that isn't
        # enough to find top_n candidate-set members. Embedding ranking
        # surfaces relevant papers near the top, so this is usually a single
        # query, but large candidate sets may need a wider pool.
        n_query = min(total, top_n * 50 if candidate_set else top_n)
        max_n_query = min(total, top_n * _MAX_OVERFETCH_MULTIPLIER)

        while True:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, n_query),
            )

            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]

            output = []
            rank = 1
            for doc_id, distance in zip(ids, distances):
                if candidate_set is not None and doc_id not in candidate_set:
                    continue
                output.append({
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": float(1 - distance),
                    "source": "dense",
                })
                rank += 1
                if len(output) >= top_n:
                    break

            if len(output) >= top_n or n_query >= max_n_query:
                break

            n_query = min(n_query * 2, max_n_query)

        return output
