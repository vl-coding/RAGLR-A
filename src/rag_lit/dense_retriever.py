from typing import Any, Dict, List, Optional, Set

import chromadb
from sentence_transformers import SentenceTransformer

from .schemas import Paper


class DenseRetriever:
    def __init__(self, model_name: str, persist_dir: str):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("arxiv_papers")

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

        # Query a larger pool then post-filter by candidate_ids since chromadb
        # does not support efficient set-based ID filtering in query().
        if candidate_set:
            n_query = min(total, max(top_n * 5, len(candidate_set)))
        else:
            n_query = min(total, top_n)

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

        return output
