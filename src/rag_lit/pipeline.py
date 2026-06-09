import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set

from .schemas import Paper, SearchResponse, PaperResult, RetrievalTrace
from .preprocessing import (
    filter_by_academic_fields,
    filter_by_candidate_ids,
    reduction_percent,
)
from .keyword_index import load_keyword_index, candidate_ids_from_keywords
from .qwen_prefilter import QwenKeywordExtractor
from .hyde import ClaudeHyDE
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .rrf import reciprocal_rank_fusion
from .justifier import ClaudeJustifier


class _PaperMeta(NamedTuple):
    arxiv_id: str
    categories: list


class RagLiteraturePipeline:
    def __init__(self, config: dict):
        self.config = config
        self._reload_lock = threading.Lock()

        self._jsonl_path = config["data"]["processed_path"]
        self._delta_jsonl_path = config["data"].get("delta_path", "")

        print("Building metadata index ...", flush=True)
        self._all_meta, self._main_offsets = self._build_meta_index(self._jsonl_path)
        self._delta_offsets: Dict[str, int] = {}
        self._delta_ids: Set[str] = set()
        self._delta_read_pos: int = 0  # bytes consumed from delta JSONL so far

        if self._delta_jsonl_path and Path(self._delta_jsonl_path).exists():
            self._load_delta_meta_from(0)

        print(f"Metadata index ready: {len(self._all_meta):,} papers", flush=True)

        self.keyword_index = load_keyword_index(config["paths"]["keyword_index"])
        kw_path = Path(config["paths"]["keyword_index"])
        self._kw_mtime = kw_path.stat().st_mtime if kw_path.exists() else 0.0

        delta_bm25_path = Path(config["paths"]["bm25_delta"])
        self.bm25_delta: Optional[BM25Retriever] = None
        self._delta_bm25_mtime: float = 0.0
        if delta_bm25_path.exists():
            self.bm25_delta = BM25Retriever.load(str(delta_bm25_path))
            self._delta_bm25_mtime = delta_bm25_path.stat().st_mtime

        self._delta_jsonl_mtime: float = (
            Path(self._delta_jsonl_path).stat().st_mtime
            if self._delta_jsonl_path and Path(self._delta_jsonl_path).exists()
            else 0.0
        )

        self.qwen = QwenKeywordExtractor(config["models"]["qwen_model"])
        self.hyde = ClaudeHyDE(config["models"]["claude_model"])
        self.justifier = ClaudeJustifier(config["models"]["claude_model"])

        self.dense = DenseRetriever(
            model_name=config["models"]["embedding_model"],
            persist_dir=config["paths"]["dense_index_dir"],
        )

        self.bm25 = BM25Retriever.load(config["paths"]["bm25_index"])

    # ------------------------------------------------------------------
    # Metadata index helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_meta_index(jsonl_path: str):
        meta: List[_PaperMeta] = []
        offsets: Dict[str, int] = {}
        with open(jsonl_path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                arxiv_id = obj.get("arxiv_id", "")
                categories = obj.get("categories", [])
                meta.append(_PaperMeta(arxiv_id, categories))
                offsets[arxiv_id] = offset
        return meta, offsets

    def _load_delta_meta_from(self, byte_pos: int) -> None:
        """Read new lines from the delta JSONL starting at byte_pos."""
        delta_path = self._delta_jsonl_path
        if not delta_path or not Path(delta_path).exists():
            return
        with open(delta_path, "rb") as f:
            f.seek(byte_pos)
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    self._delta_read_pos = offset
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                arxiv_id = obj.get("arxiv_id", "")
                if not arxiv_id or arxiv_id in self._delta_offsets:
                    continue
                categories = obj.get("categories", [])
                self._all_meta.append(_PaperMeta(arxiv_id, categories))
                self._delta_offsets[arxiv_id] = offset
                self._delta_ids.add(arxiv_id)
            self._delta_read_pos = f.tell()

    def _load_paper(self, arxiv_id: str) -> Paper:
        if arxiv_id in self._delta_offsets:
            path = self._delta_jsonl_path
            offset = self._delta_offsets[arxiv_id]
        else:
            path = self._jsonl_path
            offset = self._main_offsets[arxiv_id]
        with open(path, "rb") as f:
            f.seek(offset)
            line = f.readline()
        return Paper.model_validate_json(line)

    # ------------------------------------------------------------------
    # Hot-reload (called before each query, costs one stat() per index)
    # ------------------------------------------------------------------

    def _maybe_reload(self) -> None:
        kw_path = Path(self.config["paths"]["keyword_index"])
        delta_bm25_path = Path(self.config["paths"]["bm25_delta"])
        delta_jsonl_path = Path(self._delta_jsonl_path) if self._delta_jsonl_path else None

        needs_kw = kw_path.stat().st_mtime > self._kw_mtime
        needs_delta_bm25 = (
            delta_bm25_path.exists()
            and delta_bm25_path.stat().st_mtime > self._delta_bm25_mtime
        )
        needs_delta_meta = (
            delta_jsonl_path is not None
            and delta_jsonl_path.exists()
            and delta_jsonl_path.stat().st_mtime > self._delta_jsonl_mtime
        )

        if not (needs_kw or needs_delta_bm25 or needs_delta_meta):
            return

        with self._reload_lock:
            # Re-check under lock (double-checked locking)
            if kw_path.stat().st_mtime > self._kw_mtime:
                self.keyword_index = load_keyword_index(str(kw_path))
                self._kw_mtime = kw_path.stat().st_mtime

            if delta_bm25_path.exists() and delta_bm25_path.stat().st_mtime > self._delta_bm25_mtime:
                self.bm25_delta = BM25Retriever.load(str(delta_bm25_path))
                self._delta_bm25_mtime = delta_bm25_path.stat().st_mtime

            if (
                delta_jsonl_path is not None
                and delta_jsonl_path.exists()
                and delta_jsonl_path.stat().st_mtime > self._delta_jsonl_mtime
            ):
                self._load_delta_meta_from(self._delta_read_pos)
                self._delta_jsonl_mtime = delta_jsonl_path.stat().st_mtime

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        selected_fields: List[str],
        top_k: int = 10,
        use_qwen_prefilter: bool = True,
        use_claude_justification: bool = True,
    ) -> SearchResponse:
        self._maybe_reload()

        start_total = time.time()
        total_corpus_size = len(self._all_meta)

        field_filtered = filter_by_academic_fields(
            self._all_meta,
            selected_fields,
            self.config,
        )
        field_filtered_size = len(field_filtered)
        field_filtered_ids = {paper.arxiv_id for paper in field_filtered}

        generated_keywords = []

        if use_qwen_prefilter:
            with ThreadPoolExecutor(max_workers=2) as executor:
                qwen_future = executor.submit(self.qwen.generate_keywords, query)
                hyde_future = executor.submit(self.hyde.generate, query)
                generated_keywords = qwen_future.result()
                hyde_document = hyde_future.result()

            keyword_candidate_ids = candidate_ids_from_keywords(
                keywords=generated_keywords,
                keyword_index=self.keyword_index,
                mode="union",
            )
            final_candidate_ids = field_filtered_ids & keyword_candidate_ids

            if len(final_candidate_ids) < self.config["retrieval"]["min_prefilter_candidates"]:
                final_candidate_ids = field_filtered_ids
        else:
            final_candidate_ids = field_filtered_ids
            hyde_document = self.hyde.generate(query)

        keyword_filtered = filter_by_candidate_ids(field_filtered, final_candidate_ids)
        keyword_filtered_size = len(keyword_filtered)

        dense_start = time.time()
        dense_results = self.dense.search(
            query_text=hyde_document,
            candidate_ids=list(final_candidate_ids),
            top_n=self.config["retrieval"]["dense_candidates"],
        )
        dense_latency = time.time() - dense_start

        bm25_start = time.time()
        top_n = self.config["retrieval"]["bm25_candidates"]

        # Route candidates: delta papers go to delta BM25, rest go to main BM25
        if self._delta_ids:
            main_candidate_ids = final_candidate_ids - self._delta_ids
            delta_candidate_ids = final_candidate_ids & self._delta_ids
        else:
            main_candidate_ids = final_candidate_ids
            delta_candidate_ids = set()

        bm25_results = self.bm25.search(
            query=query,
            candidate_ids=main_candidate_ids if main_candidate_ids else final_candidate_ids,
            top_n=top_n,
        )

        bm25_delta_results = []
        if self.bm25_delta is not None and delta_candidate_ids:
            bm25_delta_results = self.bm25_delta.search(
                query=query,
                candidate_ids=delta_candidate_ids,
                top_n=top_n,
            )

        bm25_latency = time.time() - bm25_start

        ranked_lists = [dense_results, bm25_results]
        if bm25_delta_results:
            ranked_lists.append(bm25_delta_results)

        fused = reciprocal_rank_fusion(ranked_lists, k=self.config["retrieval"]["rrf_k"])
        top_items = fused[:top_k]

        top_papers = {item["doc_id"]: self._load_paper(item["doc_id"]) for item in top_items}

        justifications = {}
        if use_claude_justification:
            def _justify(item):
                paper = top_papers[item["doc_id"]]
                return item["doc_id"], self.justifier.justify(
                    query=query,
                    title=paper.title,
                    abstract=paper.abstract,
                )

            with ThreadPoolExecutor(max_workers=top_k) as executor:
                futures = {executor.submit(_justify, item): item for item in top_items}
                for future in as_completed(futures):
                    doc_id, result = future.result()
                    justifications[doc_id] = result

        final_results = []
        for item in top_items:
            paper = top_papers[item["doc_id"]]
            justification = justifications.get(item["doc_id"], {})
            final_results.append(
                PaperResult(
                    rank=item["rank"],
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    authors=paper.authors,
                    year=paper.year,
                    categories=paper.categories,
                    url=paper.url,
                    abstract_snippet=paper.abstract[:500],
                    rrf_score=item["rrf_score"],
                    dense_rank=item.get("dense_rank"),
                    bm25_rank=item.get("bm25_rank"),
                    contribution=justification.get("contribution"),
                    relevance_justification=justification.get("relevance_justification"),
                    relevance_score=justification.get("relevance_score"),
                    specificity_score=justification.get("specificity_score"),
                )
            )

        trace = RetrievalTrace(
            total_corpus_size=total_corpus_size,
            field_filtered_size=field_filtered_size,
            keyword_filtered_size=keyword_filtered_size,
            reduction_percent_after_field_filter=reduction_percent(
                total_corpus_size, field_filtered_size
            ),
            reduction_percent_after_keyword_filter=reduction_percent(
                total_corpus_size, keyword_filtered_size
            ),
            generated_keywords=generated_keywords,
            selected_fields=selected_fields,
            hyde_document=hyde_document,
            dense_latency_seconds=round(dense_latency, 3),
            bm25_latency_seconds=round(bm25_latency, 3),
            total_latency_seconds=round(time.time() - start_total, 3),
        )

        return SearchResponse(
            query=query,
            results=final_results,
            trace=trace,
            metadata={
                "pipeline_version": "v1",
                "retrieval_method": "field_filter + qwen_prefilter + hyde + dense + bm25 + rrf",
            },
        )
