import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional, Set

from .schemas import Paper, SearchResponse, PaperResult, RetrievalTrace, RetrievalDebugInfo
from .preprocessing import (
    filter_by_candidate_ids,
    reduction_percent,
)
from .canonical_boost import load_canonical_papers, match_canonical_papers
from .keyword_index import open_keyword_index_db, candidate_ids_from_keywords
from .metadata_db import build_metadata_db, load_metadata_db
from .qwen_prefilter import QwenKeywordExtractor
from .hyde import ClaudeHyDE
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .rrf import reciprocal_rank_fusion
from .justifier import ClaudeJustifier


class _PaperMeta(NamedTuple):
    arxiv_id: str


class RagLiteraturePipeline:
    def __init__(self, config: dict):
        self.config = config
        self._reload_lock = threading.Lock()

        self._jsonl_path = config["data"]["processed_path"]
        self._delta_jsonl_path = config["data"].get("delta_path", "")

        print("Loading metadata index ...", flush=True)
        self._metadata_db_path = config["paths"]["metadata_db"]
        self._all_meta, self._main_offsets = self._build_meta_index(
            self._jsonl_path, self._metadata_db_path
        )
        self._delta_offsets: Dict[str, int] = {}
        self._delta_ids: Set[str] = set()
        self._delta_read_pos: int = 0  # bytes consumed from delta JSONL so far

        if self._delta_jsonl_path and Path(self._delta_jsonl_path).exists():
            self._load_delta_meta_from(0)

        print(f"Metadata index ready: {len(self._all_meta):,} papers", flush=True)

        self.keyword_index = open_keyword_index_db(config["paths"]["keyword_index"])
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

        self._qwen_model_name = config["models"]["qwen_model"]
        self._qwen: Optional[QwenKeywordExtractor] = None
        claude_max_retries = config["models"].get("claude_max_retries", 5)
        claude_timeout_seconds = config["models"].get("claude_timeout_seconds", 60.0)
        self.hyde = ClaudeHyDE(
            config["models"]["claude_model"], claude_max_retries, claude_timeout_seconds
        )
        self.justifier = ClaudeJustifier(
            config["models"]["claude_model"], claude_max_retries, claude_timeout_seconds
        )
        self._justifier_max_concurrency = config["models"].get(
            "claude_justifier_max_concurrency", 5
        )

        self.dense = DenseRetriever(
            model_name=config["models"]["embedding_model"],
            persist_dir=config["paths"]["dense_index_dir"],
            skip_filter_threshold=config["retrieval"].get("dense_skip_filter_threshold_percent", 40) / 100,
        )

        self.bm25 = BM25Retriever.load(
            config["paths"]["bm25_index"],
            mmap=config["retrieval"].get("bm25_mmap", False),
        )

        self._canonical_papers = load_canonical_papers(
            config["retrieval"].get("canonical_papers_path", "")
        )

    # ------------------------------------------------------------------
    # Metadata index helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_meta_index(jsonl_path: str, db_path: str):
        """Load (arxiv_id, byte offset) for every paper.

        Reads from a SQLite metadata DB (built once, persisted on disk) so
        startup is a single bulk SELECT instead of json.loads-ing every line
        of a multi-GB JSONL file. The DB is (re)built automatically if it's
        missing or older than the JSONL it was built from -- run
        scripts/build_metadata_db.py ahead of time to avoid paying that cost
        on a live session.
        """
        db_file = Path(db_path)
        jsonl_mtime = Path(jsonl_path).stat().st_mtime
        if not db_file.exists() or db_file.stat().st_mtime < jsonl_mtime:
            print(
                f"Metadata DB at {db_path} is missing or stale -- rebuilding "
                f"from {jsonl_path} (one-time cost; future sessions load "
                f"this DB directly).",
                flush=True,
            )
            build_metadata_db(jsonl_path, db_path)

        rows, offsets = load_metadata_db(db_path)
        meta = [_PaperMeta(arxiv_id) for arxiv_id, _categories in rows]
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
                self._all_meta.append(_PaperMeta(arxiv_id))
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
                self.keyword_index.close()
                self.keyword_index = open_keyword_index_db(str(kw_path))
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
        top_k: int = 10,
        use_qwen_prefilter: bool = True,
        use_claude_justification: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        debug: bool = False,
        hyde_ablation: bool = False,
    ) -> SearchResponse:
        def _report(step: str, fraction: float) -> None:
            if progress_callback is not None:
                progress_callback(step, fraction)

        self._maybe_reload()

        start_total = time.time()
        total_corpus_size = len(self._all_meta)
        all_ids = {paper.arxiv_id for paper in self._all_meta}

        generated_keywords = []
        keyword_candidate_ids = None

        _report("Generating search keywords and hypothetical document ...", 0.1)
        if use_qwen_prefilter:
            if self._qwen is None:
                self._qwen = QwenKeywordExtractor(self._qwen_model_name)
            with ThreadPoolExecutor(max_workers=2) as executor:
                qwen_future = executor.submit(self._qwen.generate_keywords, query)
                hyde_future = executor.submit(self.hyde.generate, query)
                generated_keywords = qwen_future.result()
                hyde_document = hyde_future.result()

            keyword_candidate_ids = candidate_ids_from_keywords(
                keywords=generated_keywords,
                conn=self.keyword_index,
                mode="union",
            )
            final_candidate_ids = all_ids & keyword_candidate_ids

            if len(final_candidate_ids) < self.config["retrieval"]["min_prefilter_candidates"]:
                final_candidate_ids = all_ids
        else:
            final_candidate_ids = all_ids
            hyde_document = self.hyde.generate(query)

        _report("Filtering candidates by keywords ...", 0.35)
        keyword_filtered = filter_by_candidate_ids(self._all_meta, final_candidate_ids)
        keyword_filtered_size = len(keyword_filtered)

        _report("Running dense vector search ...", 0.45)
        dense_start = time.time()
        dense_results = self.dense.search(
            query_text=hyde_document,
            candidate_ids=list(final_candidate_ids),
            top_n=self.config["retrieval"]["dense_candidates"],
        )
        dense_latency = time.time() - dense_start

        dense_results_raw_query = None
        if hyde_ablation:
            dense_results_raw_query = self.dense.search(
                query_text=query,
                candidate_ids=list(final_candidate_ids),
                top_n=self.config["retrieval"]["dense_candidates"],
            )

        _report("Running BM25 keyword search ...", 0.6)
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

        _report("Fusing rankings (RRF) ...", 0.75)
        ranked_lists = [dense_results, bm25_results]
        if bm25_delta_results:
            ranked_lists.append(bm25_delta_results)

        canonical_results = match_canonical_papers(
            query=query,
            keywords=generated_keywords,
            canonical_papers=self._canonical_papers,
            corpus_ids=all_ids,
        )
        if canonical_results:
            ranked_lists.append(canonical_results)

        fused = reciprocal_rank_fusion(ranked_lists, k=self.config["retrieval"]["rrf_k"])
        top_items = fused[:top_k]

        fused_raw_query = None
        if hyde_ablation:
            raw_ranked_lists = [dense_results_raw_query, bm25_results]
            if bm25_delta_results:
                raw_ranked_lists.append(bm25_delta_results)
            if canonical_results:
                raw_ranked_lists.append(canonical_results)
            fused_raw_query = reciprocal_rank_fusion(
                raw_ranked_lists, k=self.config["retrieval"]["rrf_k"]
            )

        _report("Loading paper details ...", 0.8)
        top_papers = {item["doc_id"]: self._load_paper(item["doc_id"]) for item in top_items}

        justifications = {}
        if use_claude_justification:
            _report("Generating relevance justifications ...", 0.9)

            def _justify(item):
                paper = top_papers[item["doc_id"]]
                return item["doc_id"], self.justifier.justify(
                    query=query,
                    title=paper.title,
                    abstract=paper.abstract,
                )

            justify_workers = min(top_k, self._justifier_max_concurrency)
            with ThreadPoolExecutor(max_workers=justify_workers) as executor:
                futures = {executor.submit(_justify, item): item for item in top_items}
                for future in as_completed(futures):
                    doc_id, result = future.result()
                    justifications[doc_id] = result

        canonical_ids = {item["doc_id"] for item in canonical_results}

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
                    canonical_match=item["doc_id"] in canonical_ids,
                    contribution=justification.get("contribution"),
                    relevance_justification=justification.get("relevance_justification"),
                    relevance_score=justification.get("relevance_score"),
                    specificity_score=justification.get("specificity_score"),
                )
            )

        trace = RetrievalTrace(
            total_corpus_size=total_corpus_size,
            keyword_filtered_size=keyword_filtered_size,
            reduction_percent_after_keyword_filter=reduction_percent(
                total_corpus_size, keyword_filtered_size
            ),
            generated_keywords=generated_keywords,
            hyde_document=hyde_document,
            dense_latency_seconds=round(dense_latency, 3),
            bm25_latency_seconds=round(bm25_latency, 3),
            total_latency_seconds=round(time.time() - start_total, 3),
        )

        _report("Done", 1.0)

        debug_info = None
        if debug:
            debug_info = RetrievalDebugInfo(
                keyword_candidate_ids=(
                    sorted(keyword_candidate_ids) if keyword_candidate_ids is not None else None
                ),
                final_candidate_ids=sorted(final_candidate_ids),
                dense_results=dense_results,
                dense_results_raw_query=dense_results_raw_query,
                bm25_results=bm25_results,
                bm25_delta_results=bm25_delta_results,
                canonical_results=canonical_results,
                fused_results_raw_query=fused_raw_query,
            )

        return SearchResponse(
            query=query,
            results=final_results,
            trace=trace,
            metadata={
                "pipeline_version": "v1",
                "retrieval_method": "qwen_prefilter + hyde + dense + bm25 + rrf",
            },
            debug=debug_info,
        )
