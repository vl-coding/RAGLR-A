import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from .schemas import SearchResponse, PaperResult, RetrievalTrace
from .data_ingestion import load_papers_jsonl
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


class RagLiteraturePipeline:
    def __init__(self, config: dict):
        self.config = config

        self.papers = load_papers_jsonl(config["data"]["processed_path"])
        self.paper_lookup = {paper.arxiv_id: paper for paper in self.papers}

        self.keyword_index = load_keyword_index(config["paths"]["keyword_index"])

        self.qwen = QwenKeywordExtractor(config["models"]["qwen_model"])
        self.hyde = ClaudeHyDE(config["models"]["claude_model"])
        self.justifier = ClaudeJustifier(config["models"]["claude_model"])

        self.dense = DenseRetriever(
            model_name=config["models"]["embedding_model"],
            persist_dir=config["paths"]["dense_index_dir"],
        )

        self.bm25 = BM25Retriever.load(config["paths"]["bm25_index"])

    def run(
        self,
        query: str,
        selected_fields: List[str],
        top_k: int = 10,
        use_qwen_prefilter: bool = True,
        use_claude_justification: bool = True,
    ) -> SearchResponse:
        start_total = time.time()

        total_corpus_size = len(self.papers)

        field_filtered = filter_by_academic_fields(
            self.papers,
            selected_fields,
            self.config
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
                mode="union"
            )

            final_candidate_ids = field_filtered_ids & keyword_candidate_ids

            if len(final_candidate_ids) < self.config["retrieval"]["min_prefilter_candidates"]:
                final_candidate_ids = field_filtered_ids
        else:
            final_candidate_ids = field_filtered_ids
            hyde_document = self.hyde.generate(query)

        keyword_filtered = filter_by_candidate_ids(
            field_filtered,
            final_candidate_ids
        )

        keyword_filtered_size = len(keyword_filtered)

        dense_start = time.time()
        dense_results = self.dense.search(
            query_text=hyde_document,
            candidate_ids=list(final_candidate_ids),
            top_n=self.config["retrieval"]["dense_candidates"],
        )
        dense_latency = time.time() - dense_start

        bm25_start = time.time()
        bm25_results = self.bm25.search(
            query=query,
            candidate_ids=final_candidate_ids,
            top_n=self.config["retrieval"]["bm25_candidates"],
        )
        bm25_latency = time.time() - bm25_start

        fused = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            k=self.config["retrieval"]["rrf_k"],
        )

        top_items = fused[:top_k]

        justifications = {}
        if use_claude_justification:
            def _justify(item):
                paper = self.paper_lookup[item["doc_id"]]
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
            paper = self.paper_lookup[item["doc_id"]]
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
                total_corpus_size,
                field_filtered_size,
            ),
            reduction_percent_after_keyword_filter=reduction_percent(
                total_corpus_size,
                keyword_filtered_size,
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
                "retrieval_method": "field_filter + qwen_prefilter + hyde + dense + bm25 + rrf"
            }
        )