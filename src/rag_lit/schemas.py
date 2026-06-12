from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Paper(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str] = []
    primary_category: Optional[str] = None
    categories: List[str] = []
    category_metadata: List[Dict[str, Any]] = []
    year: int
    published_date: Optional[str] = None
    updated_date: Optional[str] = None
    url: Optional[str] = None

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.abstract}"


class PaperResult(BaseModel):
    rank: int
    arxiv_id: str
    title: str
    authors: List[str] = []
    year: int
    categories: List[str] = []
    url: Optional[str] = None
    abstract_snippet: str = ""
    rrf_score: float = Field(
        description=(
            "Reciprocal Rank Fusion score used to order results. This is an "
            "ordinal fusion score, not a relevance probability, and is not "
            "comparable across different queries or candidate set sizes. "
            "Use `rank` for the result's position in this result set."
        )
    )
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    contribution: Optional[str] = None
    relevance_justification: Optional[str] = None
    relevance_score: Optional[float] = None
    specificity_score: Optional[float] = None


class RetrievalTrace(BaseModel):
    total_corpus_size: int
    keyword_filtered_size: int
    reduction_percent_after_keyword_filter: float
    generated_keywords: List[str] = []
    hyde_document: str = ""
    dense_latency_seconds: float = 0.0
    bm25_latency_seconds: float = 0.0
    total_latency_seconds: float = 0.0


class RetrievalDebugInfo(BaseModel):
    keyword_candidate_ids: Optional[List[str]] = None
    final_candidate_ids: List[str] = []
    dense_results: List[Dict[str, Any]] = []
    dense_results_raw_query: Optional[List[Dict[str, Any]]] = None
    bm25_results: List[Dict[str, Any]] = []
    bm25_delta_results: List[Dict[str, Any]] = []


class SearchResponse(BaseModel):
    query: str
    results: List[PaperResult] = []
    trace: RetrievalTrace
    metadata: Dict[str, Any] = {}
    debug: Optional[RetrievalDebugInfo] = None
