from typing import Any, Dict, List, Optional

from pydantic import BaseModel


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
    rrf_score: float
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    contribution: Optional[str] = None
    relevance_justification: Optional[str] = None
    relevance_score: Optional[float] = None
    specificity_score: Optional[float] = None


class RetrievalTrace(BaseModel):
    total_corpus_size: int
    field_filtered_size: int
    keyword_filtered_size: int
    reduction_percent_after_field_filter: float
    reduction_percent_after_keyword_filter: float
    generated_keywords: List[str] = []
    selected_fields: List[str] = []
    hyde_document: str = ""
    dense_latency_seconds: float = 0.0
    bm25_latency_seconds: float = 0.0
    total_latency_seconds: float = 0.0


class SearchResponse(BaseModel):
    query: str
    results: List[PaperResult] = []
    trace: RetrievalTrace
    metadata: Dict[str, Any] = {}
