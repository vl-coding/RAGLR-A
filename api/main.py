import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag_lit.config import load_config
from src.rag_lit.pipeline import RagLiteraturePipeline
from src.rag_lit.schemas import SearchResponse

load_dotenv()

app = FastAPI(
    title="RAG Literature Review Assistant",
    description="Hybrid arXiv retrieval using Qwen keyword prefiltering, Claude HyDE, SBERT + BM25, and RRF.",
    version="1.0.0",
)

_pipeline: Optional[RagLiteraturePipeline] = None


def get_pipeline() -> RagLiteraturePipeline:
    global _pipeline
    if _pipeline is None:
        config = load_config()
        _pipeline = RagLiteraturePipeline(config)
    return _pipeline


class SearchRequest(BaseModel):
    query: str
    selected_fields: List[str] = ["all"]
    top_k: int = 10
    use_qwen_prefilter: bool = True
    use_claude_justification: bool = True


@app.on_event("startup")
async def startup_event() -> None:
    get_pipeline()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    if request.top_k not in (5, 10, 15, 20, 25):
        raise HTTPException(status_code=400, detail="top_k must be one of 5, 10, 15, 20, 25")

    pipeline = get_pipeline()

    return pipeline.run(
        query=request.query,
        selected_fields=request.selected_fields,
        top_k=request.top_k,
        use_qwen_prefilter=request.use_qwen_prefilter,
        use_claude_justification=request.use_claude_justification,
    )


@app.get("/fields")
async def list_fields() -> dict:
    config = load_config()
    return {
        key: data["label"]
        for key, data in config["academic_fields"].items()
    }
