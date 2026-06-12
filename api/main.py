import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.rag_lit.config import load_config
from src.rag_lit.pipeline import RagLiteraturePipeline
from src.rag_lit.rate_limiter import RateLimiter
from src.rag_lit.schemas import SearchResponse

load_dotenv()

config = load_config()

app = FastAPI(
    title="RAG Literature Review Assistant",
    description="Hybrid arXiv retrieval using Qwen keyword prefiltering, Claude HyDE, SBERT + BM25, and RRF.",
    version="1.0.0",
)

_pipeline: Optional[RagLiteraturePipeline] = None
_limiter: Optional[RateLimiter] = None


def get_pipeline() -> RagLiteraturePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagLiteraturePipeline(config)
    return _pipeline


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        demo = config.get("demo", {})
        _limiter = RateLimiter(
            max_requests=demo.get("rate_limit_requests", 10),
            window_seconds=demo.get("rate_limit_window_seconds", 3600),
        )
    return _limiter


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_qwen_prefilter: bool = True
    use_claude_justification: bool = True


@app.on_event("startup")
async def startup_event() -> None:
    get_pipeline()
    get_limiter()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/limit")
async def limit_status(request: Request) -> dict:
    limiter = get_limiter()
    demo = config.get("demo", {})
    ip = client_ip(request)
    return {
        "remaining": limiter.remaining(ip),
        "limit": demo.get("rate_limit_requests", 10),
        "window_seconds": demo.get("rate_limit_window_seconds", 3600),
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: Request, body: SearchRequest) -> SearchResponse:
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    if body.top_k not in (5, 10, 15, 20, 25):
        raise HTTPException(status_code=400, detail="top_k must be one of 5, 10, 15, 20, 25")

    limiter = get_limiter()
    ip = client_ip(request)

    if not limiter.is_allowed(ip):
        demo = config.get("demo", {})
        limit = demo.get("rate_limit_requests", 10)
        message = demo.get("limit_message", "Free demo limit reached — {limit} queries per hour.").format(limit=limit)
        retry_after = limiter.retry_after(ip)
        headers = {"Retry-After": str(retry_after)} if retry_after else {}
        return JSONResponse(
            status_code=429,
            content={"detail": message, "retry_after_seconds": retry_after},
            headers=headers,
        )

    pipeline = get_pipeline()
    return pipeline.run(
        query=body.query,
        top_k=body.top_k,
        use_qwen_prefilter=body.use_qwen_prefilter,
        use_claude_justification=body.use_claude_justification,
    )
