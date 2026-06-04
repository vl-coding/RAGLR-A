from .config import ensure_project_dirs, load_config
from .schemas import Paper, PaperResult, RetrievalTrace, SearchResponse

__all__ = [
    "load_config",
    "ensure_project_dirs",
    "Paper",
    "PaperResult",
    "RetrievalTrace",
    "SearchResponse",
]


def get_pipeline(config: dict = None):
    """Lazy import to avoid loading heavy ML libraries at package import time."""
    from .pipeline import RagLiteraturePipeline  # noqa: PLC0415

    if config is None:
        config = load_config()
    return RagLiteraturePipeline(config)
