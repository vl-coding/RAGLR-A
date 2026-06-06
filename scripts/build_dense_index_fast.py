"""
Chunked dense index builder with ONNX Runtime backend.

Streams the JSONL in chunks so RAM stays bounded (~1-2 GB peak instead of
loading all 2.68M papers at once), encodes each chunk via ONNX Runtime
(2-3x faster than PyTorch CPU), and upserts to the ChromaDB persistent store.

Crash-safe: upserts are idempotent by arxiv_id, so a restart resumes from
whatever is already in ChromaDB (chunks that were upserted survive).

After this completes, run:  python scripts/build_indexes.py --skip-dense

Usage:
    python scripts/build_dense_index_fast.py
    python scripts/build_dense_index_fast.py --backend torch --batch-size 128
"""

import argparse
import sys
import time
from typing import Generator, List

from src.rag_lit.config import load_config, ensure_project_dirs
from src.rag_lit.schemas import Paper

_DEFAULT_CHUNK_SIZE = 50_000
_DEFAULT_BATCH_SIZE = 128


def stream_papers(path: str, chunk_size: int) -> Generator[List[Paper], None, None]:
    chunk: List[Paper] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk.append(Paper.model_validate_json(line))
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _print(*args, **kwargs) -> None:
    print(*args, **kwargs, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=_DEFAULT_CHUNK_SIZE)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--backend",
        default="onnx",
        choices=["onnx", "torch"],
        help="onnx uses ONNX Runtime (faster on CPU); torch uses PyTorch",
    )
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    import chromadb

    config = load_config()
    ensure_project_dirs(config)

    jsonl_path = config["data"]["processed_path"]
    model_name = config["models"]["embedding_model"]
    persist_dir = config["paths"]["dense_index_dir"]

    _print(f"Model:        {model_name}")
    _print(f"Backend:      {args.backend}")
    _print(f"JSONL:        {jsonl_path}")
    _print(f"ChromaDB dir: {persist_dir}")
    _print(f"Chunk size:   {args.chunk_size:,} papers")
    _print()

    _print("Counting papers in JSONL …")
    total_papers = count_lines(jsonl_path)
    _print(f"Total papers: {total_papers:,}")
    _print()

    backend_kwargs = {"backend": args.backend} if args.backend != "torch" else {}
    _print(f"Loading model (backend={args.backend}) — ONNX export takes ~1 min on first run …")
    model = SentenceTransformer(model_name, **backend_kwargs)

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("arxiv_papers")

    already_indexed = collection.count()
    if already_indexed > 0:
        _print(f"ChromaDB already has {already_indexed:,} vectors — will upsert (deduped).")

    processed = 0
    start_wall = time.time()

    for chunk_idx, chunk in enumerate(stream_papers(jsonl_path, args.chunk_size)):
        texts = [p.text for p in chunk]
        ids = [p.arxiv_id for p in chunk]
        metadatas = [
            {
                "title": p.title,
                "year": p.year,
                "categories": ",".join(p.categories),
                "arxiv_id": p.arxiv_id,
            }
            for p in chunk
        ]

        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

        processed += len(chunk)
        elapsed = time.time() - start_wall
        rate = processed / elapsed
        remaining = total_papers - processed
        eta_h = (remaining / rate / 3600) if rate > 0 else 0

        _print(
            f"Chunk {chunk_idx + 1:>3}: {processed:>10,}/{total_papers:,} "
            f"({100 * processed / total_papers:.1f}%)  "
            f"{rate:.0f} papers/sec  ETA {eta_h:.1f}h"
        )

    total_time = time.time() - start_wall
    _print()
    _print(f"Done. {processed:,} papers indexed in {total_time / 3600:.2f}h.")
    _print(f"ChromaDB collection size: {collection.count():,} vectors.")


if __name__ == "__main__":
    main()
