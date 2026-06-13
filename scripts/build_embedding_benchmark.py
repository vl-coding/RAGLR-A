"""
Build a fixed-size benchmark subset of the corpus and two ChromaDB indexes
(the current embedding model and `all-mpnet-base-v2`) over it, for comparing
embedding models on the gold query set (issue #12) without a full corpus
re-embed.

The subset always includes every `relevant_ids` paper referenced in
tests/eval/gold_queries.yaml, plus a random sample of the rest of the corpus
(default 50,000 papers) so the benchmark indexes have a realistic
"needle in haystack" candidate pool.

Usage:
    python scripts/build_embedding_benchmark.py
    python scripts/build_embedding_benchmark.py --sample-size 50000 --seed 42
    python scripts/build_embedding_benchmark.py --skip-subset   # reuse existing subset file
"""
import argparse
import random
import sys
import time
from pathlib import Path
from typing import Generator, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_lit.config import load_config, ensure_project_dirs
from src.rag_lit.eval_metrics import load_gold_queries
from src.rag_lit.schemas import Paper

_DEFAULT_SAMPLE_SIZE = 50_000
_DEFAULT_SEED = 42
_BENCHMARK_SUBSET_PATH = "data/processed/embedding_benchmark_subset.jsonl"
_MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"
_GOLD_QUERIES_PATH = "tests/eval/gold_queries.yaml"
_CHROMA_MAX = 5_461  # chromadb Rust backend hard limit


def _print(*args, **kwargs) -> None:
    print(*args, **kwargs, flush=True)


def _collect_relevant_ids(gold_queries_path: str) -> Set[str]:
    ids: Set[str] = set()
    for q in load_gold_queries(gold_queries_path):
        ids.update(q.get("relevant_ids", []))
    return ids


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def build_subset(jsonl_path: str, output_path: str, relevant_ids: Set[str], sample_size: int, seed: int) -> None:
    rng = random.Random(seed)

    _print("Counting papers in source JSONL ...")
    total = count_lines(jsonl_path)
    _print(f"Total papers: {total:,}")

    sample_prob = sample_size / total if total else 0.0

    found_relevant: Set[str] = set()
    random_sampled = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            paper = Paper.model_validate_json(line)

            if paper.arxiv_id in relevant_ids:
                found_relevant.add(paper.arxiv_id)
            elif rng.random() < sample_prob:
                random_sampled += 1
            else:
                continue

            out.write(line + "\n")

    missing = relevant_ids - found_relevant
    _print(f"Relevant ids found: {len(found_relevant)}/{len(relevant_ids)}")
    if missing:
        _print(f"  Missing from corpus: {sorted(missing)}")
    _print(f"Random papers sampled: {random_sampled:,}")
    _print(f"Subset written: {output_path} ({len(found_relevant) + random_sampled:,} papers)")


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


def build_index_for_model(
    model_name: str,
    persist_dir: str,
    subset_path: str,
    batch_size: int,
    chunk_size: int,
    label: str,
) -> dict:
    from sentence_transformers import SentenceTransformer
    import chromadb

    _print(f"\n=== Building benchmark index: {label} ({model_name}) ===")
    _print(f"Loading model {model_name} ...")
    model = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("arxiv_papers")

    total = count_lines(subset_path)
    processed = 0
    start = time.time()

    for chunk in stream_papers(subset_path, chunk_size):
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
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        for i in range(0, len(ids), _CHROMA_MAX):
            sl = slice(i, i + _CHROMA_MAX)
            collection.upsert(
                ids=ids[sl],
                documents=texts[sl],
                embeddings=embeddings[sl],
                metadatas=metadatas[sl],
            )

        processed += len(chunk)
        elapsed = time.time() - start
        rate = processed / elapsed if elapsed else 0.0
        _print(f"  {processed:,}/{total:,} ({100 * processed / total:.1f}%)  {rate:.1f} papers/sec")

    total_time = time.time() - start
    rate = processed / total_time if total_time else 0.0
    _print(f"Done: {processed:,} papers in {total_time:.1f}s ({rate:.1f} papers/sec)")
    _print(f"ChromaDB collection size: {collection.count():,} vectors.")

    return {
        "label": label,
        "model": model_name,
        "persist_dir": persist_dir,
        "papers": processed,
        "seconds": round(total_time, 1),
        "papers_per_sec": round(rate, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=_DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--skip-subset",
        action="store_true",
        help=f"Reuse the existing subset file at {_BENCHMARK_SUBSET_PATH} instead of resampling.",
    )
    args = parser.parse_args()

    config = load_config()
    ensure_project_dirs(config)

    jsonl_path = config["data"]["processed_path"]
    subset_path = _BENCHMARK_SUBSET_PATH

    if args.skip_subset:
        _print(f"Reusing existing subset: {subset_path}")
    else:
        relevant_ids = _collect_relevant_ids(_GOLD_QUERIES_PATH)
        _print(f"Gold query relevant_ids to preserve: {len(relevant_ids)}")
        build_subset(jsonl_path, subset_path, relevant_ids, args.sample_size, args.seed)

    results = [
        build_index_for_model(
            config["models"]["embedding_model"],
            "artifacts/benchmark_index_minilm",
            subset_path,
            args.batch_size,
            args.chunk_size,
            "minilm",
        ),
        build_index_for_model(
            _MPNET_MODEL,
            "artifacts/benchmark_index_mpnet",
            subset_path,
            args.batch_size,
            args.chunk_size,
            "mpnet",
        ),
    ]

    _print("\n=== Build summary ===")
    for r in results:
        _print(f"{r['label']:8s} {r['model']:40s} {r['papers']:>8,} papers  {r['seconds']:>8.1f}s  {r['papers_per_sec']:>6.1f} papers/sec")


if __name__ == "__main__":
    main()
