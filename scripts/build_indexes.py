from src.rag_lit.config import load_config, ensure_project_dirs
from src.rag_lit.data_ingestion import load_papers_jsonl, write_manifest
from src.rag_lit.dense_retriever import DenseRetriever
from src.rag_lit.bm25_retriever import BM25Retriever
from src.rag_lit.keyword_index import build_keyword_inverted_index, save_keyword_index


def main():
    config = load_config()
    ensure_project_dirs(config)

    papers = load_papers_jsonl(config["data"]["processed_path"])

    print(f"Loaded {len(papers)} papers.")

    print("Building dense index...")
    dense = DenseRetriever(
        model_name=config["models"]["embedding_model"],
        persist_dir=config["paths"]["dense_index_dir"],
    )
    dense.build_index(papers)

    print("Building BM25 index...")
    bm25 = BM25Retriever()
    bm25.build_index(papers)
    bm25.save(config["paths"]["bm25_index"])

    print("Building keyword inverted index...")
    keyword_index = build_keyword_inverted_index(papers)
    save_keyword_index(keyword_index, config["paths"]["keyword_index"])

    all_categories = sorted({cat for paper in papers for cat in paper.categories})

    write_manifest(
        path=config["paths"]["manifest"],
        num_papers=len(papers),
        categories=all_categories,
        min_year=config["data"]["min_year"],
        embedding_model=config["models"]["embedding_model"],
    )

    print("Finished building indexes.")


if __name__ == "__main__":
    main()