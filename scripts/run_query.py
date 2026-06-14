import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

from src.rag_lit.config import load_config
from src.rag_lit.pipeline import RagLiteraturePipeline


def main():
    # Avoid UnicodeEncodeError on terminals (e.g. Windows cp1252) when
    # results contain non-ASCII characters such as accented author names.
    sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-qwen", action="store_true")
    parser.add_argument("--no-justification", action="store_true")

    args = parser.parse_args()

    config = load_config()
    pipeline = RagLiteraturePipeline(config)

    response = pipeline.run(
        query=args.query,
        top_k=args.top_k,
        use_qwen_prefilter=not args.no_qwen,
        use_claude_justification=not args.no_justification,
    )

    output_path = Path(config["paths"]["outputs_dir"]) / "latest_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(response.model_dump(), f, indent=2)

    print(response.model_dump_json(indent=2))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()