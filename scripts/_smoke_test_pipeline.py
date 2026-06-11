import time

from src.rag_lit.config import load_config
from src.rag_lit.pipeline import RagLiteraturePipeline

config = load_config()

t0 = time.time()
pipeline = RagLiteraturePipeline(config)
print(f"Pipeline init took {time.time() - t0:.1f}s", flush=True)

# Exercise the new custom_categories override path (mirrors streamlit subcategory selection)
t0 = time.time()
response = pipeline.run(
    query="transformer architectures for time series forecasting",
    selected_fields=["all"],
    top_k=3,
    use_qwen_prefilter=True,
    use_claude_justification=False,
    custom_categories=["cs.LG", "stat.ML"],
)
print(f"Custom-category run took {time.time() - t0:.1f}s", flush=True)
print("field_filtered_size:", response.trace.field_filtered_size)
print("keyword_filtered_size:", response.trace.keyword_filtered_size)
print("generated_keywords:", response.trace.generated_keywords[:5])
for r in response.results:
    print(r.rank, r.title, r.categories)

# Second call without qwen prefilter, default fields, to check the lazy-loaded
# qwen instance isn't required and field filter path still works
t0 = time.time()
response2 = pipeline.run(
    query="graph neural networks for molecule property prediction",
    selected_fields=["all"],
    top_k=2,
    use_qwen_prefilter=False,
    use_claude_justification=False,
)
print(f"No-qwen run took {time.time() - t0:.1f}s", flush=True)
for r in response2.results:
    print(r.rank, r.title, r.categories)

print("SMOKE TEST OK")
