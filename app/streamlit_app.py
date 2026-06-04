import os

import streamlit as st
from dotenv import load_dotenv

from src.rag_lit.config import load_config
from src.rag_lit.pipeline import RagLiteraturePipeline

load_dotenv()

st.set_page_config(
    page_title="RAG Literature Review Assistant",
    layout="wide",
)

st.title("RAG Literature Review Assistant")
st.write(
    "Search arXiv papers using multi-field filtering, Qwen keyword prefiltering, "
    "Claude HyDE, SBERT + BM25 hybrid retrieval, RRF ranking, and Claude justifications."
)

config = load_config()

field_options = {
    field_data["label"]: field_key
    for field_key, field_data in config["academic_fields"].items()
}

selected_labels = st.multiselect(
    "Choose one or more academic fields",
    options=list(field_options.keys()),
    default=["All arXiv Fields"],
)

selected_fields = [field_options[label] for label in selected_labels]

all_category_codes = sorted(
    cat
    for field_data in config["academic_fields"].values()
    for cat in (
        field_data["categories"]
        if isinstance(field_data.get("categories"), list)
        else []
    )
)

selected_categories = st.multiselect(
    "Optional: restrict to specific arXiv subcategories (overrides field selection if set)",
    options=all_category_codes,
    default=[],
)

query = st.text_area("Enter your research question or partial draft")

top_k = st.slider(
    "How many papers should be returned?",
    min_value=1,
    max_value=50,
    value=10,
)

use_qwen = st.checkbox("Use Qwen keyword prefilter", value=True)
use_justification = st.checkbox("Use Claude justifications", value=True)

if "pipeline" not in st.session_state:
    with st.spinner("Loading pipeline..."):
        st.session_state.pipeline = RagLiteraturePipeline(config)

if st.button("Find Papers"):
    if not query.strip():
        st.warning("Please enter a query.")
    elif not selected_fields and not selected_categories:
        st.warning("Please select at least one academic field or subcategory.")
    else:
        # If specific subcategories are chosen, use a synthetic field config that
        # contains only those codes so the pipeline can filter correctly.
        if selected_categories:
            run_config = dict(config)
            run_config["academic_fields"] = {
                "_custom": {
                    "label": "Custom selection",
                    "categories": selected_categories,
                }
            }
            run_fields = ["_custom"]
        else:
            run_config = config
            run_fields = selected_fields if selected_fields else ["all"]

        with st.spinner("Searching papers..."):
            response = st.session_state.pipeline.run(
                query=query,
                selected_fields=run_fields,
                top_k=top_k,
                use_qwen_prefilter=use_qwen,
                use_claude_justification=use_justification,
            )

        st.subheader("Search Trace")
        col1, col2, col3 = st.columns(3)

        col1.metric("Total corpus", response.trace.total_corpus_size)
        col2.metric("After field filter", response.trace.field_filtered_size)
        col3.metric("After keyword filter", response.trace.keyword_filtered_size)

        st.write(
            f"Reduction after keyword filter: "
            f"**{response.trace.reduction_percent_after_keyword_filter}%**"
        )

        st.write("Generated keywords:")
        st.write(response.trace.generated_keywords)

        st.subheader("Recommended Papers")

        for paper in response.results:
            st.markdown(f"### {paper.rank}. {paper.title}")
            st.write(f"**Year:** {paper.year}")
            st.write(f"**Categories:** {', '.join(paper.categories)}")
            st.write(f"**RRF Score:** {paper.rrf_score:.4f}")

            if paper.relevance_justification:
                st.write(f"**Why relevant:** {paper.relevance_justification}")

            st.write(paper.abstract_snippet)

            if paper.url:
                st.markdown(f"[View on arXiv]({paper.url})")

            st.write("---")

        st.download_button(
            label="Download JSON Results",
            data=response.model_dump_json(indent=2),
            file_name="rag_literature_results.json",
            mime="application/json",
        )
