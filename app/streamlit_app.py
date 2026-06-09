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

config = load_config()
demo_cfg = config.get("demo", {})
QUERY_LIMIT = demo_cfg.get("rate_limit_requests", 10)
WINDOW_SECONDS = demo_cfg.get("rate_limit_window_seconds", 3600)
LIMIT_MESSAGE = demo_cfg.get(
    "limit_message",
    "Free demo limit reached — {limit} queries per hour. Full access coming soon."
).format(limit=QUERY_LIMIT)

# Session state init
if "queries_used" not in st.session_state:
    st.session_state.queries_used = 0
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

# -----------------------------------------------------------------------
# Sidebar — branding + query counter
# -----------------------------------------------------------------------
with st.sidebar:
    st.title("RAG Literature Review Assistant")
    st.caption(
        "Hybrid arXiv search using Qwen keyword prefiltering, "
        "Claude HyDE, SBERT + BM25 retrieval, and RRF ranking."
    )

    st.divider()

    remaining = max(0, QUERY_LIMIT - st.session_state.queries_used)
    window_label = f"{WINDOW_SECONDS // 3600}h" if WINDOW_SECONDS >= 3600 else f"{WINDOW_SECONDS // 60}m"

    if remaining > 0:
        st.metric("Free queries remaining", f"{remaining} / {QUERY_LIMIT}")
        st.caption(f"Limit resets every {window_label} per session.")
    else:
        st.error(LIMIT_MESSAGE)

    st.divider()

    selected_labels = st.multiselect(
        "Academic fields",
        options=[d["label"] for d in config["academic_fields"].values()],
        default=["All arXiv Fields"],
    )

    field_options = {d["label"]: k for k, d in config["academic_fields"].items()}
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
        "Restrict to specific arXiv subcategories (optional)",
        options=all_category_codes,
        default=[],
    )

    top_k = st.select_slider(
        "Results to return",
        options=[5, 10, 15, 20, 25],
        value=10,
    )
    use_qwen = st.checkbox("Qwen keyword prefilter", value=True)
    use_justification = st.checkbox("Claude justifications", value=True)

# -----------------------------------------------------------------------
# Main area — query input + results
# -----------------------------------------------------------------------
st.header("Search arXiv Literature")

query = st.text_area(
    "Enter your research question",
    placeholder="e.g. What are the latest methods for time series forecasting with transformers?",
    height=100,
)

limit_reached = st.session_state.queries_used >= QUERY_LIMIT

search_button = st.button(
    "Find Papers",
    disabled=limit_reached,
    type="primary",
)

if limit_reached and not search_button:
    st.warning(LIMIT_MESSAGE)

if search_button:
    if not query.strip():
        st.warning("Please enter a research question.")
    elif not selected_fields and not selected_categories:
        st.warning("Please select at least one academic field or subcategory.")
    else:
        # Load pipeline on first use
        if st.session_state.pipeline is None:
            with st.spinner("Loading pipeline (first load takes ~30s) ..."):
                st.session_state.pipeline = RagLiteraturePipeline(config)

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

        with st.spinner("Searching 3M+ arXiv papers ..."):
            response = st.session_state.pipeline.run(
                query=query,
                selected_fields=run_fields,
                top_k=top_k,
                use_qwen_prefilter=use_qwen,
                use_claude_justification=use_justification,
            )

        st.session_state.queries_used += 1

        # ---------------------------------------------------------------
        # Retrieval trace
        # ---------------------------------------------------------------
        st.subheader("How the search narrowed down 3M+ papers")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total corpus", f"{response.trace.total_corpus_size:,}")
        c2.metric(
            "After field filter",
            f"{response.trace.field_filtered_size:,}",
            delta=f"-{response.trace.reduction_percent_after_field_filter:.0f}%",
            delta_color="off",
        )
        c3.metric(
            "After keyword filter",
            f"{response.trace.keyword_filtered_size:,}",
            delta=f"-{response.trace.reduction_percent_after_keyword_filter:.0f}%",
            delta_color="off",
        )

        if response.trace.generated_keywords:
            st.caption("Keywords used: " + ", ".join(f"`{k}`" for k in response.trace.generated_keywords))

        st.divider()

        # ---------------------------------------------------------------
        # Results
        # ---------------------------------------------------------------
        st.subheader(f"Top {len(response.results)} papers")

        for paper in response.results:
            with st.container():
                st.markdown(f"#### {paper.rank}. {paper.title}")
                col_a, col_b, col_c = st.columns([2, 2, 3])
                col_a.caption(f"**Year:** {paper.year}")
                col_b.caption(f"**RRF score:** {paper.rrf_score:.4f}")
                col_c.caption(f"**Categories:** {', '.join(paper.categories)}")

                if paper.relevance_justification:
                    st.info(f"**Why relevant:** {paper.relevance_justification}")

                if paper.contribution:
                    st.caption(f"**Contribution:** {paper.contribution}")

                scores = []
                if paper.relevance_score is not None:
                    scores.append(f"Relevance: {paper.relevance_score}/10")
                if paper.specificity_score is not None:
                    scores.append(f"Specificity: {paper.specificity_score}/10")
                if scores:
                    st.caption(" · ".join(scores))

                st.write(paper.abstract_snippet + ("..." if len(paper.abstract_snippet) >= 500 else ""))

                if paper.url:
                    st.markdown(f"[View on arXiv →]({paper.url})")

                st.divider()

        st.download_button(
            label="Download results as JSON",
            data=response.model_dump_json(indent=2),
            file_name="literature_results.json",
            mime="application/json",
        )

        # Update sidebar counter without rerun
        remaining_after = max(0, QUERY_LIMIT - st.session_state.queries_used)
        if remaining_after == 0:
            st.sidebar.error(LIMIT_MESSAGE)
        else:
            st.sidebar.metric("Free queries remaining", f"{remaining_after} / {QUERY_LIMIT}")
