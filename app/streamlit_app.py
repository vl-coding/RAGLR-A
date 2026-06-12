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

    remaining = max(0, QUERY_LIMIT - st.session_state.queries_used)
    window_label = f"{WINDOW_SECONDS // 3600}h" if WINDOW_SECONDS >= 3600 else f"{WINDOW_SECONDS // 60}m"

    if remaining > 0:
        st.caption(f"**Free queries remaining:** {remaining} / {QUERY_LIMIT} (resets every {window_label})")
    else:
        st.error(LIMIT_MESSAGE)

    top_k = config["retrieval"]["default_top_k"]
    use_qwen = st.checkbox("Qwen keyword prefilter", value=True)
    use_justification = st.checkbox("Claude justifications", value=True)

    st.markdown(f"**This demo returns the top {top_k} papers per query.**")

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

st.caption("Searches typically take around 2 minutes to complete — please be patient after clicking search.")

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
    else:
        # Load pipeline on first use
        if st.session_state.pipeline is None:
            with st.spinner("Loading pipeline (first load takes ~30s) ..."):
                st.session_state.pipeline = RagLiteraturePipeline(config)

        progress_bar = st.progress(0, text="Starting search ...")

        def _update_progress(step: str, fraction: float) -> None:
            progress_bar.progress(fraction, text=step)

        response = st.session_state.pipeline.run(
            query=query,
            top_k=top_k,
            use_qwen_prefilter=use_qwen,
            use_claude_justification=use_justification,
            progress_callback=_update_progress,
        )

        progress_bar.empty()

        st.session_state.queries_used += 1

        # ---------------------------------------------------------------
        # Retrieval trace
        # ---------------------------------------------------------------
        st.subheader("How the search narrowed down 3M+ papers")
        c1, c2 = st.columns(2)
        c1.metric("Total corpus", f"{response.trace.total_corpus_size:,}")
        c2.metric(
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

        if len(response.results) < top_k:
            st.warning(
                f"Only {len(response.results)} of {top_k} papers were found for this query.\n\n"
                "**Why this happens:** the search narrows 3M+ arXiv papers down step by step "
                "(keywords and semantic similarity). If your topic is very "
                "specific, niche, or uses uncommon terminology, fewer papers may survive "
                "every filtering step.\n\n"
                "**Tips to get more results:**\n"
                "- Use more general search terms (e.g. \"transformer time series forecasting\" "
                "instead of \"sparse attention transformers for irregular multivariate clinical time series\").\n"
                "- Try rephrasing with synonyms or related terminology.\n"
                "- Turn off the Qwen keyword prefilter — it narrows candidates before semantic "
                "search and can be too aggressive for niche topics.\n\n"
                "**Tip for your literature review:** a narrow niche often doesn't have many "
                "papers written directly on it — but that's normal for original research. "
                "Try running follow-up searches on the broader techniques, methods, or "
                "application areas your topic builds on (e.g. the general method you're "
                "adapting, the broader problem domain, or a related application area). "
                "Papers from these adjacent searches are often the ones you'll cite to "
                "justify your approach and show how your work fits into the existing "
                "literature."
            )

        for paper in response.results:
            with st.container():
                st.markdown(f"#### {paper.rank}. {paper.title}")
                col_a, col_b = st.columns([2, 2])
                col_a.caption(f"**Year:** {paper.year}")
                col_b.caption(
                    f"**Fusion score:** {paper.rrf_score:.4f}",
                    help=(
                        "Reciprocal Rank Fusion score — an ordinal value used "
                        "to order these results, not a relevance probability. "
                        "Not comparable across different searches. Use the "
                        "result's position (#{}) for ranking.".format(paper.rank)
                    ),
                )

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
