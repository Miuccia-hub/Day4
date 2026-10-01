import os
from pathlib import Path

import streamlit as st

from rag_core import (
    EvidenceStore,
    build_due_diligence_report,
    change_control_review,
    generate_ai_report,
    generate_deep_analysis_report,
)


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Canvassian DealScope", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_store() -> EvidenceStore:
    store = EvidenceStore(ROOT)
    store.ensure_index()
    return store


st.title("Canvassian DealScope")
st.caption("Evidence-grounded buyer-side M&A due diligence • every finding links back to supplied text")

with st.sidebar:
    st.header("Index")
    if st.button("Rebuild index", use_container_width=True):
        with st.spinner("Reading, chunking and indexing the corpus…"):
            get_store().ensure_index(force=True)
        st.success("Index rebuilt")
    store = get_store()
    stats = store.stats()
    st.metric("Documents", stats["documents"])
    st.metric("Searchable chunks", stats["chunks"])
    st.caption(f"Backend: {stats['backend']}")
    st.warning("The brief says 1,000 documents; only 750 were supplied.")
    st.divider()
    st.subheader("Optional AI synthesis")
    entered_api_key = st.text_input(
        "OpenAI API key",
        type="password",
        help="Kept only in this browser session; alternatively set OPENAI_API_KEY as a Codespaces secret.",
    )
    api_key = entered_api_key or os.getenv("OPENAI_API_KEY", "")
    model = st.selectbox(
        "Report model",
        ["gpt-4o", "gpt-4o-mini", "gpt-5-pro"],
        help=(
        "GPT-4o is the recommended default; "
        "GPT-4o Mini is cheaper; "
        "GPT-5 Pro is strongest but may take several minutes."
    ),
)

report_tab, search_tab, clauses_tab, about_tab = st.tabs(
    ["Board report", "Evidence search", "Control clauses", "Methodology"]
)

with report_tab:
    st.subheader("Preliminary Board report")
    if "report" not in st.session_state:
        st.session_state.report = build_due_diligence_report(store)
        st.session_state.report_mode = "Local evidence report"
    local_col, ai_col, deep_col = st.columns(3)
    if local_col.button("Regenerate local report", use_container_width=True):
        with st.spinner("Running the risk workplan…"):
            st.session_state.report = build_due_diligence_report(store)
            st.session_state.report_mode = "Local evidence report"
    if ai_col.button("Generate enhanced AI report", use_container_width=True, disabled=not api_key):
        try:
            with st.spinner(f"Synthesising the retrieved evidence with {model}…"):
                st.session_state.report = generate_ai_report(store, api_key=api_key, model=model)
                st.session_state.report_mode = f"AI-enhanced report ({model})"
        except Exception as exc:
            st.error(f"AI generation failed: {exc}")
            st.info("The local evidence report remains available; check the API key, model access and billing.")
    if deep_col.button("Deep analysis", use_container_width=True, disabled=not api_key):
        progress_bar = st.progress(0.0)
        progress_text = st.empty()

        def update_progress(current: int, total: int, message: str) -> None:
            progress_bar.progress(min(current / max(total, 1), 1.0))
            progress_text.caption(message)

        try:
            st.warning("Deep analysis makes multiple API calls and may take several minutes.")
            st.session_state.report, deep_stats = generate_deep_analysis_report(
                store,
                api_key=api_key,
                synthesis_model=model,
                extraction_model="gpt-4o-mini",
                progress_callback=update_progress,
            )
            st.session_state.report_mode = f"Deep analysis ({model})"
            st.session_state.deep_stats = deep_stats
            st.success(
                f"Reviewed {deep_stats['chunks_reviewed']} chunks in {deep_stats['batches']} batches; "
                f"extracted {deep_stats['findings']} findings."
            )
        except Exception as exc:
            st.error(f"Deep analysis failed: {exc}")
            st.info("The previous report remains available. Check the API key, model access, billing and rate limits.")
    if not api_key:
        st.caption("Add an API key in the sidebar to enable AI synthesis. Local search and reporting still work without it.")
    else:
        st.caption("Only the locally selected evidence pack is sent to the API, not the full 750-document corpus.")
    st.success(f"Current output: {st.session_state.report_mode}")
    if st.session_state.get("deep_stats") and st.session_state.report_mode.startswith("Deep analysis"):
        st.caption(
            f"Deep-analysis audit: {st.session_state.deep_stats['chunks_reviewed']} chunks • "
            f"{st.session_state.deep_stats['batches']} extraction batches • "
            f"{st.session_state.deep_stats['findings']} structured findings"
        )
    st.markdown(st.session_state.report)
    st.download_button(
        "Download report (.md)", st.session_state.report,
        file_name="canvassian_due_diligence_report.md", mime="text/markdown",
    )

with search_tab:
    st.subheader("Search all evidence")
    query = st.text_input("Question or search terms", placeholder="e.g. What evidence supports PayWise financial distress?")
    left, right = st.columns([1, 1])
    category_label = left.selectbox("Document type", ["All", "Board papers", "Contracts", "Emails"])
    top_k = right.slider("Results", 3, 20, 8)
    mapping = {"Board papers": "board_papers", "Contracts": "contracts", "Emails": "emails"}
    if query:
        results = store.search(query, top_k=top_k, category=mapping.get(category_label))
        for rank, result in enumerate(results, 1):
            with st.expander(f"{rank}. {result.chunk.title} — score {result.score:.3f}", expanded=rank <= 3):
                st.caption(f"{result.citation} • {result.chunk.category} • {result.chunk.date or 'date not parsed'}")
                st.markdown(result.chunk.text)

with clauses_tab:
    st.subheader("Six major clients — change-in-control triage")
    rows = change_control_review(ROOT)
    st.dataframe(
        [{k: v for k, v in row.items() if k not in {"sources", "evidence"}} for row in rows],
        use_container_width=True, hide_index=True,
    )
    st.info("ADVERSE = express change-in-control language found. REVIEW = no express match, but termination-for-convenience language found. NOT FOUND is not a legal clearance.")
    for row in rows:
        if row["evidence"]:
            with st.expander(f"{row['client']} clause evidence"):
                for evidence in row["evidence"]:
                    st.write(evidence)

with about_tab:
    st.subheader("How it works")
    st.markdown(
        """
1. Every supplied `.txt` file is read with source, type, title, date and major-client metadata.
2. Documents are split into overlapping, line-addressable chunks.
3. ChromaDB stores deterministic local embeddings; search then reranks results using exact legal-language overlap.
4. A fixed M&A risk workplan retrieves evidence for the instructed risks and other material issues.
5. The report quotes the evidence and retains file/line citations. No external API or confidential-data upload is required.

Optionally, the retrieved evidence pack can be sent to the OpenAI Responses API for higher-quality synthesis.
The API is never used for primary retrieval or clause detection, and the local report remains available as a fallback.

Deep analysis broadens retrieval across ten diligence themes, extracts structured findings in bounded batches with
GPT-4o Mini, then uses the selected report model for a final deduplicated synthesis. It makes multiple API calls.

This is rapid document triage. It does not determine privilege, authenticity, execution status, completeness or legal effect.
        """
    )

