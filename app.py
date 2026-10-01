# ─────────────────────────────────────────────────────────
# app.py — The Main Streamlit UI
#
# WHAT THIS FILE DOES:
#   This is the face of the application — the thing the user sees
#   and interacts with. It ties together all the other modules:
#   ingest.py → embedder.py → retriever.py → generator.py
#
# HOW STREAMLIT WORKS:
#   Streamlit re-runs this entire file from top to bottom every time
#   the user interacts with anything (clicks a button, uploads a file,
#   types a question). It uses st.session_state to persist data between
#   those re-runs.
#
# RUN WITH:
#   streamlit run app.py
# ─────────────────────────────────────────────────────────

import os
import streamlit as st

# Import our RAG modules
from rag.ingest import ingest_pdf
from rag.embedder import store_chunks, chunk_already_exists, get_collection_stats
from rag.retriever import retrieve_with_confidence
from rag.generator import generate_answer, format_sources

# ── Page configuration ─────────────────────────────────────────────────────────
# Must be the first Streamlit call in the file
st.set_page_config(
    page_title="Insurance Document Assistant",
    page_icon="📋",
    layout="wide",          # use full screen width
    initial_sidebar_state="expanded"
)


# ── Custom styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #4DA6FF;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #aaa;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #1a2a3a;
        border-left: 4px solid #4DA6FF;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #E8E8E8 !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .answer-box * { color: #E8E8E8 !important; }
    .source-box {
        background-color: #1e1e2e;
        border: 1px solid #333;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #ccc !important;
    }
    .source-box * { color: #ccc !important; }
    .confidence-high   { color: #0d9b76; font-weight: 600; }
    .confidence-medium { color: #e6a817; font-weight: 600; }
    .confidence-low    { color: #dc3545; font-weight: 600; }
    .stTextInput > div > div > input {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────────
#
# WHY SESSION STATE?
#   Streamlit re-runs the script on every interaction. Without session_state,
#   variables would reset to their defaults on every re-run.
#   session_state persists data across re-runs within the same browser session.

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []  # list of filenames that have been indexed

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {question, answer, sources, confidence}


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📁 Document Library")
    st.markdown("Upload PDFs to build your searchable knowledge base.")
    st.divider()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Supported: text-based PDFs. Scanned PDFs may have lower accuracy."
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name

        # Check if this file has already been indexed in this session
        if file_name in st.session_state.indexed_files:
            st.info(f"✅ **{file_name}** is already indexed.")
        else:
            if st.button(f"📥 Index '{file_name}'", use_container_width=True):
                # ── File processing pipeline ──────────────────────────────────
                #
                # This is where all our modules connect:
                # 1. Save the uploaded file to disk
                # 2. ingest_pdf() extracts text and creates chunks
                # 3. store_chunks() embeds and stores them in ChromaDB
                #
                with st.spinner("Reading and indexing document..."):
                    # Step 1: Save file to disk
                    upload_dir = os.path.join("data", "uploaded")
                    os.makedirs(upload_dir, exist_ok=True)
                    file_path = os.path.join(upload_dir, file_name)

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Step 2: Ingest (extract text + chunk)
                    st.markdown("📄 Reading PDF...")
                    ingest_result = ingest_pdf(file_path)

                    # Warn if document appears to be scanned
                    if ingest_result["is_scanned"]:
                        st.warning(
                            "⚠️ This PDF appears to be a scanned image. "
                            "Text extraction may be incomplete. "
                            "For best results, use digitally-created PDFs."
                        )

                    if ingest_result["chunk_count"] == 0:
                        st.error("❌ No text could be extracted from this PDF.")
                    else:
                        # Step 3: Embed and store
                        st.markdown("🧠 Embedding chunks...")
                        new_chunks = [
                            c for c in ingest_result["chunks"]
                            if not chunk_already_exists(c["chunk_id"])
                        ]

                        if new_chunks:
                            stored = store_chunks(new_chunks)
                            st.session_state.indexed_files.append(file_name)
                            st.success(
                                f"✅ **{file_name}** indexed!\n\n"
                                f"• {ingest_result['page_count']} pages read\n"
                                f"• {stored} new chunks stored"
                            )
                        else:
                            st.info("This document (or all its chunks) is already in the database.")
                            st.session_state.indexed_files.append(file_name)

    # Show indexed files
    st.divider()
    st.markdown("### Indexed Documents")
    stats = get_collection_stats()
    st.metric("Total chunks in database", stats["total_chunks"])

    if st.session_state.indexed_files:
        for fname in st.session_state.indexed_files:
            st.markdown(f"• 📄 {fname}")
    else:
        st.markdown("*No documents indexed yet.*")

    # Clear database button
    st.divider()
    if st.button("🗑️ Clear Database", use_container_width=True, type="secondary"):
        # This deletes all stored chunks — use with caution
        import chromadb
        db_path = os.path.join("data", "chroma_db")
        client = chromadb.PersistentClient(path=db_path)
        try:
            client.delete_collection("insurance_documents")
        except Exception:
            pass
        st.session_state.indexed_files = []
        st.session_state.chat_history = []
        st.success("Database cleared.")
        st.rerun()


# ── Main content area ──────────────────────────────────────────────────────────

st.markdown('<div class="main-header">📋 Insurance Document Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your policy documents in plain English. Every answer cites its source.</div>', unsafe_allow_html=True)

# Check if any documents are indexed before showing the query box
stats = get_collection_stats()
if stats["total_chunks"] == 0:
    st.info(
        "👈 **Get started:** Upload a PDF using the sidebar on the left, "
        "then click 'Index' to make it searchable. Once indexed, you can ask "
        "questions about it here."
    )
else:
    # ── Question input ──────────────────────────────────────────────────────────
    st.markdown("### Ask a Question")

    # Example questions to help users get started
    example_questions = [
        "What does this policy cover?",
        "What are the exclusions?",
        "What is the claims process?",
        "What is the maximum coverage amount?",
        "Is flood damage covered?",
    ]

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Your question:",
            placeholder="e.g. Is flood damage covered under this policy?",
            label_visibility="collapsed"
        )
    with col2:
        ask_button = st.button("Ask →", use_container_width=True, type="primary")

    # Example question pills
    st.markdown("**Try an example:**")
    example_cols = st.columns(len(example_questions))
    for i, example in enumerate(example_questions):
        with example_cols[i]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                query = example
                ask_button = True

    # ── Process the question ────────────────────────────────────────────────────

    if ask_button and query:
        with st.spinner("Searching documents and generating answer..."):

            # Step 1: Retrieve relevant chunks
            # retrieve_with_confidence() returns chunks AND a confidence level
            retrieval_result = retrieve_with_confidence(query, top_k=5)
            chunks = retrieval_result["chunks"]
            confidence = retrieval_result["confidence"]
            warning = retrieval_result["warning"]

            # Step 2: Generate answer from those chunks
            generation_result = generate_answer(query, chunks)
            answer = generation_result["answer"]
            sources = generation_result["sources"]

        # Store in chat history
        st.session_state.chat_history.insert(0, {
            "question":   query,
            "answer":     answer,
            "sources":    sources,
            "confidence": confidence,
            "warning":    warning
        })

    # ── Display chat history ────────────────────────────────────────────────────

    if st.session_state.chat_history:
        st.divider()
        st.markdown("### Results")

        for i, entry in enumerate(st.session_state.chat_history):
            # Question
            st.markdown(f"**❓ {entry['question']}**")

            # Confidence badge
            conf = entry["confidence"]
            if conf == "HIGH":
                st.markdown('<span class="confidence-high">● High confidence</span>', unsafe_allow_html=True)
            elif conf == "MEDIUM":
                st.markdown('<span class="confidence-medium">● Medium confidence</span>', unsafe_allow_html=True)
            elif conf == "LOW":
                st.markdown('<span class="confidence-low">● Low confidence</span>', unsafe_allow_html=True)

            # Warning if confidence is low
            if entry["warning"]:
                st.warning(entry["warning"])

            # Answer
            st.markdown(
                f'<div class="answer-box">{entry["answer"]}</div>',
                unsafe_allow_html=True
            )

            # Sources expander
            if entry["sources"]:
                with st.expander(f"📄 View source passages ({len(entry['sources'])} found)"):
                    for j, source in enumerate(entry["sources"]):
                        st.markdown(
                            f'<div class="source-box">'
                            f'<strong>{source["source"]} — Page {source["page_num"]}</strong>'
                            f'<br><em>Relevance score: {source["score"]}</em>'
                            f'<hr style="margin: 0.5rem 0">'
                            f'{source["text"]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        if j < len(entry["sources"]) - 1:
                            st.markdown("")

            # Divider between questions (except last one)
            if i < len(st.session_state.chat_history) - 1:
                st.divider()

    # ── Clear history button ────────────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("Clear conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
