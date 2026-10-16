# Architecture Overview

## Runtime flow

1. User uploads a PDF in Streamlit (`app.py`).
2. `rag.app_utils` validates upload size/type and creates a safe storage path.
3. `rag.ingest` extracts text via `pdfplumber`, detects likely scanned PDFs, and chunks text.
4. `rag.embedder` creates embeddings via FastEmbed and stores vectors+metadata in ChromaDB.
5. On a question, `rag.retriever` embeds query and retrieves top chunks from ChromaDB.
6. `rag.generator` sends retrieved excerpts + question to Groq and returns an answer.
7. UI renders answer, confidence level, and source passages.

## Components

- **UI:** Streamlit (`app.py`)
- **Ingestion:** `rag/ingest.py`
- **Storage/Embedding:** `rag/embedder.py`
- **Retrieval:** `rag/retriever.py`
- **Generation:** `rag/generator.py`
- **Helpers:** `rag/app_utils.py`

## Data persistence

- Uploaded files: `data/uploaded/`
- Vector DB: `data/chroma_db/`

## Safety/reliability choices

- Lazy initialization for Groq and FastEmbed to avoid startup dependency on API key/network.
- Conservative confidence thresholds (`HIGH >= 0.75`, `MEDIUM >= 0.55`).
- User-safe error messages (no raw provider tracebacks in UI).
- No unsafe HTML rendering for model output/source text.
