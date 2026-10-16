# inurance-rag (Prototype)

Streamlit Retrieval-Augmented Generation (RAG) assistant for insurance PDF documents.

- **Entry point:** `streamlit run app.py`
- **Embedding:** FastEmbed (`BAAI/bge-small-en-v1.5`)
- **Vector store:** ChromaDB (local persistent DB)
- **LLM provider:** Groq (`qwen/qwen3.8-27b`)
- **Status:** company-ready **prototype** (not production-hardened)

## Supported Python

Tested with **Python 3.11** (see `runtime.txt`).

## Repository layout

```text
inurance-rag/
├── app.py
├── rag/
│   ├── app_utils.py
│   ├── ingest.py
│   ├── embedder.py
│   ├── retriever.py
│   └── generator.py
├── tests/
├── .streamlit/config.toml
├── requirements.txt
├── packages.txt
└── ARCHITECTURE.md
```

## Local setup

```bash
cd /home/runner/work/inurance-rag/inurance-rag
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set GROQ_API_KEY in .env
```

Optional environment variables:

- `GROQ_API_KEY` (required only when asking questions)
- `MAX_UPLOAD_SIZE_MB` (default: `20`)

## Run

```bash
streamlit run app.py
```

The app can start without `GROQ_API_KEY`; a key is required only when an answer is requested.

## Tests and checks

```bash
ruff check .
pytest -q
```

## Deployment notes (Streamlit Cloud)

- `packages.txt` is included for required system packages.
- `runtime.txt` pins Python 3.11.
- Store `GROQ_API_KEY` as a deployment secret.

## Privacy and data flow warning

- Uploaded PDFs are stored locally under `data/uploaded/`.
- Chunks and embeddings are stored in local ChromaDB under `data/chroma_db/`.
- Retrieved text excerpts are sent to Groq to generate answers.
- Do **not** upload sensitive documents unless this data flow is acceptable for your policy/regulatory requirements.

## Known limitations / non-production caveats

- No authentication or multi-tenant isolation.
- No OCR pipeline for scanned image PDFs.
- Retrieval quality depends on extracted text quality and chunking.
- Not a substitute for legal/compliance review.

## Additional docs

- Architecture: `ARCHITECTURE.md`
- Security policy: `SECURITY.md`
- Contribution guide: `CONTRIBUTING.md`
- Changes: `CHANGELOG.md`

## License

No license file is currently included in this repository. If your organization needs reuse/redistribution rights, add an explicit license before external distribution.
