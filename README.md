# Insurance Document Assistant — Phase 1

Ask questions about your policy documents in plain English.
Every answer comes with the exact source it came from.

---

## What This Does

Upload any insurance PDF → Ask questions in plain English → Get cited answers

Built for insurance companies that spend too much time searching through
policy documents and claims sheets for answers that should take seconds.

---

## Quick Start

### 1. Get a free Groq API key
Go to [console.groq.com](https://console.groq.com) — no credit card needed.
Copy your API key.

### 2. Set up the project

```bash
# Navigate to the project folder
cd insurance-rag

# Create a virtual environment (keeps dependencies isolated)
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Open .env in any text editor and paste your Groq API key
```

### 3. Run the app

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## Project Structure

```
insurance-rag/
├── app.py                ← Run this. It's the UI and main entry point.
├── rag/
│   ├── __init__.py       ← Makes 'rag' importable as a Python package
│   ├── ingest.py         ← Reads PDFs, extracts text, creates chunks
│   ├── embedder.py       ← Converts text to meaning-numbers, stores in DB
│   ├── retriever.py      ← Searches the DB for relevant chunks
│   └── generator.py      ← Sends chunks + question to AI, returns answer
├── data/
│   ├── uploaded/         ← Your uploaded PDFs are saved here
│   └── chroma_db/        ← The vector database lives here (auto-created)
├── HOW_IT_ALL_FITS.md    ← Read this to understand how everything connects
├── .env.example          ← Copy to .env and add your API key
├── .env                  ← Your actual keys — NEVER share or commit this
└── requirements.txt      ← All Python libraries this project needs
```

---

## How It Works (Plain English)

1. **You upload a PDF** — the system reads every page
2. **It cuts the text into small pieces** (chunks) and converts each piece into numbers that represent its meaning
3. **Those numbers are stored** in a local database on your computer
4. **You ask a question** — the question is also converted to meaning-numbers
5. **The database finds** the 5 chunks whose meaning is closest to your question
6. **Those 5 chunks are sent** to an AI along with your question and a strict instruction: "Only answer from what's in these chunks, and cite your sources"
7. **The AI returns** a precise answer with page citations

---

## Technology Stack

| Component | Tool | Why |
|-----------|------|-----|
| UI | Streamlit | Python-native web UI, no JS needed |
| PDF reading | pdfplumber | Better table extraction than pypdf |
| Embeddings | sentence-transformers (bge-large) | Top-ranked free model, runs locally |
| Vector database | ChromaDB | Runs locally, persistent, zero setup |
| LLM | Groq + Llama 3.1 70B | Fast, free tier, great instruction following |

**Total cost to run: $0** (within Groq free tier limits)

---

## Phase 2 (Coming Next)

- Excel file support (premium tables, claims data)
- Hybrid search (vector + keyword)
- Reranking (smarter result selection)
- Query routing (auto-detect if question needs spreadsheet data)

---

## Files to Read (in order)

1. `HOW_IT_ALL_FITS.md` — understand the architecture before touching code
2. `rag/ingest.py` — start here, it's the simplest module
3. `rag/embedder.py` — how text becomes searchable vectors
4. `rag/retriever.py` — how questions find their answers
5. `rag/generator.py` — how the AI writes the final answer
6. `app.py` — how it all connects into one working app
