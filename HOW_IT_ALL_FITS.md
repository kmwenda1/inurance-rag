# HOW IT ALL FITS TOGETHER
## A visual walkthrough of every file, every connection, and every decision

---

## The Big Picture

When a user asks a question, here is exactly what happens, step by step:

```
User uploads a PDF
       │
       ▼
  [ app.py ]
  Saves file to disk
       │
       ▼
  [ rag/ingest.py ]
  Opens the PDF with pdfplumber
  Extracts text page by page
  Cuts text into overlapping chunks (600 chars, 120 char overlap)
  Returns: list of chunk dicts
       │
       ▼
  [ rag/embedder.py ]
  Loads BAAI/bge-large-en-v1.5 embedding model
  Converts each chunk's text into a list of ~1024 numbers (a vector)
  Stores those vectors + metadata in ChromaDB (persisted to disk)
       │
       ▼
       │ ← (document is now indexed. User can now ask questions.)
       │
User types a question
       │
       ▼
  [ app.py ]
  Receives the question string
       │
       ▼
  [ rag/retriever.py ]
  Embeds the question using the same model
  Asks ChromaDB: "which stored vectors are closest to this question vector?"
  Gets top 5 most similar chunks back
  Assesses confidence (HIGH / MEDIUM / LOW based on similarity score)
  Returns: chunks + confidence
       │
       ▼
  [ rag/generator.py ]
  Builds a prompt that includes:
    - system instructions (be precise, cite sources, don't hallucinate)
    - the 5 retrieved chunks as context
    - the user's question
  Sends to Groq API (Llama 3.1 70B model)
  Receives the answer
  Returns: answer + sources used
       │
       ▼
  [ app.py ]
  Displays the answer
  Shows confidence badge
  Shows source passages (expandable)
  Stores in session history
```

---

## File-by-File Breakdown

### `app.py` — The Orchestrator
- **Imports from:** `rag.ingest`, `rag.embedder`, `rag.retriever`, `rag.generator`
- **Responsibility:** UI, user interaction, calling the right module at the right time
- **Knows about:** Everything — it's the entry point
- **Does NOT handle:** PDF parsing, embedding math, API calls (delegates all of that)

### `rag/ingest.py` — The Reader
- **Imports from:** `pdfplumber`, `os`
- **Responsibility:** Open PDFs, extract text per page, create overlapping chunks
- **Knows about:** Files on disk, PDF structure
- **Does NOT handle:** Embeddings, database, AI — pure file processing only
- **Key output:** List of chunk dicts with text, page_num, source, chunk_id

### `rag/embedder.py` — The Memory Builder
- **Imports from:** `sentence_transformers`, `chromadb`
- **Responsibility:** Load embedding model, convert text to vectors, store in ChromaDB
- **Knows about:** The embedding model, ChromaDB
- **Does NOT handle:** PDF reading, question answering, UI
- **Key output:** Vectors stored in ChromaDB on disk

### `rag/retriever.py` — The Searcher
- **Imports from:** `rag.embedder` (reuses the same model instance)
- **Responsibility:** Embed the question, search ChromaDB, return top chunks with scores
- **Knows about:** The embedding model, ChromaDB
- **Does NOT handle:** PDF reading, answer generation
- **Key output:** List of chunk dicts sorted by similarity + confidence assessment

### `rag/generator.py` — The Writer
- **Imports from:** `groq`, `dotenv`
- **Responsibility:** Build the prompt, call Groq API, return the answer
- **Knows about:** Groq API, how to structure prompts
- **Does NOT handle:** Document reading, searching, UI
- **Key output:** Answer string + source list

---

## Why Each Module Is Separate

This is the single most important architectural decision in Phase 1.

Each module does ONE thing. This matters because:

**Debugging:** If the answer is wrong, you check generator.py.
If wrong chunks are being returned, you check retriever.py.
If the PDF isn't being read properly, you check ingest.py.
You never have to read the whole codebase to fix one problem.

**Swapping parts:** Want to switch from ChromaDB to Qdrant? Only embedder.py
and retriever.py change. The rest of the code stays identical.

Want to switch from Groq to OpenAI? Only generator.py changes.
Want to add Excel support? You add a new excel_ingest.py and plug it into app.py.
Nothing else breaks.

**Testing:** You can test each module independently. Run ingest.py on a PDF
and print the chunks. Run retriever.py with a fake question and print what
it finds. You don't need the full app running to test one component.

---

## The Data Flow (with types)

```
File on disk (str: file_path)
       ↓
ingest_pdf(file_path) → dict:
    {
        "chunks": [
            {
                "text":     "Flood damage is excluded...",  # str
                "page_num": 12,                             # int
                "source":   "home_policy.pdf",             # str
                "chunk_id": "home_policy.pdf_page12_chunk0" # str (unique ID)
            },
            ...  # typically 30-200 chunks per document
        ],
        "page_count":  24,   # int
        "chunk_count": 87,   # int
        "is_scanned":  False # bool
    }
       ↓
store_chunks(chunks) → int (number of chunks stored)
    [Chunks are now in ChromaDB on disk]
       ↓
       ↓ ← (later, when user asks a question)
       ↓
retrieve_with_confidence(query) → dict:
    {
        "chunks": [
            {
                "text":     "...",
                "page_num": 12,
                "source":   "home_policy.pdf",
                "score":    0.87   # float: higher = more relevant
            },
            ...  # top 5
        ],
        "confidence": "HIGH",         # str: HIGH / MEDIUM / LOW / NONE
        "top_score":  0.87,           # float
        "warning":    None            # str or None
    }
       ↓
generate_answer(query, chunks) → dict:
    {
        "answer":  "Based on page 12 of home_policy.pdf, flood damage...",
        "sources": [...],  # same chunks as above
        "model":   "llama-3.1-70b-versatile",
        "error":   None    # str or None
    }
       ↓
Displayed in Streamlit UI
```

---

## The Prompt Template (what the AI actually receives)

Understanding the prompt is critical. Here is the exact structure:

```
SYSTEM MESSAGE (sets the AI's behaviour):
    "You are a precise document assistant for an insurance company.
     Only use information from the provided context.
     Cite the source for every statement.
     If not found, say so. Do not guess."

USER MESSAGE (the actual query + context):
    "Here are the relevant excerpts from the company documents:

    [Source: home_policy_2024.pdf, Page 12]
    Flood damage is excluded from coverage under section 4.2 of this policy.
    The policyholder acknowledges that standard policies do not...

    ---

    [Source: home_policy_2024.pdf, Page 8]
    Coverage is provided for accidental water damage caused by burst pipes...

    Based ONLY on the above excerpts, please answer this question:

    Is flood damage covered under this policy?"

AI RESPONSE:
    "Based on page 12 of home_policy_2024.pdf, flood damage is explicitly
     excluded from coverage under section 4.2. However, as noted on page 8,
     accidental water damage from burst pipes IS covered..."
```

The AI is not "thinking" from training data. It's reading the text we gave it
and summarising it. That's the entire mechanism of RAG.

---

## Common Questions While Reading the Code

**Q: Why is EMBEDDING_MODEL loaded at module level in embedder.py?**
A: Loading an ML model takes 3-10 seconds. Module-level loading means it
   happens once when the app starts. If it were inside a function, it would
   reload every time that function is called — multiple times per question.

**Q: Why does retriever.py import from embedder.py?**
A: Both retrieval and storage use the same embedding model. If we created two
   separate model instances, we'd use twice the memory. Importing the same
   instance from embedder.py means one model, shared by both operations.

**Q: Why do chunks overlap?**
A: Insurance policy sentences often continue across paragraph breaks. A clause
   that starts at position 580 of a chunk that ends at 600 would be cut off.
   The 120-character overlap means that clause appears fully in both the chunk
   that starts it and the one that follows.

**Q: Why temperature=0 in generator.py?**
A: LLMs have a randomness setting (temperature). At temperature 0, the model
   picks the single most probable next word at each step — making it
   deterministic and conservative. For insurance documents, we want precision
   over creativity. Same question = same answer every time.

**Q: What is a similarity score and why does 0.87 mean "good"?**
A: After normalisation, vectors have values between -1 and 1.
   Two identical sentences score 1.0. Completely unrelated text scores
   near 0 or negative. In practice:
   - 0.80+  = very strong match
   - 0.65+  = good match
   - 0.55+  = moderate match
   - Below  = likely not relevant

**Q: Why ChromaDB and not just a regular database?**
A: A regular database (SQLite, PostgreSQL) searches by exact values.
   "Find rows where text = 'flood damage'" would find nothing if the
   document says "water ingress from storms." ChromaDB stores vectors
   and finds the mathematically closest ones — which corresponds to
   finding the most semantically similar content.

---

## What Phase 2 Adds (and where it plugs in)

```
Phase 1 (what you have now)              Phase 2 additions
─────────────────────────────────────────────────────────────────────
ingest.py (PDFs only)              →    + excel_ingest.py
embedder.py (store chunks)         →    + reranker.py (rerank after retrieval)
retriever.py (vector search)       →    + hybrid_retriever.py (vector + keyword)
generator.py (Groq API)            →    + router.py (structured vs semantic)
app.py (single pipeline)           →    + multi-source routing in app.py
```

Every addition is additive. You don't rewrite Phase 1. You extend it.
```

---

## Running the App

```bash
# 1. Clone / navigate to the project
cd insurance-rag

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Open .env and replace 'your_groq_api_key_here' with your real key
# Get one free at: https://console.groq.com

# 5. Run the app
streamlit run app.py

# 6. Open in browser
# Streamlit will print a URL like: http://localhost:8501
```

---

*This guide is part of Phase 1 of the Insurance Document Intelligence System.*
*Phase 2 adds Excel support, hybrid search, reranking, and routing.*
