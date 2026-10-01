# ─────────────────────────────────────────────────────────
# rag/embedder.py — Embedding & Vector Database Management
#
# WHAT THIS FILE DOES:
#   Takes chunks of text and converts them into numbers that represent
#   their meaning, then stores them in a searchable database.
#
# THE PROCESS:
#   chunk text → embedding model → list of ~1000 numbers (a vector)
#             → stored in ChromaDB → searchable by meaning
#
# WHY THIS EXISTS AS A SEPARATE FILE:
#   The embedding model is heavy (~500MB). We load it once here and
#   reuse it everywhere. If we loaded it in every file that needs it,
#   we'd waste memory and time.
# ─────────────────────────────────────────────────────────

import os

# Disable ChromaDB telemetry — stops harmless "Failed to send telemetry" warnings
# ChromaDB tries to send anonymous usage stats; this just turns that off.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from sentence_transformers import SentenceTransformer
import os

# ── Load the embedding model once at module level ─────────────────────────────
#
# WHY 'BAAI/bge-large-en-v1.5'?
#   It's consistently top-ranked on the MTEB leaderboard (a standard
#   benchmark for embedding models). It handles technical/legal language
#   well — critical for insurance documents. It runs locally, meaning
#   no API call and no cost per embedding.
#
# WHY LOAD AT MODULE LEVEL (not inside a function)?
#   Loading an ML model takes 3-10 seconds. If we loaded it inside a
#   function, it would reload every time that function is called.
#   Loading at module level means it loads once when the app starts,
#   then stays in memory ready to use instantly.
#
# NOTE: First run will download the model (~1.3GB). Subsequent runs use
# the cached version and are instant.

print("Loading embedding model... (first run downloads ~90MB, then it's cached)")
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.")


# ── Set up ChromaDB ────────────────────────────────────────────────────────────
#
# WHY PersistentClient?
#   PersistentClient saves the database to disk (our data/chroma_db/ folder).
#   If we used the in-memory client, the entire database would be lost
#   every time we restart the app. Persistence means documents you indexed
#   yesterday are still there today.
#
# WHY ChromaDB?
#   It runs entirely on your machine — no cloud account, no API key,
#   no cost. For Phase 1, it's perfect. When we scale to production,
#   we swap this for Qdrant (same interface, just a different backend).

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
CHROMA_CLIENT = chromadb.PersistentClient(path=DB_PATH)

# The name of our collection (like a table in a regular database)
COLLECTION_NAME = "insurance_documents"


# ── Core functions ─────────────────────────────────────────────────────────────

def get_collection():
    """
    Gets (or creates if it doesn't exist) our document collection in ChromaDB.

    WHY get_or_create?
        First time the app runs, the collection doesn't exist — so we create it.
        Every time after that, we just get the existing one.
        This way the same code works for both first run and all future runs.
    """
    return CHROMA_CLIENT.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Insurance document chunks with embeddings"}
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Converts a list of text strings into a list of embedding vectors.

    Each embedding is a list of ~1024 floats (numbers between -1 and 1).
    These numbers represent the MEANING of the text in mathematical form.

    Two sentences with similar meaning will produce embeddings that are
    mathematically 'close' to each other. That's how we search by meaning.

    Example:
        embed_texts(["flood damage is covered"])
        → [[0.023, -0.412, 0.889, ...]]  (1024 numbers)

    WHY normalize_embeddings=True?
        Normalization puts all embeddings on the same scale, making
        distance comparisons more accurate. bge models specifically
        recommend this setting.
    """
    embeddings = EMBEDDING_MODEL.encode(
        texts,
        normalize_embeddings=True,  # recommended for bge models
        show_progress_bar=len(texts) > 20  # show progress for large batches
    )
    return embeddings.tolist()


def store_chunks(chunks: list[dict]) -> int:
    """
    Takes a list of chunk dicts (from ingest.py) and stores them in ChromaDB.

    Each chunk is stored with:
        - its text (the actual words)
        - its embedding (the meaning-numbers)
        - its metadata (page number, source filename, chunk_id)

    Returns the number of chunks successfully stored.

    WHY STORE METADATA?
        The metadata is what lets us say "this answer came from page 12
        of home_policy_2024.pdf". Without metadata, we'd have the answer
        but no citation. Citations are what make this trustworthy.

    WHY BATCH IN GROUPS OF 100?
        ChromaDB handles large inserts better in batches. Also, if something
        fails halfway through, we've already stored the first batches safely.
    """
    collection = get_collection()

    # Separate the components ChromaDB needs
    ids        = [chunk["chunk_id"] for chunk in chunks]
    texts      = [chunk["text"]     for chunk in chunks]
    metadatas  = [
        {
            "page_num": chunk["page_num"],
            "source":   chunk["source"],
        }
        for chunk in chunks
    ]

    # Generate embeddings for all chunks
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    # Insert in batches of 100
    batch_size = 100
    stored = 0

    for i in range(0, len(chunks), batch_size):
        batch_end = i + batch_size
        collection.add(
            ids=ids[i:batch_end],
            documents=texts[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
        stored += len(ids[i:batch_end])

    return stored


def chunk_already_exists(chunk_id: str) -> bool:
    """
    Checks if a chunk with this ID is already stored in the database.

    WHY THIS EXISTS:
        If a user uploads the same PDF twice, we don't want to store
        duplicate chunks. Duplicates would cause the AI to return the
        same text multiple times in its context, wasting the context
        window and potentially biasing answers.
    """
    collection = get_collection()
    result = collection.get(ids=[chunk_id])
    return len(result["ids"]) > 0


def get_collection_stats() -> dict:
    """
    Returns basic statistics about the current database state.
    Used by the UI to show the user what's been indexed.
    """
    collection = get_collection()
    count = collection.count()
    return {
        "total_chunks": count,
        "collection_name": COLLECTION_NAME
    }
