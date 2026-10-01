# ─────────────────────────────────────────────────────────
# rag/retriever.py — Finding Relevant Chunks
#
# WHAT THIS FILE DOES:
#   Takes a user's question, converts it to an embedding (meaning-numbers),
#   and finds the most similar chunks stored in ChromaDB.
#
# THE PROCESS:
#   user question → embed it → search ChromaDB → return top matching chunks
#
# ANALOGY:
#   Think of it like a library card catalogue — but instead of searching
#   by exact title or keyword, you search by the MEANING of what you want.
#   "flood damage exclusion" finds chunks about "water damage not covered"
#   even though the words are different.
# ─────────────────────────────────────────────────────────

from rag.embedder import EMBEDDING_MODEL, get_collection


# ── Main retrieval function ────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 8) -> list[dict]:
    """
    Finds the most relevant document chunks for a given question.

    Parameters:
        query:  the user's question (plain English)
        top_k:  how many chunks to return (default 5)

    Returns a list of result dicts:
        [
            {
                "text":     "Flood damage is excluded under clause 4.2...",
                "page_num": 12,
                "source":   "home_policy_2024.pdf",
                "score":    0.87   # similarity score: higher = more relevant
            },
            ...
        ]

    HOW IT WORKS STEP BY STEP:
        1. We embed the query — convert it to meaning-numbers
        2. ChromaDB compares those numbers to every stored chunk's numbers
        3. It returns the chunks whose numbers are closest (most similar meaning)
        4. We format those results cleanly and return them

    WHY top_k=5?
        5 chunks is usually enough context for the LLM to answer well
        without overwhelming it with too much text. Too few = missing info.
        Too many = diluted context with irrelevant passages. 5 is the sweet spot
        for most insurance document queries.
    """
    collection = get_collection()

    # Check if database has any documents
    if collection.count() == 0:
        return []

    # Embed the user's query
    # WHY embed the query?
    # Because our stored chunks are embeddings (meaning-numbers).
    # To compare the query against the chunks, we need them in the same
    # number format. It's like translating everything into the same language.
    query_embedding = EMBEDDING_MODEL.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Search ChromaDB for similar chunks
    # include= tells ChromaDB what data to return alongside the chunk IDs
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),  # can't ask for more than we have
        include=["documents", "metadatas", "distances"]
    )

    # ChromaDB returns results wrapped in extra lists (because it supports
    # batch queries). We queried one query, so we take index [0].
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Format into clean, readable dicts
    chunks = []
    for i in range(len(documents)):
        # ChromaDB returns distances (lower = more similar).
        # We convert to a similarity score (higher = more relevant) for clarity.
        similarity_score = 1 - distances[i]

        chunks.append({
            "text":     documents[i],
            "page_num": metadatas[i].get("page_num", "?"),
            "source":   metadatas[i].get("source", "unknown"),
            "score":    round(similarity_score, 3)
        })

    # Sort by score descending (best match first)
    chunks.sort(key=lambda x: x["score"], reverse=True)

    return chunks


def retrieve_with_confidence(query: str, top_k: int = 8) -> dict:
    """
    Same as retrieve(), but also returns a confidence assessment.

    WHY THIS MATTERS:
        If the best matching chunk has a very low similarity score,
        it means the database doesn't contain relevant information
        about this query. In that case, we should warn the user
        rather than letting the LLM hallucinate an answer.

    Confidence levels:
        HIGH   (score > 0.75): Good match found, answer should be reliable
        MEDIUM (score > 0.55): Decent match, verify the answer
        LOW    (score ≤ 0.55): Poor match, database may not cover this topic

    Returns:
        {
            "chunks":     [...],      # list of retrieved chunks
            "confidence": "HIGH",     # confidence assessment
            "top_score":  0.87,       # score of the best match
            "warning":    None        # or a warning message if confidence is low
        }
    """
    chunks = retrieve(query, top_k)

    if not chunks:
        return {
            "chunks": [],
            "confidence": "NONE",
            "top_score": 0,
            "warning": "No documents have been indexed yet. Please upload a PDF first."
        }

    top_score = chunks[0]["score"]

    if top_score > 0.25:
        confidence = "HIGH"
        warning = None
    elif top_score > 0.15:
        confidence = "MEDIUM"
        warning = "⚠️ Moderate confidence — verify against the source document."
    else:
        confidence = "LOW"
        warning = "⚠️ Low confidence. This topic may not be covered in the uploaded documents."

    return {
        "chunks":     chunks,
        "confidence": confidence,
        "top_score":  top_score,
        "warning":    warning
    }
