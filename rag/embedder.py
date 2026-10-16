import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from fastembed import TextEmbedding

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
CHROMA_CLIENT = chromadb.PersistentClient(path=DB_PATH)
COLLECTION_NAME = "insurance_documents"
EMBEDDING_MODEL = None


def get_embedding_model() -> TextEmbedding:
    """Lazily initialize the embedding model to avoid startup-time downloads."""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return EMBEDDING_MODEL

def get_collection():
    return CHROMA_CLIENT.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Insurance document chunks with embeddings"}
    )

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    # fastembed returns a generator of numpy arrays
    embeddings_gen = get_embedding_model().embed(texts)
    return [emb.tolist() for emb in embeddings_gen]

def store_chunks(chunks: list[dict]) -> int:
    if not chunks:
        return 0

    collection = get_collection()
    deduped_chunks = list({chunk["chunk_id"]: chunk for chunk in chunks}.values())
    ids        = [chunk["chunk_id"] for chunk in deduped_chunks]
    texts      = [chunk["text"]     for chunk in deduped_chunks]
    metadatas  = [
        {
            "page_num": chunk["page_num"],
            "source":   chunk["source"],
        }
        for chunk in deduped_chunks
    ]
    embeddings = embed_texts(texts)
    batch_size = 100
    stored = 0
    for i in range(0, len(deduped_chunks), batch_size):
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
    collection = get_collection()
    result = collection.get(ids=[chunk_id])
    return len(result["ids"]) > 0

def get_collection_stats() -> dict:
    collection = get_collection()
    count = collection.count()
    return {
        "total_chunks": count,
        "collection_name": COLLECTION_NAME
    }
